"""Training-only class-aware optimal transport for MoPE prompt tokens."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sinkhorn_from_cost(
    cost: torch.Tensor,
    epsilon: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an entropic OT plan and one transport distance per batch item."""
    if epsilon <= 0 or iterations < 1:
        raise ValueError("Sinkhorn epsilon must be positive and iterations must be at least one.")
    kernel = torch.exp((-cost / epsilon).clamp_min(-50.0))
    row_mass = torch.full(
        cost.shape[:2], 1.0 / cost.shape[1], dtype=cost.dtype, device=cost.device
    )
    col_mass = torch.full(
        (cost.shape[0], cost.shape[2]),
        1.0 / cost.shape[2], dtype=cost.dtype, device=cost.device,
    )
    left = torch.ones_like(row_mass)
    right = torch.ones_like(col_mass)
    for _ in range(iterations):
        left = row_mass / torch.bmm(kernel, right.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
        right = col_mass / torch.bmm(
            kernel.transpose(1, 2), left.unsqueeze(-1)
        ).squeeze(-1).clamp_min(1e-8)
    plan = left.unsqueeze(-1) * kernel * right.unsqueeze(1)
    return plan, (plan * cost).sum(dim=(1, 2))


def pairwise_token_ot(
    eeg_tokens: torch.Tensor,
    prompt_tokens: torch.Tensor,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    """Return D[i,j] = OT(EEG_i, Prompt_j) for all pairs in a minibatch."""
    if eeg_tokens.ndim != 3 or prompt_tokens.ndim != 3:
        raise ValueError("Expected EEG and prompt tokens with shape [B,tokens,dimension].")
    if eeg_tokens.shape[0] != prompt_tokens.shape[0] or eeg_tokens.shape[2] != prompt_tokens.shape[2]:
        raise ValueError(
            "EEG and prompt tokens must share batch size and embedding dimension; got "
            f"{tuple(eeg_tokens.shape)} and {tuple(prompt_tokens.shape)}."
        )
    batch, eeg_count, dimension = eeg_tokens.shape
    prompt_count = prompt_tokens.shape[1]
    sources = eeg_tokens[:, None].expand(-1, batch, -1, -1).reshape(
        batch * batch, eeg_count, dimension
    )
    targets = prompt_tokens[None].expand(batch, -1, -1, -1).reshape(
        batch * batch, prompt_count, dimension
    )
    sources = F.normalize(sources.float(), dim=-1)
    targets = F.normalize(targets.float(), dim=-1)
    cost = (1.0 - torch.bmm(sources, targets.transpose(1, 2))).clamp_min(0.0)
    _, distance = sinkhorn_from_cost(cost, epsilon, iterations)
    return distance.reshape(batch, batch)


def _directional_class_aware_contrast(
    distance: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = -distance / temperature
    batch = target.numel()
    identity = torch.eye(batch, dtype=torch.bool, device=target.device)
    same_class = target[:, None].eq(target[None, :])
    different_class = ~same_class

    # Same-class non-paired samples are weak positives, not false negatives.
    pair_candidates = identity | different_class
    pair_denominator = torch.logsumexp(
        scores.masked_fill(~pair_candidates, -torch.inf), dim=1
    )
    pair_loss = -(scores.diagonal() - pair_denominator).mean()

    weak_positive = same_class & ~identity
    candidates = ~identity
    class_denominator = torch.logsumexp(
        scores.masked_fill(~candidates, -torch.inf), dim=1
    )
    per_pair = -(scores - class_denominator[:, None])
    positive_count = weak_positive.sum(dim=1)
    valid = positive_count > 0
    if valid.any():
        class_loss = (
            (per_pair * weak_positive).sum(dim=1)[valid] / positive_count[valid]
        ).mean()
    else:
        class_loss = scores.sum() * 0.0
    return pair_loss, class_loss


def class_aware_ot_losses(
    distance: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric strong-pair and weak same-class contrastive losses."""
    if temperature <= 0:
        raise ValueError("OT contrast temperature must be positive.")
    eeg_pair, eeg_class = _directional_class_aware_contrast(distance, target, temperature)
    prompt_pair, prompt_class = _directional_class_aware_contrast(
        distance.transpose(0, 1), target, temperature
    )
    return 0.5 * (eeg_pair + prompt_pair), 0.5 * (eeg_class + prompt_class)
