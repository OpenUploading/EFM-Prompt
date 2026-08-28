"""Model-agnostic final TMPA adapter for EEG foundation models.

The adapter deliberately treats both modalities as unordered token sets. It
uses multi-mode prompts and hierarchical optimal transport, but never assumes
that an EFM token axis represents space or time.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def sinkhorn_plan(
    source_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    epsilon: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute batched entropic OT between two token sets."""
    source_tokens = F.normalize(source_tokens, dim=-1)
    target_tokens = F.normalize(target_tokens, dim=-1)
    cost = (1.0 - torch.bmm(source_tokens, target_tokens.transpose(1, 2))).clamp_min(0.0)
    return sinkhorn_from_cost(cost, epsilon, iterations)


def sinkhorn_from_cost(
    cost: torch.Tensor,
    epsilon: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute batched entropic OT from a precomputed cost matrix."""
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


class FnirsTokenEncoder(nn.Module):
    """Encode [node,HbO/HbR,time] data into an unordered token set.

    The node count is intentionally variable: SHIN has 36 measurement
    channels while FineMI has 24.  No synthetic/padded fNIRS nodes are used.
    """

    def __init__(self, output_dim: int, temporal_tokens: int = 10, dropout: float = 0.1) -> None:
        super().__init__()
        if output_dim % 8:
            raise ValueError("alignment dimension must be divisible by 8")
        self.temporal_tokens = int(temporal_tokens)
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, output_dim, kernel_size=5, padding=2),
            nn.GroupNorm(8, output_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(self.temporal_tokens),
        )

    def forward(self, fnirs: torch.Tensor) -> torch.Tensor:
        if fnirs.ndim != 4 or fnirs.shape[1] < 1 or fnirs.shape[2] != 2:
            raise ValueError(f"Expected fNIRS [B,nodes,2,T], got {tuple(fnirs.shape)}")
        batch, nodes, chromophores, samples = fnirs.shape
        encoded = self.encoder(fnirs.reshape(batch * nodes, chromophores, samples))
        return encoded.transpose(1, 2).reshape(batch, nodes * self.temporal_tokens, -1)


class ModePromptPool(nn.Module):
    """Use K learnable prompt groups to discover K sample-conditioned modes."""

    def __init__(
        self,
        dim: int,
        mode_count: int,
        tokens_per_mode: int,
        attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.mode_count = int(mode_count)
        self.tokens_per_mode = int(tokens_per_mode)
        self.prompts = nn.Parameter(torch.empty(mode_count, tokens_per_mode, dim))
        nn.init.normal_(self.prompts, std=0.02)
        self.attention = nn.MultiheadAttention(
            dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        queries = self.prompts.reshape(1, -1, tokens.shape[-1]).expand(batch, -1, -1)
        conditioned, _ = self.attention(queries, tokens, tokens, need_weights=False)
        conditioned = self.norm(queries + conditioned)
        return conditioned.reshape(
            batch, self.mode_count, self.tokens_per_mode, tokens.shape[-1]
        )


class FoundationTMPAFinalAdapter(nn.Module):
    """TMPA-style token-level + prompt-level OT without axis partitioning."""

    def __init__(
        self,
        eeg_dim: int = 200,
        alignment_dim: int = 128,
        fnirs_temporal_tokens: int = 10,
        mode_count: int = 4,
        prompt_tokens_per_mode: int = 8,
        attention_heads: int = 8,
        token_cost_weight: float = 1.0,
        prompt_scale: float = 0.05,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iterations: int = 20,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if sinkhorn_epsilon <= 0 or sinkhorn_iterations < 1:
            raise ValueError("Invalid Sinkhorn settings")
        if mode_count < 2 or prompt_tokens_per_mode < 1:
            raise ValueError("Final TMPA requires at least two prompt modes")
        if alignment_dim % attention_heads:
            raise ValueError("alignment_dim must be divisible by attention_heads")
        self.mode_count = int(mode_count)
        self.tokens_per_mode = int(prompt_tokens_per_mode)
        self.token_cost_weight = float(token_cost_weight)
        self.prompt_scale = float(prompt_scale)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.eeg_projection = nn.Sequential(nn.LayerNorm(eeg_dim), nn.Linear(eeg_dim, alignment_dim))
        self.fnirs_encoder = FnirsTokenEncoder(alignment_dim, fnirs_temporal_tokens, dropout)
        self.eeg_modes = ModePromptPool(
            alignment_dim, mode_count, prompt_tokens_per_mode, attention_heads, dropout
        )
        self.fnirs_modes = ModePromptPool(
            alignment_dim, mode_count, prompt_tokens_per_mode, attention_heads, dropout
        )
        self.token_to_prompt = nn.MultiheadAttention(
            alignment_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(alignment_dim), nn.Linear(alignment_dim, eeg_dim)
        )
        self.gate = nn.Parameter(torch.tensor(-4.0))

    def _all_pair_hierarchical_transport(
        self,
        eeg_modes: torch.Tensor,
        fnirs_modes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute TMPA's inner token OT and outer prompt OT for all BxB pairs."""
        batch, modes, tokens, dim = eeg_modes.shape
        eeg_pairs = eeg_modes[:, None, :, None].expand(
            -1, batch, -1, modes, -1, -1
        )
        fnirs_pairs = fnirs_modes[None, :, None, :].expand(
            batch, -1, modes, -1, -1, -1
        )
        flat_eeg = eeg_pairs.reshape(batch * batch * modes * modes, tokens, dim)
        flat_fnirs = fnirs_pairs.reshape(batch * batch * modes * modes, tokens, dim)
        token_plan, token_distance = sinkhorn_plan(
            flat_eeg, flat_fnirs, self.sinkhorn_epsilon, self.sinkhorn_iterations
        )
        token_distance = token_distance.reshape(batch, batch, modes, modes)

        eeg_global = F.normalize(eeg_modes.mean(dim=2), dim=-1)
        fnirs_global = F.normalize(fnirs_modes.mean(dim=2), dim=-1)
        global_similarity = torch.einsum("bkd,cld->bckl", eeg_global, fnirs_global)
        prompt_cost = (
            1.0 - global_similarity
            + self.token_cost_weight * token_distance
        )
        mode_plan, hierarchical_distance = sinkhorn_from_cost(
            prompt_cost.reshape(batch * batch, modes, modes),
            self.sinkhorn_epsilon,
            self.sinkhorn_iterations,
        )
        mode_plan = mode_plan.reshape(batch, batch, modes, modes)
        hierarchical_distance = hierarchical_distance.reshape(batch, batch)

        # Classification injects only the true EEG_i/fNIRS_i pair. Other BxB
        # combinations provide training-time contrast and never affect inference.
        token_plan = token_plan.reshape(
            batch, batch, modes, modes, tokens, tokens
        )
        diagonal = torch.arange(batch, device=eeg_modes.device)
        paired_token_plan = token_plan[diagonal, diagonal]
        paired_fnirs = fnirs_modes[:, None].expand(-1, modes, -1, -1, -1)
        token_weights = token_plan / token_plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        paired_token_weights = token_weights[diagonal, diagonal]
        pair_transport = torch.einsum(
            "bmnuv,bmnvd->bmnud", paired_token_weights, paired_fnirs
        )
        paired_mode_plan = mode_plan[diagonal, diagonal]
        mode_weights = paired_mode_plan / paired_mode_plan.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        transported = (mode_weights[..., None, None] * pair_transport).sum(dim=2)
        return transported, hierarchical_distance, token_distance, mode_plan

    def _paired_hierarchical_transport(
        self,
        eeg_modes: torch.Tensor,
        fnirs_modes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute hierarchical OT only for the true EEG_i/fNIRS_i pairs."""
        batch, modes, tokens, dim = eeg_modes.shape
        eeg_pairs = eeg_modes[:, :, None].expand(-1, -1, modes, -1, -1)
        fnirs_pairs = fnirs_modes[:, None, :].expand(-1, modes, -1, -1, -1)
        flat_eeg = eeg_pairs.reshape(batch * modes * modes, tokens, dim)
        flat_fnirs = fnirs_pairs.reshape(batch * modes * modes, tokens, dim)
        token_plan, token_distance = sinkhorn_plan(
            flat_eeg, flat_fnirs, self.sinkhorn_epsilon, self.sinkhorn_iterations
        )
        token_distance = token_distance.reshape(batch, modes, modes)

        eeg_global = F.normalize(eeg_modes.mean(dim=2), dim=-1)
        fnirs_global = F.normalize(fnirs_modes.mean(dim=2), dim=-1)
        global_similarity = torch.einsum("bkd,bld->bkl", eeg_global, fnirs_global)
        prompt_cost = 1.0 - global_similarity + self.token_cost_weight * token_distance
        mode_plan, pair_distance = sinkhorn_from_cost(
            prompt_cost, self.sinkhorn_epsilon, self.sinkhorn_iterations
        )
        token_weights = token_plan.reshape(batch, modes, modes, tokens, tokens)
        token_weights = token_weights / token_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        pair_transport = torch.einsum("bmnuv,bmnvd->bmnud", token_weights, fnirs_pairs)
        mode_weights = mode_plan / mode_plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        transported = (mode_weights[..., None, None] * pair_transport).sum(dim=2)
        return transported, pair_distance, token_distance, mode_plan

    def forward(
        self,
        eeg_tokens: torch.Tensor,
        fnirs: torch.Tensor,
        compute_pair_matrix: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if eeg_tokens.ndim < 3:
            raise ValueError(f"Expected EFM tokens [B,...,D], got {tuple(eeg_tokens.shape)}")
        original_shape = eeg_tokens.shape
        eeg_set = eeg_tokens.reshape(eeg_tokens.shape[0], -1, eeg_tokens.shape[-1])
        eeg_set_aligned = self.eeg_projection(eeg_set)
        fnirs_set = self.fnirs_encoder(fnirs)
        eeg_modes = self.eeg_modes(eeg_set_aligned)
        fnirs_modes = self.fnirs_modes(fnirs_set)
        if compute_pair_matrix:
            transported_modes, sample_distance, token_distance, mode_plan = (
                self._all_pair_hierarchical_transport(eeg_modes, fnirs_modes)
            )
            diagonal = torch.arange(eeg_tokens.shape[0], device=eeg_tokens.device)
            paired_prompt_ot = sample_distance[diagonal, diagonal].mean()
            paired_token_ot = token_distance[diagonal, diagonal].mean()
        else:
            transported_modes, pair_distance, token_distance, mode_plan = (
                self._paired_hierarchical_transport(eeg_modes, fnirs_modes)
            )
            # Preserve the result schema while ensuring no cross-trial pair is built.
            sample_distance = torch.diag_embed(pair_distance)
            paired_prompt_ot = pair_distance.mean()
            paired_token_ot = token_distance.mean()
        transported_set = transported_modes.flatten(1, 2)
        residual, _ = self.token_to_prompt(
            eeg_set_aligned, transported_set, transported_set, need_weights=False
        )
        residual = self.output_projection(residual).reshape(original_shape)
        effective_scale = self.prompt_scale * torch.sigmoid(self.gate)
        enhanced = eeg_tokens + effective_scale * residual
        eeg_descriptors = F.normalize(eeg_modes.mean(dim=2), dim=-1)
        fnirs_descriptors = F.normalize(fnirs_modes.mean(dim=2), dim=-1)
        eeg_mode_similarity = torch.matmul(
            eeg_descriptors, eeg_descriptors.transpose(1, 2)
        )
        fnirs_mode_similarity = torch.matmul(
            fnirs_descriptors, fnirs_descriptors.transpose(1, 2)
        )
        identity = torch.eye(self.mode_count, device=eeg_tokens.device, dtype=eeg_tokens.dtype)
        off_diagonal = (1.0 - identity).unsqueeze(0)
        mode_similarity = 0.5 * (
            (eeg_mode_similarity * off_diagonal).sum() / off_diagonal.sum().clamp_min(1.0) / eeg_tokens.shape[0]
            + (fnirs_mode_similarity * off_diagonal).sum() / off_diagonal.sum().clamp_min(1.0) / eeg_tokens.shape[0]
        )
        return enhanced, {
            "sample_distance": sample_distance,
            "paired_prompt_ot": paired_prompt_ot,
            "paired_token_ot": paired_token_ot,
            "mode_similarity": mode_similarity,
            "gate": torch.sigmoid(self.gate).detach(),
        }


# Compatibility alias for older imports; the active implementation is final TMPA.
FoundationTokenOTAdapter = FoundationTMPAFinalAdapter
