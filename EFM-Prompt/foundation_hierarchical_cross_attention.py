"""Hierarchical cross-attention alternative to the final OT adapter.

This module keeps the same four-mode/two-token interface as the current final
TMPA experiment, but contains no Sinkhorn transport. Token-level and
prompt-level alignment are both learned cross-attention operations.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from foundation_tmpa_token_alignment import FnirsTokenEncoder, ModePromptPool


def _normalized_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Return attention entropy normalized to [0, 1]."""
    probabilities = weights.clamp_min(1e-8)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    denominator = torch.log(
        torch.tensor(weights.shape[-1], dtype=weights.dtype, device=weights.device)
    ).clamp_min(1.0)
    return (entropy / denominator).mean()


class FoundationHierarchicalCrossAttentionAdapter(nn.Module):
    """Two-level cross-attention prompt adapter without optimal transport."""

    method_name = "foundation_hierarchical_cross_attention_contrastive"
    alignment_description = (
        "multi-mode prompts + token-level cross-attention + prompt-level "
        "cross-attention; no Sinkhorn/OT"
    )
    metric_note = (
        "Compatibility fields paired_prompt_ot/paired_token_ot contain "
        "normalized prompt/token attention entropy, not OT distance."
    )

    def __init__(
        self,
        eeg_dim: int = 200,
        alignment_dim: int = 128,
        fnirs_temporal_tokens: int = 10,
        mode_count: int = 4,
        prompt_tokens_per_mode: int = 2,
        attention_heads: int = 8,
        token_cost_weight: float = 1.0,
        prompt_scale: float = 0.05,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iterations: int = 100,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        del token_cost_weight, sinkhorn_epsilon, sinkhorn_iterations
        if mode_count < 2 or prompt_tokens_per_mode < 1:
            raise ValueError("Hierarchical attention requires at least two prompt modes")
        if alignment_dim % attention_heads:
            raise ValueError("alignment_dim must be divisible by attention_heads")
        self.mode_count = int(mode_count)
        self.tokens_per_mode = int(prompt_tokens_per_mode)
        self.prompt_scale = float(prompt_scale)
        self.eeg_projection = nn.Sequential(
            nn.LayerNorm(eeg_dim), nn.Linear(eeg_dim, alignment_dim)
        )
        self.fnirs_encoder = FnirsTokenEncoder(
            alignment_dim, fnirs_temporal_tokens, dropout
        )
        self.eeg_modes = ModePromptPool(
            alignment_dim, mode_count, prompt_tokens_per_mode, attention_heads, dropout
        )
        self.fnirs_modes = ModePromptPool(
            alignment_dim, mode_count, prompt_tokens_per_mode, attention_heads, dropout
        )
        self.token_cross_attention = nn.MultiheadAttention(
            alignment_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.prompt_cross_attention = nn.MultiheadAttention(
            alignment_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.token_norm = nn.LayerNorm(alignment_dim)
        self.prompt_norm = nn.LayerNorm(alignment_dim)
        self.token_to_prompt = nn.MultiheadAttention(
            alignment_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(alignment_dim), nn.Linear(alignment_dim, eeg_dim)
        )
        self.gate = nn.Parameter(torch.tensor(-4.0))

    def _paired_hierarchical_attention(
        self, eeg_modes: torch.Tensor, fnirs_modes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse paired trials at token level, then at prompt-mode level."""
        batch, modes, tokens, dim = eeg_modes.shape
        eeg_pairs = eeg_modes[:, :, None].expand(-1, -1, modes, -1, -1)
        fnirs_pairs = fnirs_modes[:, None, :].expand(-1, modes, -1, -1, -1)
        token_query = eeg_pairs.reshape(batch * modes * modes, tokens, dim)
        token_context = fnirs_pairs.reshape(batch * modes * modes, tokens, dim)
        token_fused, token_weights = self.token_cross_attention(
            token_query,
            token_context,
            token_context,
            need_weights=True,
            average_attn_weights=False,
        )
        token_fused = self.token_norm(token_query + token_fused).reshape(
            batch, modes, modes, tokens, dim
        )

        # Each EEG mode receives one summary from every fNIRS mode. The second
        # attention level chooses among those mode-level summaries.
        mode_context = token_fused.mean(dim=3).reshape(batch * modes, modes, dim)
        mode_query = eeg_modes.mean(dim=2).reshape(batch * modes, 1, dim)
        prompt_fused, prompt_weights = self.prompt_cross_attention(
            mode_query,
            mode_context,
            mode_context,
            need_weights=True,
            average_attn_weights=False,
        )
        prompt_fused = self.prompt_norm(mode_query + prompt_fused).reshape(
            batch, modes, 1, dim
        )
        transported = prompt_fused.expand(-1, -1, tokens, -1)
        return (
            transported,
            _normalized_entropy(token_weights),
            _normalized_entropy(prompt_weights),
        )

    @staticmethod
    def _sample_distance(
        eeg_modes: torch.Tensor, fnirs_modes: torch.Tensor
    ) -> torch.Tensor:
        """Cosine distance matrix used by the unchanged class-aware contrast."""
        eeg_descriptor = F.normalize(eeg_modes.mean(dim=(1, 2)), dim=-1)
        fnirs_descriptor = F.normalize(fnirs_modes.mean(dim=(1, 2)), dim=-1)
        return (1.0 - eeg_descriptor @ fnirs_descriptor.transpose(0, 1)).clamp_min(0.0)

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
        eeg_aligned = self.eeg_projection(eeg_set)
        fnirs_set = self.fnirs_encoder(fnirs)
        eeg_modes = self.eeg_modes(eeg_aligned)
        fnirs_modes = self.fnirs_modes(fnirs_set)
        transported, token_entropy, prompt_entropy = self._paired_hierarchical_attention(
            eeg_modes, fnirs_modes
        )

        residual_context = transported.flatten(1, 2)
        residual, _ = self.token_to_prompt(
            eeg_aligned, residual_context, residual_context, need_weights=False
        )
        residual = self.output_projection(residual).reshape(original_shape)
        enhanced = eeg_tokens + self.prompt_scale * torch.sigmoid(self.gate) * residual

        sample_distance = self._sample_distance(eeg_modes, fnirs_modes)
        if not compute_pair_matrix:
            sample_distance = torch.diag_embed(sample_distance.diagonal())
        eeg_descriptors = F.normalize(eeg_modes.mean(dim=2), dim=-1)
        fnirs_descriptors = F.normalize(fnirs_modes.mean(dim=2), dim=-1)
        identity = torch.eye(
            self.mode_count, device=eeg_tokens.device, dtype=eeg_tokens.dtype
        )
        off_diagonal = (1.0 - identity).unsqueeze(0)
        mode_similarity = 0.5 * (
            (torch.matmul(eeg_descriptors, eeg_descriptors.transpose(1, 2)) * off_diagonal).sum()
            + (torch.matmul(fnirs_descriptors, fnirs_descriptors.transpose(1, 2)) * off_diagonal).sum()
        ) / (off_diagonal.sum().clamp_min(1.0) * eeg_tokens.shape[0] * 2.0)
        return enhanced, {
            "sample_distance": sample_distance,
            # Compatibility aliases consumed by the shared runner.
            "paired_prompt_ot": prompt_entropy,
            "paired_token_ot": token_entropy,
            "prompt_attention_entropy": prompt_entropy,
            "token_attention_entropy": token_entropy,
            "mode_similarity": mode_similarity,
            "gate": torch.sigmoid(self.gate).detach(),
        }

