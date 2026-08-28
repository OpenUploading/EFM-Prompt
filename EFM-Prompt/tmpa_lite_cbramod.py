"""TMPA-lite adapter for CBraMod and paired SHIN EEG-fNIRS trials.

This is the first stage of the TMPA-inspired experiment: one transport-aware
fNIRS residual is aligned to CBraMod's spatial and temporal EEG tokens. It does
not implement multi-mode prompt banks or hierarchical prompt-level OT yet.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def sinkhorn_plan(
    source: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 0.1,
    iterations: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an entropy-regularized OT plan and its transport cost."""
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("source and target must be [B,N,D] and [B,M,D]")
    if source.shape[0] != target.shape[0] or source.shape[2] != target.shape[2]:
        raise ValueError(f"Incompatible token shapes: {source.shape}, {target.shape}")
    if epsilon <= 0 or iterations < 1:
        raise ValueError("epsilon must be positive and iterations must be >= 1")

    source = F.normalize(source, dim=-1)
    target = F.normalize(target, dim=-1)
    cost = (1.0 - torch.bmm(source, target.transpose(1, 2))).clamp_min(0.0)
    kernel = torch.exp((-cost / epsilon).clamp_min(-50.0))
    rows = torch.full(
        (source.shape[0], source.shape[1]),
        1.0 / source.shape[1],
        dtype=source.dtype,
        device=source.device,
    )
    cols = torch.full(
        (target.shape[0], target.shape[1]),
        1.0 / target.shape[1],
        dtype=target.dtype,
        device=target.device,
    )
    u = torch.ones_like(rows)
    v = torch.ones_like(cols)
    for _ in range(iterations):
        u = rows / torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
        v = cols / torch.bmm(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
    plan = u.unsqueeze(-1) * kernel * v.unsqueeze(1)
    distance = (plan * cost).sum(dim=(1, 2))
    return plan, distance


def transport_features(plan: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Transport target-side features into the source token positions."""
    weights = plan / plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.bmm(weights, source)


class FnirsTokenEncoder(nn.Module):
    """Encode normalized SHIN HbO/HbR graph trials as [B,36,10,D] tokens."""

    def __init__(self, output_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, output_dim, kernel_size=5, padding=2),
            nn.GroupNorm(8, output_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(10),
        )

    def forward(self, fnirs: torch.Tensor) -> torch.Tensor:
        if fnirs.ndim != 4 or tuple(fnirs.shape[1:3]) != (36, 2):
            raise ValueError(f"Expected fNIRS [B,36,2,T], got {tuple(fnirs.shape)}")
        batch, nodes, chromophores, samples = fnirs.shape
        tokens = self.net(fnirs.reshape(batch * nodes, chromophores, samples))
        return tokens.transpose(1, 2).reshape(batch, nodes, 10, -1)


class TMPALiteAdapter(nn.Module):
    """Build a single OT-aligned fNIRS residual for CBraMod patch tokens."""

    def __init__(
        self,
        d_model: int = 200,
        alignment_dim: int = 128,
        prompt_scale: float = 0.05,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iterations: int = 20,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.alignment_dim = alignment_dim
        self.prompt_scale = float(prompt_scale)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.eeg_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, alignment_dim),
        )
        self.fnirs_encoder = FnirsTokenEncoder(alignment_dim, dropout)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(alignment_dim),
            nn.Linear(alignment_dim, d_model),
        )
        self.gate = nn.Parameter(torch.tensor(-4.0))

    def forward(
        self,
        eeg_patch_tokens: torch.Tensor,
        fnirs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if eeg_patch_tokens.ndim != 4 or tuple(eeg_patch_tokens.shape[1:3]) != (30, 10):
            raise ValueError(
                f"Expected CBraMod patch tokens [B,30,10,D], got {tuple(eeg_patch_tokens.shape)}"
            )
        eeg = self.eeg_projection(eeg_patch_tokens)
        fnirs_tokens = self.fnirs_encoder(fnirs)

        eeg_spatial = eeg.mean(dim=2)
        fnirs_spatial = fnirs_tokens.mean(dim=2)
        spatial_plan, spatial_distance = sinkhorn_plan(
            eeg_spatial,
            fnirs_spatial,
            self.sinkhorn_epsilon,
            self.sinkhorn_iterations,
        )
        spatial_aligned = transport_features(spatial_plan, fnirs_spatial)

        eeg_temporal = eeg.mean(dim=1)
        fnirs_temporal = fnirs_tokens.mean(dim=1)
        temporal_plan, temporal_distance = sinkhorn_plan(
            eeg_temporal,
            fnirs_temporal,
            self.sinkhorn_epsilon,
            self.sinkhorn_iterations,
        )
        temporal_aligned = transport_features(temporal_plan, fnirs_temporal)

        aligned = spatial_aligned.unsqueeze(2) + temporal_aligned.unsqueeze(1)
        aligned = self.output_projection(aligned)
        scale = self.prompt_scale * torch.sigmoid(self.gate)
        enhanced = eeg_patch_tokens + scale * aligned
        losses = {
            "spatial_ot": spatial_distance.mean(),
            "temporal_ot": temporal_distance.mean(),
            "gate": torch.sigmoid(self.gate).detach(),
            "spatial_plan": spatial_plan,
            "temporal_plan": temporal_plan,
        }
        return enhanced, losses
