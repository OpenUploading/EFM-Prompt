"""Shared fNIRS-conditioned deep residual prompt module.

The module keeps an EFM token grid unchanged.  It is therefore usable for
Transformer backbones and CodeBrain's SSSM blocks without appending tokens.
"""

from __future__ import annotations

import torch
from torch import nn

from foundation_tmpa_token_alignment import FnirsTokenEncoder


class DeepConditionalPrompt(nn.Module):
    """One shared fNIRS encoder and cross-attention prompt for several depths."""

    method_name = "unified_deep_conditional_residual_prompt"

    def __init__(
        self,
        eeg_dim: int = 200,
        prompt_dim: int = 128,
        prompt_tokens: int = 4,
        stages: int = 3,
        fnirs_temporal_tokens: int = 10,
        attention_heads: int = 8,
        prompt_scale: float = 0.05,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if prompt_dim % attention_heads:
            raise ValueError("prompt_dim must be divisible by attention_heads")
        self.prompt_scale = float(prompt_scale)
        self.stage_count = int(stages)
        self.fnirs_encoder = FnirsTokenEncoder(prompt_dim, fnirs_temporal_tokens, dropout)
        # Stage-specific static queries; all fNIRS encoding and attention maps are shared.
        self.static_prompts = nn.Parameter(torch.empty(stages, prompt_tokens, prompt_dim))
        nn.init.normal_(self.static_prompts, std=0.02)
        self.prompt_from_fnirs = nn.MultiheadAttention(
            prompt_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.tokens_from_prompt = nn.MultiheadAttention(
            prompt_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.eeg_in = nn.Sequential(nn.LayerNorm(eeg_dim), nn.Linear(eeg_dim, prompt_dim))
        self.eeg_out = nn.Sequential(nn.LayerNorm(prompt_dim), nn.Linear(prompt_dim, eeg_dim))
        self.prompt_norm = nn.LayerNorm(prompt_dim)
        self.token_norm = nn.LayerNorm(prompt_dim)
        # Zero gate preserves the exact frozen EEG-only function at initialization.
        self.gates = nn.Parameter(torch.zeros(stages))

    def forward(self, eeg_tokens: torch.Tensor, fnirs: torch.Tensor, stage: int) -> torch.Tensor:
        if not 0 <= stage < self.stage_count:
            raise ValueError(f"Invalid deep-prompt stage {stage}")
        original_shape = eeg_tokens.shape
        flat_eeg = eeg_tokens.reshape(eeg_tokens.shape[0], -1, eeg_tokens.shape[-1])
        eeg_query = self.eeg_in(flat_eeg)
        fnirs_tokens = self.fnirs_encoder(fnirs)
        base = self.static_prompts[stage].unsqueeze(0).expand(eeg_tokens.shape[0], -1, -1)
        conditioned, _ = self.prompt_from_fnirs(base, fnirs_tokens, fnirs_tokens, need_weights=False)
        prompt = self.prompt_norm(base + conditioned)
        residual, _ = self.tokens_from_prompt(eeg_query, prompt, prompt, need_weights=False)
        residual = self.eeg_out(self.token_norm(eeg_query + residual)).reshape(original_shape)
        return eeg_tokens + self.prompt_scale * torch.tanh(self.gates[stage]) * residual


class SharedDeepThreeComponentPrompt(nn.Module):
    """Shared three-component fNIRS prompt with light layer-specific adapters."""

    method_name = "deep_three_component_shared_low_rank"

    def __init__(
        self,
        eeg_dim: int = 200,
        prompt_dim: int = 128,
        prompt_tokens: int = 6,
        stages: int = 4,
        fnirs_temporal_tokens: int = 10,
        attention_heads: int = 8,
        prompt_scale: float = 0.05,
        dropout: float = 0.1,
        expert_count: int = 16,
        router_temperature: float = 0.1,
        router_noise_std: float = 0.00390625,
        importance_threshold: float = 0.05,
        prompt_rank: int = 8,
        prompt_hidden: int = 256,
    ) -> None:
        super().__init__()
        if prompt_dim % attention_heads:
            raise ValueError("prompt_dim must be divisible by attention_heads")
        if expert_count < 2 or prompt_tokens < 1 or stages < 1 or prompt_rank < 1:
            raise ValueError("Invalid shared deep three-component prompt dimensions")
        if router_temperature <= 0 or router_noise_std < 0 or importance_threshold < 0:
            raise ValueError("Invalid router settings")

        self.prompt_scale = float(prompt_scale)
        self.stage_count = int(stages)
        self.prompt_tokens = int(prompt_tokens)
        self.expert_count = int(expert_count)
        self.router_temperature = float(router_temperature)
        self.router_noise_std = float(router_noise_std)
        self.importance_threshold = float(importance_threshold)

        # These modules are deliberately shared by all injection depths.
        self.fnirs_encoder = FnirsTokenEncoder(prompt_dim, fnirs_temporal_tokens, dropout)
        self.prompt_experts = nn.Parameter(torch.empty(expert_count, prompt_tokens, prompt_dim))
        self.router = nn.Linear(prompt_dim, expert_count)
        self.mapper = nn.Sequential(
            nn.Linear(prompt_dim, prompt_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prompt_hidden, prompt_dim),
        )
        self.eeg_in = nn.Sequential(nn.LayerNorm(eeg_dim), nn.Linear(eeg_dim, prompt_dim))
        self.tokens_from_prompt = nn.MultiheadAttention(
            prompt_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.token_norm = nn.LayerNorm(prompt_dim)

        # Only static prompts, low-rank projections and gates are stage-specific.
        self.static_prompts = nn.Parameter(torch.empty(stages, prompt_tokens, prompt_dim))
        self.stage_down = nn.ModuleList([
            nn.Linear(prompt_dim, prompt_rank, bias=False) for _ in range(stages)
        ])
        self.stage_up = nn.ModuleList([
            nn.Linear(prompt_rank, eeg_dim, bias=False) for _ in range(stages)
        ])
        self.gates = nn.Parameter(torch.zeros(stages))
        self._routing_scores: torch.Tensor | None = None

        nn.init.normal_(self.static_prompts, std=0.02)
        nn.init.normal_(self.prompt_experts, std=0.02)
        nn.init.normal_(self.router.weight, std=1e-3)
        nn.init.zeros_(self.router.bias)
        for module in self.mapper:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        for down, up in zip(self.stage_down, self.stage_up):
            nn.init.normal_(down.weight, std=1e-3)
            nn.init.normal_(up.weight, std=1e-3)

    def encode_fnirs(self, fnirs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Create sample-conditioned components once for all injection stages."""
        fnirs_tokens = self.fnirs_encoder(fnirs)
        condition = fnirs_tokens.mean(dim=1)
        clean_logits = self.router(condition) / self.router_temperature
        clean_scores = torch.softmax(clean_logits, dim=-1)
        routed_logits = clean_logits
        if self.training and self.router_noise_std > 0:
            routed_logits = routed_logits + torch.randn_like(routed_logits) * self.router_noise_std
        routing_scores = torch.softmax(routed_logits, dim=-1)
        dynamic_tokens = torch.einsum("bk,kld->bld", routing_scores, self.prompt_experts)
        mapped_tokens = self.mapper(condition).unsqueeze(1)
        self._routing_scores = clean_scores
        return {
            "dynamic_tokens": dynamic_tokens,
            "mapped_tokens": mapped_tokens,
            "routing_scores": clean_scores,
        }

    def inject(
        self,
        eeg_tokens: torch.Tensor,
        context: dict[str, torch.Tensor],
        stage: int,
    ) -> torch.Tensor:
        if not 0 <= stage < self.stage_count:
            raise ValueError(f"Invalid deep three-component prompt stage {stage}")
        original_shape = eeg_tokens.shape
        flat_eeg = eeg_tokens.reshape(eeg_tokens.shape[0], -1, eeg_tokens.shape[-1])
        eeg_query = self.eeg_in(flat_eeg)
        static_tokens = self.static_prompts[stage].unsqueeze(0).expand(eeg_tokens.shape[0], -1, -1)
        prompt_tokens = torch.cat(
            (static_tokens, context["dynamic_tokens"], context["mapped_tokens"]), dim=1
        )
        attended, _ = self.tokens_from_prompt(eeg_query, prompt_tokens, prompt_tokens, need_weights=False)
        residual = self.token_norm(eeg_query + attended)
        residual = self.stage_up[stage](self.stage_down[stage](residual)).reshape(original_shape)
        return eeg_tokens + self.prompt_scale * torch.tanh(self.gates[stage]) * residual

    def importance_loss(self) -> torch.Tensor:
        if self._routing_scores is None:
            return self.prompt_experts.new_zeros(())
        importance = self._routing_scores.sum(dim=0)
        coefficient_of_variation = importance.std(unbiased=False) / importance.mean().clamp_min(1e-8)
        active = (coefficient_of_variation.detach() >= self.importance_threshold).to(importance.dtype)
        return coefficient_of_variation.square() * active

    @torch.no_grad()
    def routing_statistics(self) -> dict[str, float]:
        if self._routing_scores is None:
            return {}
        scores = self._routing_scores.detach()
        usage = scores.mean(dim=0)
        entropy = -(scores.clamp_min(1e-8) * scores.clamp_min(1e-8).log()).sum(dim=-1).mean()
        coefficient_of_variation = usage.std(unbiased=False) / usage.mean().clamp_min(1e-8)
        return {
            "routing_entropy": float(entropy.item()),
            "routing_normalized_entropy": float((entropy / torch.log(torch.tensor(float(self.expert_count), device=entropy.device))).item()),
            "routing_cv": float(coefficient_of_variation.item()),
            "active_experts": float((usage >= usage.mean() * 0.5).sum().item()),
        }
