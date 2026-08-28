"""MoPE conditional prompts adapted to CBraMod's fixed spatiotemporal grid."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sgformer_mapped_prompt import (
    DenseGraphConv,
    SGFormerMappedReadout,
    normalized_adjacency,
)


TAP_ATTRIBUTE_NAMES = ("hbo", "hbr", "spatial", "temporal")


class TAPFourAttributeFnirsEncoder(nn.Module):
    """Encode four hard-separated physiological views from HbO/HbR graph trials."""

    def __init__(
        self,
        positions_3d: torch.Tensor,
        edge_index: torch.Tensor,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if positions_3d.shape != (36, 3):
            raise ValueError(f"Expected 36 fNIRS positions, got {tuple(positions_3d.shape)}")
        hidden_dim = min(128, max(64, output_dim // 2))
        geometry = positions_3d.float()
        geometry = (geometry - geometry.mean(dim=0, keepdim=True)) / (
            geometry.std(dim=0, keepdim=True).clamp_min(1e-6)
        )
        self.register_buffer("positions_3d", geometry)
        self.register_buffer("adjacency", normalized_adjacency(36, edge_index.long()))

        def chromophore_path() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(36, hidden_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, output_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.LayerNorm(output_dim),
            )

        self.hbo_path = chromophore_path()
        self.hbr_path = chromophore_path()
        self.spatial_node_projection = nn.Linear(6, output_dim)
        self.spatial_geometry_projection = nn.Linear(3, output_dim)
        self.spatial_node_embedding = nn.Embedding(36, output_dim)
        self.spatial_graph = DenseGraphConv(output_dim, dropout)
        self.spatial_norm = nn.LayerNorm(output_dim)
        self.temporal_path = nn.Sequential(
            nn.Conv1d(4, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, fnirs: torch.Tensor) -> torch.Tensor:
        if fnirs.ndim != 4 or tuple(fnirs.shape[1:3]) != (36, 2):
            raise ValueError(f"Expected fNIRS [B,36,2,T], got {tuple(fnirs.shape)}")
        hbo = self.hbo_path(fnirs[:, :, 0, :])
        hbr = self.hbr_path(fnirs[:, :, 1, :])

        spatial_statistics = torch.cat(
            (
                fnirs.mean(dim=-1),
                fnirs.std(dim=-1, unbiased=False),
                fnirs[..., -1] - fnirs[..., 0],
            ),
            dim=-1,
        )
        node_ids = torch.arange(36, device=fnirs.device)
        spatial_nodes = (
            self.spatial_node_projection(spatial_statistics)
            + self.spatial_geometry_projection(self.positions_3d).unsqueeze(0)
            + self.spatial_node_embedding(node_ids).unsqueeze(0)
        )
        spatial = self.spatial_norm(
            self.spatial_graph(spatial_nodes, self.adjacency).mean(dim=1)
        )

        temporal_mean = fnirs.mean(dim=1)
        temporal_std = fnirs.std(dim=1, unbiased=False)
        temporal = self.temporal_path(torch.cat((temporal_mean, temporal_std), dim=1))
        return torch.stack((hbo, hbr, spatial, temporal), dim=1)


class MoPEBoundaryPrompt(nn.Module):
    """Generate static, expert-routed dynamic, and mapped prompts.

    The paper concatenates these prompts as tokens at each Transformer layer.
    CBraMod's criss-cross attention requires a fixed channel-by-patch grid, so
    this adapter projects the concatenated prompts to an additive grid residual.
    """

    def __init__(
        self,
        condition_dim: int,
        d_model: int,
        prompt_count: int,
        rank: int,
        token_count: int,
        hidden_dim: int,
        dropout: float,
        expert_count: int = 16,
        temperature: float = 0.1,
        router_noise_std: float = 0.00390625,
        importance_threshold: float = 0.05,
        drop_component: str = "none",
        mapped_mode: str = "mlp",
        dynamic_expert_mode: str = "flat",
        class_count: int = 2,
    ) -> None:
        super().__init__()
        if prompt_count < 1 or expert_count < 2 or rank < 1:
            raise ValueError("MoPE requires prompt_count >= 1, expert_count >= 2, and rank >= 1.")
        if temperature <= 0 or router_noise_std < 0 or importance_threshold < 0:
            raise ValueError("Invalid MoPE temperature, noise, or importance threshold.")
        if drop_component not in {"none", "static", "dynamic", "mapped"}:
            raise ValueError(f"Invalid MoPE component ablation: {drop_component}")
        if mapped_mode not in {"mlp", "sgformer"}:
            raise ValueError(f"Invalid mapped prompt mode: {mapped_mode}")
        if dynamic_expert_mode not in {"flat", "tap4x4"}:
            raise ValueError(f"Invalid dynamic expert mode: {dynamic_expert_mode}")
        if dynamic_expert_mode == "tap4x4" and expert_count != 16:
            raise ValueError("TAP 4x4 dynamic prompts require exactly 16 experts")

        self.prompt_count = prompt_count
        self.expert_count = expert_count
        self.d_model = d_model
        self.temperature = temperature
        self.router_noise_std = router_noise_std
        self.importance_threshold = importance_threshold
        self.drop_component = drop_component
        self.mapped_mode = mapped_mode
        self.dynamic_expert_mode = dynamic_expert_mode
        self.attribute_count = len(TAP_ATTRIBUTE_NAMES)
        self.experts_per_attribute = expert_count // self.attribute_count

        self.static_prompt = None if drop_component == "static" else nn.Parameter(torch.empty(prompt_count, d_model))
        if drop_component == "dynamic":
            self.prompt_experts = None
            self.router = None
            self.attribute_heads = None
        elif dynamic_expert_mode == "tap4x4":
            self.prompt_experts = nn.Parameter(torch.empty(
                self.attribute_count, self.experts_per_attribute, prompt_count, d_model
            ))
            self.router = nn.ModuleList([
                nn.Linear(condition_dim, self.experts_per_attribute)
                for _ in range(self.attribute_count)
            ])
            self.attribute_heads = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, class_count))
                for _ in range(self.attribute_count)
            ])
        else:
            self.prompt_experts = nn.Parameter(torch.empty(expert_count, prompt_count, d_model))
            self.router = nn.Linear(condition_dim, expert_count)
            self.attribute_heads = None
        if drop_component == "mapped":
            self.mapper = None
        elif mapped_mode == "mlp":
            self.mapper = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
            )
        else:
            self.mapper = SGFormerMappedReadout(
                d_model=d_model,
                heads=8,
                dropout=dropout,
            )

        # [P_s, P_d, P_m] contains 2*l+1 prompt vectors. The projection below
        # preserves CBraMod's [channel, patch] topology instead of appending tokens.
        combined_tokens = (
            (0 if drop_component == "static" else prompt_count)
            + (0 if drop_component == "dynamic" else prompt_count)
            + (0 if drop_component == "mapped" else 1)
        )
        combined_dim = combined_tokens * d_model
        self.to_rank = nn.Linear(combined_dim, rank)
        self.token_basis = nn.Parameter(torch.empty(rank, token_count, d_model))
        self.alpha = nn.Parameter(torch.zeros(()))
        self._routing_scores: torch.Tensor | None = None
        self._attribute_logits: list[torch.Tensor] | None = None

        if self.static_prompt is not None:
            nn.init.normal_(self.static_prompt, std=0.02)
        if self.prompt_experts is not None:
            nn.init.normal_(self.prompt_experts, std=0.02)
            routers = self.router if isinstance(self.router, nn.ModuleList) else [self.router]
            for router in routers:
                nn.init.normal_(router.weight, std=1e-3)
                nn.init.zeros_(router.bias)
        if self.mapper is not None and self.mapped_mode == "mlp":
            for module in self.mapper:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.to_rank.weight, std=1e-3)
        nn.init.zeros_(self.to_rank.bias)
        nn.init.normal_(self.token_basis, std=0.02)

    def _route_logits(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clean_logits = logits / self.temperature
        clean_scores = torch.softmax(clean_logits, dim=-1)
        routed_logits = clean_logits
        if self.training and self.router_noise_std > 0:
            routed_logits = routed_logits + torch.randn_like(routed_logits) * self.router_noise_std
        return torch.softmax(routed_logits, dim=-1), clean_scores

    def route(
        self,
        condition: torch.Tensor,
        attribute_conditions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dynamic_expert_mode == "flat":
            return self._route_logits(self.router(condition))
        if attribute_conditions is None:
            raise ValueError("TAP 4x4 dynamic prompts require four fNIRS attribute conditions")
        expected = (condition.shape[0], self.attribute_count, condition.shape[1])
        if tuple(attribute_conditions.shape) != expected:
            raise ValueError(
                f"Expected attribute conditions {expected}, got {tuple(attribute_conditions.shape)}"
            )
        logits = torch.stack(
            [router(attribute_conditions[:, index]) for index, router in enumerate(self.router)],
            dim=1,
        )
        return self._route_logits(logits)

    def forward(
        self,
        condition: torch.Tensor,
        mapped_nodes: torch.Tensor | None = None,
        attribute_conditions: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prompts = []
        dynamic_tokens = None
        mapped_tokens = None
        if self.static_prompt is not None:
            prompts.append(self.static_prompt.unsqueeze(0).expand(condition.shape[0], -1, -1))
        if self.prompt_experts is not None:
            routing_scores, clean_scores = self.route(condition, attribute_conditions)
            self._routing_scores = clean_scores
            if self.dynamic_expert_mode == "tap4x4":
                attribute_prompts = torch.einsum(
                    "bam,amld->bald", routing_scores, self.prompt_experts
                )
                self._attribute_logits = [
                    head(attribute_prompts[:, index].mean(dim=1))
                    for index, head in enumerate(self.attribute_heads)
                ]
                dynamic_tokens = attribute_prompts.mean(dim=1)
                prompts.append(dynamic_tokens)
            else:
                self._attribute_logits = None
                dynamic_tokens = torch.einsum("bk,kld->bld", routing_scores, self.prompt_experts)
                prompts.append(dynamic_tokens)
        if self.mapper is not None:
            if self.mapped_mode == "sgformer":
                if mapped_nodes is None:
                    raise ValueError("SGFormer mapped mode requires 36 mapped node tokens")
                mapped_tokens = self.mapper(mapped_nodes)
            else:
                mapped_tokens = self.mapper(condition).unsqueeze(1)
            prompts.append(mapped_tokens)
        prompts = torch.cat(prompts, dim=1)
        coefficients = self.to_rank(prompts.flatten(1))
        residual = self.alpha * torch.einsum("br,rtd->btd", coefficients, self.token_basis)
        if not return_aux:
            return residual
        if dynamic_tokens is None or mapped_tokens is None:
            raise ValueError(
                "dynamic_mapped_class_ot requires both dynamic and mapped MoPE prompt components."
            )
        return residual, {
            "dynamic_tokens": dynamic_tokens,
            "mapped_tokens": mapped_tokens,
            "contrast_tokens": torch.cat((dynamic_tokens, mapped_tokens), dim=1),
        }

    def attribute_loss(self, target: torch.Tensor) -> torch.Tensor:
        if not self._attribute_logits:
            return self.token_basis.new_zeros(())
        return torch.stack([
            F.cross_entropy(logits, target) for logits in self._attribute_logits
        ]).mean()

    def importance_loss(self) -> torch.Tensor:
        """Thresholded squared coefficient of variation from Eq. (5)-(6)."""
        if self._routing_scores is None:
            return self.token_basis.new_zeros(())
        importance = self._routing_scores.sum(dim=0)
        if self.dynamic_expert_mode == "tap4x4":
            coefficient_of_variation = importance.std(dim=-1, unbiased=False) / (
                importance.mean(dim=-1).clamp_min(1e-8)
            )
        else:
            coefficient_of_variation = importance.std(unbiased=False) / importance.mean().clamp_min(1e-8)
        active = (coefficient_of_variation.detach() >= self.importance_threshold).to(importance.dtype)
        return (coefficient_of_variation.square() * active).mean()

    @torch.no_grad()
    def routing_statistics(self) -> dict[str, float]:
        result = {}
        if self._routing_scores is not None:
            scores = self._routing_scores.detach()
            mean_usage = scores.mean(dim=0)
            entropy = -(scores.clamp_min(1e-8) * scores.clamp_min(1e-8).log()).sum(dim=-1).mean()
            normalizer = self.experts_per_attribute if self.dynamic_expert_mode == "tap4x4" else self.expert_count
            normalized_entropy = entropy / math.log(normalizer)
            if self.dynamic_expert_mode == "tap4x4":
                coefficient_of_variation = (
                    mean_usage.std(dim=-1, unbiased=False)
                    / mean_usage.mean(dim=-1).clamp_min(1e-8)
                ).mean()
            else:
                coefficient_of_variation = mean_usage.std(unbiased=False) / mean_usage.mean().clamp_min(1e-8)
            result.update({
                "normalized_entropy": float(normalized_entropy.item()),
                "importance_cv": float(coefficient_of_variation.item()),
                "maximum_mean_usage": float(mean_usage.max().item()),
                "minimum_mean_usage": float(mean_usage.min().item()),
            })
            if self.dynamic_expert_mode == "tap4x4":
                for index, name in enumerate(TAP_ATTRIBUTE_NAMES):
                    group = mean_usage[index]
                    result[f"{name}_maximum_mean_usage"] = float(group.max().item())
                    result[f"{name}_minimum_mean_usage"] = float(group.min().item())
        if self.mapper is not None and hasattr(self.mapper, "attention_statistics"):
            result.update(self.mapper.attention_statistics())
        return result
