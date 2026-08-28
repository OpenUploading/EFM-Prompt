"""Official CodeBrain downstream head adapted to SHIN feature dimensions."""

import torch
from torch import nn


class OfficialClassificationHead(nn.Module):
    """Flatten all CodeBrain features and apply its official three-layer MLP.

    The official downstream models use
    ``C*P*200 -> P*200 -> 200 -> classes`` with ELU and dropout between
    linear layers.  SHIN changes only ``C`` and the number of one-second
    patches ``P``.
    """

    def __init__(
        self,
        num_channels: int,
        num_patches: int,
        feature_dim: int = 200,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        in_features = num_channels * num_patches * feature_dim
        first_hidden = num_patches * feature_dim
        self.layers = nn.Sequential(
            nn.Linear(in_features, first_hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(first_hidden, feature_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def flatten(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            expected = self.num_channels * self.num_patches * self.feature_dim
            if features.shape[1] != expected:
                raise ValueError(
                    f"expected flattened CodeBrain width {expected}, "
                    f"got {features.shape[1]}"
                )
            return features
        if features.ndim != 4:
            raise ValueError(
                "expected flattened [batch, channels*patches*dim] or CodeBrain features "
                "[batch, channels, patches, dim], "
                f"got {tuple(features.shape)}"
            )
        expected = (self.num_channels, self.num_patches, self.feature_dim)
        if tuple(features.shape[1:]) != expected:
            raise ValueError(
                f"expected CodeBrain features [batch,{expected[0]},{expected[1]},"
                f"{expected[2]}], got {tuple(features.shape)}"
            )
        return features.contiguous().flatten(1)

    # Kept as a compatibility alias for the frozen-backbone feature cache.
    def pool(self, features: torch.Tensor) -> torch.Tensor:
        return self.flatten(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(self.flatten(features))
