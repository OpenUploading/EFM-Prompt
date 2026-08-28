"""PyTorch port of the official ARL EEGNet-8,2 architecture.

Architecture reference:
    vlawhern/arl-eegmodels, EEGModels.py::EEGNet

The upstream repository is TensorFlow/Keras. This module preserves its layer
order and defaults while making the model usable in the existing PyTorch CUDA
environment. It returns logits because PyTorch CrossEntropyLoss includes the
softmax operation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet-8,2 for 30-channel, 200 Hz, 10-second SHIN trials."""

    def __init__(
        self,
        *,
        channels: int = 30,
        samples: int = 2000,
        classes: int = 2,
        dropout: float = 0.5,
        kernel_length: int = 100,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        depthwise_max_norm: float = 1.0,
        classifier_max_norm: float = 0.25,
    ) -> None:
        super().__init__()
        if f2 != f1 * depth_multiplier:
            raise ValueError("Official EEGNet-8,2 uses f2 == f1 * depth_multiplier")
        self.channels = channels
        self.samples = samples
        self.depthwise_max_norm = depthwise_max_norm
        self.classifier_max_norm = classifier_max_norm

        # Official block 1:
        # Conv2D -> BN -> DepthwiseConv2D -> BN -> ELU -> AvgPool(1,4) -> Dropout
        self.temporal_conv = nn.Conv2d(
            1,
            f1,
            kernel_size=(1, kernel_length),
            padding="same",
            bias=False,
        )
        # Keras BatchNormalization defaults: momentum=0.99, epsilon=1e-3.
        # PyTorch's momentum is the new-statistics weight, hence 1 - 0.99.
        self.temporal_bn = nn.BatchNorm2d(f1, momentum=0.01, eps=1e-3)
        self.spatial_depthwise = nn.Conv2d(
            f1,
            f1 * depth_multiplier,
            kernel_size=(channels, 1),
            groups=f1,
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(f1 * depth_multiplier, momentum=0.01, eps=1e-3)
        self.activation1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout)

        # Official block 2 SeparableConv2D = depthwise temporal convolution
        # followed by a 1x1 pointwise convolution.
        self.separable_depthwise = nn.Conv2d(
            f1 * depth_multiplier,
            f1 * depth_multiplier,
            kernel_size=(1, 16),
            padding="same",
            groups=f1 * depth_multiplier,
            bias=False,
        )
        self.separable_pointwise = nn.Conv2d(
            f1 * depth_multiplier,
            f2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.separable_bn = nn.BatchNorm2d(f2, momentum=0.01, eps=1e-3)
        self.activation2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout)

        # Both official pooling layers use valid padding with factors 4 and 8.
        self.feature_dim = int(f2 * ((samples // 4) // 8))
        self.classifier = nn.Linear(self.feature_dim, classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_bn(self.temporal_conv(x))
        x = self.spatial_depthwise(x)
        x = self.dropout1(self.pool1(self.activation1(self.spatial_bn(x))))
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.dropout2(self.pool2(self.activation2(self.separable_bn(x))))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim == 4 and x.shape[1:] == (30, 10, 200):
            x = x.reshape(x.shape[0], 1, 30, 2000)
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, self.channels, self.samples):
            raise ValueError(
                f"Expected [B,{self.channels},{self.samples}] or "
                f"[B,1,{self.channels},{self.samples}], got {tuple(x.shape)}"
            )
        return self.classifier(self._features(x).flatten(1))

    @torch.no_grad()
    def constrain_weights(self) -> None:
        """Apply the two max-norm constraints used by official Keras EEGNet."""

        depthwise = self.spatial_depthwise.weight
        depthwise_norm = depthwise.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
        depthwise_scale = (self.depthwise_max_norm / depthwise_norm).clamp(max=1.0)
        depthwise.mul_(depthwise_scale.view(-1, 1, 1, 1))

        dense = self.classifier.weight
        dense_norm = dense.norm(dim=1, keepdim=True).clamp_min(1e-12)
        dense_scale = (self.classifier_max_norm / dense_norm).clamp(max=1.0)
        dense.mul_(dense_scale)
