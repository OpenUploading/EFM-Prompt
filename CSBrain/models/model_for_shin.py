"""Official CSBrain downstream model adapted to the 30-channel SHIN montage."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .CSBrain import CSBrain


SHIN_ELECTRODES = [
    "F7", "AFF5h", "F3", "AFp1", "AFp2", "AFF6h", "F4", "F8",
    "AFF1h", "AFF2h", "Cz", "Pz", "FCC5h", "FCC3h", "CCP5h",
    "CCP3h", "T7", "P7", "P3", "PPO1h", "POO1", "POO2", "PPO2h",
    "P4", "FCC4h", "FCC6h", "CCP4h", "CCP6h", "P8", "T8",
]

# Official convention:
# frontal=0, parietal=1, temporal=2, occipital=3, central=4.
SHIN_BRAIN_REGIONS = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 1, 4, 4, 4, 4, 2, 1, 1, 1,
    3, 3, 1, 1, 4, 4, 4, 4, 1, 2,
]

SHIN_TOPOLOGY = {
    0: ["AFp1", "AFF5h", "AFF1h", "F7", "F3",
        "AFF2h", "F4", "F8", "AFF6h", "AFp2"],
    1: ["P7", "P3", "PPO1h", "Pz", "PPO2h", "P4", "P8"],
    2: ["T7", "T8"],
    3: ["POO1", "POO2"],
    4: ["FCC5h", "FCC3h", "CCP5h", "CCP3h", "Cz",
        "CCP4h", "CCP6h", "FCC4h", "FCC6h"],
}


def shin_sorted_indices() -> list[int]:
    groups: dict[int, list[tuple[int, str]]] = {}
    for index, region in enumerate(SHIN_BRAIN_REGIONS):
        groups.setdefault(region, []).append((index, SHIN_ELECTRODES[index]))
    indices: list[int] = []
    for region in sorted(groups):
        ordered = sorted(
            groups[region],
            key=lambda item: SHIN_TOPOLOGY[region].index(item[1]),
        )
        indices.extend(index for index, _ in ordered)
    if sorted(indices) != list(range(len(SHIN_ELECTRODES))):
        raise RuntimeError("SHIN electrode topology is not a permutation of 0..29")
    return indices


class Model(nn.Module):
    """Official CSBrain backbone plus the official 30ch/10-patch MLP head."""

    def __init__(
        self,
        foundation_dir: str | Path,
        *,
        num_classes: int = 2,
        n_layer: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sorted_indices = shin_sorted_indices()
        self.backbone = CSBrain(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=800,
            seq_len=30,
            n_layer=n_layer,
            nhead=8,
            brain_regions=SHIN_BRAIN_REGIONS,
            sorted_indices=self.sorted_indices,
        )
        self.pretrained_report = self._load_foundation_weights(foundation_dir)
        self.backbone.proj_out = nn.Identity()

        # This follows the official model_for_faced.py head. FACED and SHIN both
        # use 30 channels, 10 one-second patches, and 200 samples per patch.
        self.classifier = nn.Sequential(
            nn.Linear(30 * 10 * 200, 10 * 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(10 * 200, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, num_classes),
        )

    def _load_foundation_weights(self, foundation_dir: str | Path) -> dict:
        checkpoint_path = Path(foundation_dir)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Official CSBrain checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        cleaned = {key.removeprefix("module."): value for key, value in state.items()}
        current = self.backbone.state_dict()
        matched = {
            key: value for key, value in cleaned.items()
            if key in current and current[key].shape == value.shape
        }
        missing = sorted(set(current) - set(matched))
        unexpected = sorted(set(cleaned) - set(current))
        shape_mismatch = sorted(
            key for key in cleaned
            if key in current and current[key].shape != cleaned[key].shape
        )
        if not matched:
            raise RuntimeError("No official CSBrain checkpoint tensors matched the backbone")
        current.update(matched)
        self.backbone.load_state_dict(current)
        return {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_tensor_count": len(cleaned),
            "matched_tensor_count": len(matched),
            "matched_parameter_count": sum(value.numel() for value in matched.values()),
            "backbone_parameter_count": sum(value.numel() for value in current.values()),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "shape_mismatch_keys": shape_mismatch,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (30, 10, 200):
            raise ValueError(f"Expected [B,30,10,200], got {tuple(x.shape)}")
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode(x)
        return self.classifier(features.flatten(1))
