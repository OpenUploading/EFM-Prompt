"""SHIN sensor-node adapter around the repository's official DAMFNet."""

from __future__ import annotations

import torch
import torch.nn as nn

from .fusion_net import MYFusion


class SHINDAMFNet(nn.Module):
    """Adapt SHIN's nodes to DAMFNet's public-dataset node layout.

    The official fusion core expects each 3-second sample as:
      EEG:  [B, 600, 8]
      HbR:  [B, 30, 24]

    ``project_all`` retains all 30/36 SHIN nodes and learns 30->8 and 36->24
    projections. ``damf_fixed`` receives the already selected official-sized
    8/24 node layout and uses no node projection.
    """

    def __init__(
        self,
        dropout: float = 0.4,
        sensor_layout: str = "project_all",
        num_classes: int = 2,
        eeg_input_nodes: int | None = None,
        hbr_input_nodes: int | None = None,
    ) -> None:
        super().__init__()
        self.sensor_layout = sensor_layout
        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if sensor_layout == "project_all":
            self.eeg_input_nodes = int(eeg_input_nodes or 30)
            self.hbr_input_nodes = int(hbr_input_nodes or 36)
            self.eeg_node_projection = nn.Linear(self.eeg_input_nodes, 8, bias=False)
            self.hbr_node_projection = nn.Linear(self.hbr_input_nodes, 24, bias=False)
        elif sensor_layout == "damf_fixed":
            self.eeg_input_nodes = 8
            self.hbr_input_nodes = 24
            self.eeg_node_projection = nn.Identity()
            self.hbr_node_projection = nn.Identity()
        else:
            raise ValueError(f"Unknown sensor layout: {sensor_layout}")
        self.fusion = MYFusion(
            in_places=256,
            places=64,
            OutResTA=True,
            OutTAM=False,
            OutCAM=False,
            OutScale=False,
        )
        # The repository's public fusion head is binary, but its auxiliary
        # branch heads are hard-coded to four outputs.  Make all three heads
        # explicit so SHIN binary and ds004022 four-class experiments share
        # one faithful fusion core.
        self.fusion.fc1_1 = nn.Linear(1024, self.num_classes)
        self.fusion.fc2_1 = nn.Linear(1024, self.num_classes)
        self.fusion.fc2 = nn.Linear(1024, self.num_classes)
        self.fusion.dropout = nn.Dropout(dropout)

    def forward(
        self,
        eeg: torch.Tensor,
        hbr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != (600, self.eeg_input_nodes):
            raise ValueError(
                f"Expected EEG [B,600,{self.eeg_input_nodes}], got {tuple(eeg.shape)}"
            )
        if hbr.ndim != 3 or tuple(hbr.shape[1:]) != (30, self.hbr_input_nodes):
            raise ValueError(
                f"Expected HbR [B,30,{self.hbr_input_nodes}], got {tuple(hbr.shape)}"
            )
        eeg_projected = self.eeg_node_projection(eeg.float())
        hbr_projected = self.hbr_node_projection(hbr.float())
        outputs = self.fusion(
            eeg_projected,
            hbr_projected,
            modality1=False,
            modality2=False,
        )
        eeg_logits, hbr_logits, fusion_logits = outputs[:3]
        return eeg_logits, hbr_logits, fusion_logits
