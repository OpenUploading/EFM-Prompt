"""Bidirectional contrast-distance variant of hierarchical cross-attention.

The original one-way token/prompt fusion is intentionally inherited unchanged.
Only the BxB sample-distance matrix used by the class-aware contrastive loss
is replaced: each EEG_i/fNIRS_j pair exchanges information in both directions
before its cosine distance is calculated.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from foundation_hierarchical_cross_attention import (
    FoundationHierarchicalCrossAttentionAdapter,
)


class FoundationHierarchicalBidirectionalContrastAdapter(
    FoundationHierarchicalCrossAttentionAdapter
):
    """Original Cross-Attention Prompt plus bidirectional contrast distance."""

    method_name = "foundation_hierarchical_cross_attention_bidirectional_contrastive"
    alignment_description = (
        "original token-level and prompt-level EEG<-fNIRS cross-attention fusion; "
        "BxB contrast distance from bidirectional EEG<->fNIRS cross-attention"
    )
    metric_note = (
        "paired_prompt_ot/paired_token_ot contain normalized prompt/token attention "
        "entropy from the original fusion; sample_distance uses bidirectional "
        "cross-attention and is not OT."
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dim = self.eeg_projection[1].out_features
        heads = self.token_cross_attention.num_heads
        dropout = self.token_cross_attention.dropout
        self.contrast_eeg_from_fnirs = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.contrast_fnirs_from_eeg = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.contrast_eeg_norm = nn.LayerNorm(dim)
        self.contrast_fnirs_norm = nn.LayerNorm(dim)

    def _sample_distance(
        self,
        eeg_modes: torch.Tensor,
        fnirs_modes: torch.Tensor,
    ) -> torch.Tensor:
        """Build D[i,j] with bidirectional attention for contrastive training.

        This method is only a distance constructor. The inherited forward path
        still injects the original paired, one-way hierarchical prompt, so
        other batch samples can never affect classification logits.
        """
        batch, modes, tokens, dim = eeg_modes.shape
        eeg_pairs = eeg_modes[:, None].expand(-1, batch, -1, -1, -1)
        fnirs_pairs = fnirs_modes[None, :].expand(batch, -1, -1, -1, -1)
        eeg_set = eeg_pairs.reshape(batch * batch, modes * tokens, dim)
        fnirs_set = fnirs_pairs.reshape(batch * batch, modes * tokens, dim)

        eeg_context, _ = self.contrast_eeg_from_fnirs(
            eeg_set, fnirs_set, fnirs_set, need_weights=False
        )
        fnirs_context, _ = self.contrast_fnirs_from_eeg(
            fnirs_set, eeg_set, eeg_set, need_weights=False
        )
        eeg_descriptor = F.normalize(
            self.contrast_eeg_norm(eeg_set + eeg_context).mean(dim=1), dim=-1
        )
        fnirs_descriptor = F.normalize(
            self.contrast_fnirs_norm(fnirs_set + fnirs_context).mean(dim=1), dim=-1
        )
        return (1.0 - (eeg_descriptor * fnirs_descriptor).sum(dim=-1)).clamp_min(0.0).reshape(
            batch, batch
        )
