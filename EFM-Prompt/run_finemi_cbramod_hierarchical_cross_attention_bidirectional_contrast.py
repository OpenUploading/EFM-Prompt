"""FineMI entry point for the SHIN bidirectional cross-attention variant.

It deliberately reuses the FineMI CBraMod protocol and simply substitutes the
adapter.  This preserves the SHIN variant exactly: ordinary one-way prompt
injection and bidirectional EEG<->fNIRS attention only for the contrastive
sample-distance matrix.
"""

from __future__ import annotations

import run_finemi_cbramod_prompt as shared_runner
from foundation_hierarchical_cross_attention_bidirectional_contrast import (
    FoundationHierarchicalBidirectionalContrastAdapter,
)


def main() -> None:
    shared_runner.FoundationTMPAFinalAdapter = (
        FoundationHierarchicalBidirectionalContrastAdapter
    )
    shared_runner.main()


if __name__ == "__main__":
    main()
