"""Run original Cross-Attention Prompt with bidirectional contrast distance."""

from __future__ import annotations

import run_shin2017_foundation_tmpa_token_alignment as shared_runner

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
