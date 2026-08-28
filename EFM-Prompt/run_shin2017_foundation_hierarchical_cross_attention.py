"""Run the no-OT hierarchical cross-attention prompt experiment on SHIN."""

from __future__ import annotations

import run_shin2017_foundation_tmpa_token_alignment as shared_runner

from foundation_hierarchical_cross_attention import (
    FoundationHierarchicalCrossAttentionAdapter,
)


def main() -> None:
    # Reuse the exact data split, backbone interfaces, classifier and training
    # loop of the OT experiment; only the adapter implementation is replaced.
    shared_runner.FoundationTMPAFinalAdapter = FoundationHierarchicalCrossAttentionAdapter
    shared_runner.main()


if __name__ == "__main__":
    main()

