"""Named entry point for the CBraMod spatiotemporal-partition TMPA-lite model.

The implementation remains in ``tmpa_lite_cbramod`` so existing experiments
and commands keep working. This name distinguishes the preserved first-stage
spatial/temporal OT design from later hierarchical TMPA variants.
"""

from tmpa_lite_cbramod import (  # noqa: F401
    FnirsTokenEncoder,
    TMPALiteAdapter,
    sinkhorn_plan,
    transport_features,
)

__all__ = [
    "FnirsTokenEncoder",
    "TMPALiteAdapter",
    "sinkhorn_plan",
    "transport_features",
]
