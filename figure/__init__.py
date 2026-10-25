"""Publication figures generated from finalized analysis CSV artifacts."""

from .behavior import BehaviorFigureResult, make_behavior_figures
from .exclusive_channels import (
    ExclusiveChannelFigureResult,
    make_exclusive_channel_figures,
)

__all__ = [
    "BehaviorFigureResult",
    "ExclusiveChannelFigureResult",
    "make_behavior_figures",
    "make_exclusive_channel_figures",
]
