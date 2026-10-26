"""EEG encoders and subject-aware visual routing for THINGS-EEG2."""

from .model import (
    EEGEncoderConfig,
    ModelConfig,
    SAMGAModel,
    VisualRouterConfig,
    count_parameters,
)

__all__ = [
    "EEGEncoderConfig",
    "ModelConfig",
    "SAMGAModel",
    "VisualRouterConfig",
    "count_parameters",
]
