"""
Modules for Diffusion-based Action Prediction

This package contains modular components for extracting features from DiT models
and predicting actions from those features.
"""

from .diffusion_feature_extractor import (
    DiffusionFeatureExtractor,
    model_fn_wan_video_with_intermediate_output,
)

from .diffusion_action_decoder import (
    DiffusionActionDecoder,
    compute_action_decoder_loss,
)

from .wan_pipeline_loader import (
    load_wan_pipeline,
)

__all__ = [
    'DiffusionFeatureExtractor',
    'model_fn_wan_video_with_intermediate_output',
    'DiffusionActionDecoder',
    'compute_action_decoder_loss',
    'load_wan_pipeline',
]
