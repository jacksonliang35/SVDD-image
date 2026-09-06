"""Use AestheticScorerLatent_CVaR for new latent checkpoints.
Legacy exports remain for old checkpoints and training scripts.
"""
from aesthetic_scorer_cvar_obsolete import (
    AestheticScorerDiff_CVaR, MLPDiff_CVaR, _get_clip_image_features,
)
from aesthetic_scorer_latent import AestheticScorerLatent_CVaR
