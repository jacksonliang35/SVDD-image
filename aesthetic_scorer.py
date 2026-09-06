"""Compatibility exports; use AestheticScorerLatent for new vanilla MC models.
AestheticScorerDiff remains the terminal CLIP aesthetic oracle.
"""
from aesthetic_scorer_obsolete import (
    AestheticScorerDiff, AestheticScorerDiff_Time, SinusoidalTimeMLP,
    MLPDiff, MLPDiff_class, condition_AestheticScorerDiff,
    classify_aesthetic_scores_easy, _get_clip_image_features,
)
from aesthetic_scorer_latent import AestheticScorerLatent
