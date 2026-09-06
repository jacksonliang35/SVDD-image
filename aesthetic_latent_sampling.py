"""Prompt features shared by latent aesthetic training and MC sampling."""
from functools import wraps
import inspect

import torch
from torch.nn import functional as F


@torch.no_grad()
def encode_prompt_features(pipe, prompts):
    inputs = pipe.tokenizer(prompts, padding="max_length",
                            max_length=pipe.tokenizer.model_max_length,
                            truncation=True, return_tensors="pt")
    mask = None
    if getattr(pipe.text_encoder.config, "use_attention_mask", False):
        mask = inputs.attention_mask.to(pipe._execution_device)
    output = pipe.text_encoder(inputs.input_ids.to(pipe._execution_device), attention_mask=mask)
    pooled = getattr(output, "pooler_output", None)
    if pooled is None:
        hidden = output[0]
        if mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
    return F.normalize(pooled.float(), dim=-1)


def uses_latent_aesthetic(pipe):
    return (getattr(pipe, "reward", None) == "aesthetic"
            and getattr(getattr(pipe, "scorer", None), "input_is_latent", False))


def with_aesthetic_prompt_context(function):
    """Cache one feature per actual sample, never including CFG negative prompts.

    `value_prompt_embedding` can supply normalized pooled features when the
    diffusion prompt is provided only through token-level `prompt_embeds`.
    """
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        explicit_feature = kwargs.pop("value_prompt_embedding", None)
        if not uses_latent_aesthetic(self):
            return function(self, *args, **kwargs)
        if self.variant != "MC":
            raise ValueError("Latent aesthetic value scorers require variant='MC'")
        feature = None
        if self.scorer.prompt_conditioned:
            arguments = signature.bind(self, *args, **kwargs)
            arguments.apply_defaults()
            prompt = arguments.arguments.get("prompt")
            if explicit_feature is not None:
                feature = explicit_feature
            else:
                if prompt is None or arguments.arguments.get("prompt_embeds") is not None:
                    raise ValueError("Supply value_prompt_embedding when using precomputed prompt_embeds")
                prompts = [prompt] if isinstance(prompt, str) else list(prompt)
                feature = encode_prompt_features(self, prompts)
            count = arguments.arguments.get("num_images_per_prompt", 1)
            feature = feature.repeat_interleave(count, dim=0)
        sentinel = object()
        previous = getattr(self, "_aesthetic_value_prompt", sentinel)
        self._aesthetic_value_prompt = feature
        try:
            return function(self, *args, **kwargs)
        finally:
            if previous is sentinel:
                delattr(self, "_aesthetic_value_prompt")
            else:
                self._aesthetic_value_prompt = previous
    return wrapped


def score_latent_aesthetic(pipe, latents, timesteps, eta=None):
    if getattr(pipe.scorer, "prompt_conditioned", False) and getattr(pipe, "_aesthetic_value_prompt", None) is None:
        raise ValueError("Prompt-conditioned latent scoring requires the pipeline's sampling context")
    return pipe.scorer(latents, timesteps=timesteps, eta=eta,
                       prompt_embedding=getattr(pipe, "_aesthetic_value_prompt", None))
