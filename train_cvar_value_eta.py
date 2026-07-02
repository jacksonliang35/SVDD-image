#!/usr/bin/env python
"""
train_cvar_value_eta.py

Prompt-conditioned trainer for the CVaR value function used by MC-SVDD.

Put this file at the root of the SVDD-image repo, next to:
    sd_pipeline_cvar.py
    aesthetic_scorer.py
    compressibility_scorer.py
    prompts.py

This script trains two objects jointly:

1. eta_net(prompt) -> eta(prompt)

   eta(prompt) minimizes the prompt-conditional pretrained objective

       J_y(eta) = eta - alpha * log E_pre(
           exp(-relu(c(x0) - eta) / (alpha * (1 - beta))) | prompt=y
       ).

   In each training step, the script samples P prompt groups and K pretrained
   rollouts per prompt, giving a Monte Carlo estimate of the expectation for
   each prompt.

2. value_model(x_t, t, prompt) -> tail_reward/log_weight/exp_weight

   The default target is the log-space tail reward

       tail_reward = -relu(c(x0) - eta(prompt)) / (1 - beta).

   This is the recommended target because it is the quantity whose scaled value
   becomes the CVaR log-potential:

       log_weight = tail_reward / alpha.

Important inference note
------------------------
If you use --target_type tail_reward, the trained model already outputs the
CVaR tail reward. In the current sd_pipeline_cvar.py, calculate_mc_cost() treats
MC scorer output as a reward and calculate_weighted_value() may apply another
CVaR hinge. To avoid a double hinge, use the trained prompt-conditioned scorer
as the MC scorer with:

    pipe.set_cvar_lambda(0.0)
    pipe.set_cvar_eta(0.0)

Then the pipeline uses cost = -tail_reward and log_weight = tail_reward / alpha.

The script saves state_dict checkpoints rather than relying on torch.save(model)
only. This is safer for prompt-conditioned modules because inference should
construct the model class explicitly and then load the state_dict.
"""

import argparse
import contextlib
import datetime as _datetime
import io
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler
from PIL import Image

from diffusers_patch.ddim_with_kl import ddim_step_KL
from sd_pipeline_cvar import Decoding_nonbatch_SDPipeline_CVaR
from aesthetic_scorer import AestheticScorerDiff


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


@dataclass
class TrainConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    local_files_only: bool = False
    device: str = "cuda:0"
    diffusion_dtype: str = "float32"
    oracle_dtype: str = "float32"

    reward: str = "aesthetic"  # aesthetic or compressibility
    prompt: Optional[str] = None
    prompt_fn: str = "eval_aesthetic_animals"
    prompt_file: Optional[str] = None
    negative_prompt: Optional[str] = None

    output_dir: str = "prompt_cvar_value_runs"
    run_name: Optional[str] = None
    seed: int = 42

    num_train_steps: int = 1000
    num_prompt_groups: int = 2
    samples_per_prompt: int = 10
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    ddim_eta: float = 1.0
    height: int = 512
    width: int = 512

    alpha: float = 10.0
    beta: float = 0.8
    eta_loss_weight: float = 1.0
    eta_lr: float = 0.1
    value_lr: float = 0.1
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0

    eta_init: float = 100.0
    eta_min: Optional[float] = 0.
    eta_max: Optional[float] = 200.

    # tail_reward is recommended.
    # tail_reward = -relu(c-eta)/(1-beta)
    # log_weight = tail_reward / alpha
    # exp_weight = exp(log_weight)
    target_type: str = "tail_reward"

    # Store this many noisy states per trajectory. Use -1 for all states.
    states_per_trajectory: int = 4
    state_selection: str = "random"  # random or uniform
    include_last_noisy_state: bool = False
    store_latents_dtype: str = "float32"
    value_batch_size: int = 8

    prompt_dim: int = 768
    time_dim: int = 128
    hidden_dim: int = 512
    prompt_channels: int = 32

    save_every: int = 20
    log_every: int = 10
    resume: Optional[str] = None


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train prompt-conditioned eta and CVaR value models for SVDD-image."
    )

    parser.add_argument("--model_id", type=str, default=TrainConfig.model_id)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument(
        "--diffusion_dtype",
        type=str,
        default=TrainConfig.diffusion_dtype,
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--oracle_dtype",
        type=str,
        default=TrainConfig.oracle_dtype,
        choices=["float32", "float16", "bfloat16"],
    )

    parser.add_argument("--reward", type=str, default=TrainConfig.reward, choices=["aesthetic", "compressibility"])
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_fn", type=str, default=TrainConfig.prompt_fn)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)

    parser.add_argument("--num_train_steps", type=int, default=TrainConfig.num_train_steps)
    parser.add_argument("--num_prompt_groups", type=int, default=TrainConfig.num_prompt_groups)
    parser.add_argument("--samples_per_prompt", type=int, default=TrainConfig.samples_per_prompt)
    parser.add_argument("--num_inference_steps", type=int, default=TrainConfig.num_inference_steps)
    parser.add_argument("--guidance_scale", type=float, default=TrainConfig.guidance_scale)
    parser.add_argument("--ddim_eta", type=float, default=TrainConfig.ddim_eta)
    parser.add_argument("--height", type=int, default=TrainConfig.height)
    parser.add_argument("--width", type=int, default=TrainConfig.width)

    parser.add_argument("--alpha", type=float, default=TrainConfig.alpha)
    parser.add_argument("--beta", type=float, default=TrainConfig.beta)
    parser.add_argument("--eta_loss_weight", type=float, default=TrainConfig.eta_loss_weight)
    parser.add_argument("--eta_lr", type=float, default=TrainConfig.eta_lr)
    parser.add_argument("--value_lr", type=float, default=TrainConfig.value_lr)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--grad_clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--eta_init", type=float, default=TrainConfig.eta_init)
    parser.add_argument("--eta_min", type=float, default=None)
    parser.add_argument("--eta_max", type=float, default=None)

    parser.add_argument(
        "--target_type",
        type=str,
        default=TrainConfig.target_type,
        choices=["tail_reward", "tail_cost", "log_weight", "exp_weight"],
    )
    parser.add_argument("--states_per_trajectory", type=int, default=TrainConfig.states_per_trajectory)
    parser.add_argument(
        "--state_selection",
        type=str,
        default=TrainConfig.state_selection,
        choices=["random", "uniform"],
    )
    parser.add_argument("--include_last_noisy_state", action="store_true")
    parser.add_argument(
        "--store_latents_dtype",
        type=str,
        default=TrainConfig.store_latents_dtype,
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--value_batch_size", type=int, default=TrainConfig.value_batch_size)

    parser.add_argument("--prompt_dim", type=int, default=TrainConfig.prompt_dim)
    parser.add_argument("--time_dim", type=int, default=TrainConfig.time_dim)
    parser.add_argument("--hidden_dim", type=int, default=TrainConfig.hidden_dim)
    parser.add_argument("--prompt_channels", type=int, default=TrainConfig.prompt_channels)

    parser.add_argument("--save_every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--log_every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    return TrainConfig(
        model_id=args.model_id,
        local_files_only=bool(args.local_files_only),
        device=args.device,
        diffusion_dtype=args.diffusion_dtype,
        oracle_dtype=args.oracle_dtype,
        reward=args.reward,
        prompt=args.prompt,
        prompt_fn=args.prompt_fn,
        prompt_file=args.prompt_file,
        negative_prompt=args.negative_prompt,
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        num_train_steps=args.num_train_steps,
        num_prompt_groups=args.num_prompt_groups,
        samples_per_prompt=args.samples_per_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        ddim_eta=args.ddim_eta,
        height=args.height,
        width=args.width,
        alpha=args.alpha,
        beta=args.beta,
        eta_loss_weight=args.eta_loss_weight,
        eta_lr=args.eta_lr,
        value_lr=args.value_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eta_init=args.eta_init,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        target_type=args.target_type,
        states_per_trajectory=args.states_per_trajectory,
        state_selection=args.state_selection,
        include_last_noisy_state=bool(args.include_last_noisy_state),
        store_latents_dtype=args.store_latents_dtype,
        value_batch_size=args.value_batch_size,
        prompt_dim=args.prompt_dim,
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim,
        prompt_channels=args.prompt_channels,
        save_every=args.save_every,
        log_every=args.log_every,
        resume=args.resume,
    )


def get_torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {name}")


def set_seed(seed: int) -> None:
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(cfg: TrainConfig) -> str:
    if cfg.run_name is None:
        stamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.run_name = f"prompt_cvar_{cfg.reward}_{cfg.target_type}_{stamp}"
    run_dir = os.path.join(cfg.output_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def load_prompt_file(path: str) -> List[str]:
    if not os.path.exists(path):
        assets_path = os.path.join("assets", path)
        if os.path.exists(assets_path):
            path = assets_path
    with open(path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    if not prompts:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompts


def sample_prompt_groups(cfg: TrainConfig, num_groups: int) -> List[str]:
    if cfg.prompt is not None:
        return [cfg.prompt] * num_groups

    if cfg.prompt_file is not None:
        prompt_pool = load_prompt_file(cfg.prompt_file)
        return [random.choice(prompt_pool) for _ in range(num_groups)]

    try:
        import prompts as prompts_file
    except Exception as exc:
        raise RuntimeError(
            "Could not import prompts.py. Pass --prompt, --prompt_file, or run from the repo root."
        ) from exc

    if not hasattr(prompts_file, cfg.prompt_fn):
        raise AttributeError(f"prompts.py has no function named {cfg.prompt_fn!r}")

    prompt_fn = getattr(prompts_file, cfg.prompt_fn)
    out: List[str] = []
    for _ in range(num_groups):
        value = prompt_fn()
        if isinstance(value, (tuple, list)):
            out.append(str(value[0]))
        else:
            out.append(str(value))
    return out


def repeat_prompts(prompt_groups: Sequence[str], samples_per_prompt: int) -> Tuple[List[str], torch.Tensor]:
    flat_prompts: List[str] = []
    group_ids: List[int] = []
    for group_idx, prompt in enumerate(prompt_groups):
        for _ in range(samples_per_prompt):
            flat_prompts.append(str(prompt))
            group_ids.append(group_idx)
    return flat_prompts, torch.tensor(group_ids, dtype=torch.long)


def make_negative_prompt_batch(cfg: TrainConfig, total_batch: int) -> Optional[List[str]]:
    if cfg.negative_prompt is None:
        return None
    return [cfg.negative_prompt] * total_batch


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dimension must be even")
        self.dim = int(dim)
        frequencies = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / float(dim))
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, timesteps: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        t = timesteps.float().reshape(-1) / 1000.0
        freq = self.frequencies.to(device=t.device)
        args = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if dtype is not None:
            emb = emb.to(dtype=dtype)
        return emb


class PromptEtaNet(nn.Module):
    def __init__(
        self,
        prompt_dim: int = 768,
        hidden_dim: int = 256,
        eta_init: float = 100.0,
        eta_min: Optional[float] = None,
        eta_max: Optional[float] = None,
    ):
        super().__init__()
        self.prompt_dim = int(prompt_dim)
        self.hidden_dim = int(hidden_dim)
        self.eta_init = float(eta_init)
        self.eta_min = None if eta_min is None else float(eta_min)
        self.eta_max = None if eta_max is None else float(eta_max)

        if (self.eta_min is None) != (self.eta_max is None):
            raise ValueError("eta_min and eta_max must be both None or both set")
        if self.eta_min is not None and self.eta_min >= self.eta_max:
            raise ValueError("eta_min must be less than eta_max")

        self.net = nn.Sequential(
            nn.LayerNorm(self.prompt_dim),
            nn.Linear(self.prompt_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.reset_eta_initialization(self.eta_init)
    
    def reset_eta_initialization(self, eta_init: float) -> None:
        eta_init = float(eta_init)
        last = self.net[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError("Expected final eta layer to be nn.Linear")

        nn.init.zeros_(last.weight)

        if self.eta_min is None:
            bias_value = eta_init
        else:
            if not (self.eta_min < eta_init < self.eta_max):
                raise ValueError(
                    f"eta_init={eta_init} must be strictly between "
                    f"eta_min={self.eta_min} and eta_max={self.eta_max}."
                )
            p = (eta_init - self.eta_min) / (self.eta_max - self.eta_min)
            bias_value = math.log(p / (1.0 - p))

        with torch.no_grad():
            last.bias.fill_(bias_value)

    def forward(self, prompt_embed: torch.Tensor) -> torch.Tensor:
        raw = self.net(prompt_embed.float()).squeeze(-1)
        if self.eta_min is None:
            return raw
        return self.eta_min + (self.eta_max - self.eta_min) * torch.sigmoid(raw)


class PromptConditionedAestheticValueNet(nn.Module):
    """Value head for MC aesthetic.

    Input convention:
        image_embed:  CLIP image feature, shape [B, 768]
        prompt_embed: pooled SD text feature, shape [B, 768]
        timesteps:    diffusion scheduler timesteps, shape [B]
    """

    def __init__(self, image_dim: int = 768, prompt_dim: int = 768, time_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.image_dim = int(image_dim)
        self.prompt_dim = int(prompt_dim)
        self.time_dim = int(time_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(image_dim + prompt_dim + time_dim),
            nn.Linear(image_dim + prompt_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, image_embed: torch.Tensor, prompt_embed: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = image_embed.dtype
        t_embed = self.time_embed(timesteps, dtype=dtype)
        x = torch.cat([image_embed, prompt_embed.to(dtype=dtype), t_embed], dim=1)
        return self.net(x.float()).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return out


class PromptConditionedTimeConvNet(nn.Module):
    """Value head for MC compressibility.

    Input convention:
        latents:      SD latent x_t, shape [B,4,64,64]
        prompt_embed: pooled SD text feature, shape [B,768]
        timesteps:    diffusion scheduler timesteps, shape [B]
    """

    def __init__(
        self,
        latent_channels: int = 4,
        prompt_dim: int = 768,
        time_dim: int = 128,
        prompt_channels: int = 32,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.prompt_dim = int(prompt_dim)
        self.time_dim = int(time_dim)
        self.prompt_channels = int(prompt_channels)

        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(prompt_dim),
            nn.Linear(prompt_dim, prompt_channels),
            nn.SiLU(),
        )

        self.layer1 = ResidualBlock(latent_channels, 64, stride=1)
        self.layer2 = ResidualBlock(64 + time_dim + prompt_channels, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1)

    def forward(self, latents: torch.Tensor, prompt_embed: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        x = latents.float()
        out = self.layer1(x)
        b, _, h, w = out.shape

        t_embed = self.time_embed(timesteps, dtype=out.dtype).view(b, self.time_dim, 1, 1)
        t_map = t_embed.expand(b, self.time_dim, h, w)

        p_embed = self.prompt_proj(prompt_embed.float()).to(dtype=out.dtype).view(b, self.prompt_channels, 1, 1)
        p_map = p_embed.expand(b, self.prompt_channels, h, w)

        out = torch.cat([out, t_map, p_map], dim=1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out).flatten(1)
        return self.fc(out).squeeze(-1)


# ---------------------------------------------------------------------------
# Optional inference scorers for the current pipeline interface
# ---------------------------------------------------------------------------


class PromptConditionedAestheticCvarScorer(nn.Module):
    """Scorer wrapper compatible with calculate_mc_cost(..., reward='aesthetic').

    The pipeline's MC aesthetic branch passes CLIP-preprocessed image tensors
    and timesteps to scorer(images, timesteps). This wrapper adds the prompt
    feature internally. Call set_prompt_features(...) before sampling.
    """

    def __init__(self, value_model: PromptConditionedAestheticValueNet, eta_net: Optional[PromptEtaNet] = None):
        super().__init__()
        self.clip = None
        self.value_model = value_model
        self.eta_net = eta_net
        self.register_buffer("prompt_features", torch.empty(0), persistent=False)

    def attach_clip_from_aesthetic_scorer(self, aesthetic_scorer: AestheticScorerDiff) -> None:
        self.clip = aesthetic_scorer.clip
        self.clip.requires_grad_(False)
        self.clip.eval()

    def set_prompt_features(self, prompt_features: torch.Tensor) -> None:
        self.prompt_features = prompt_features.detach().float()

    def _prompt_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.prompt_features.numel() == 0:
            raise RuntimeError("Call scorer.set_prompt_features(...) before sampling.")
        p = self.prompt_features.to(device=device)
        if p.shape[0] == batch_size:
            return p
        if p.shape[0] == 1:
            return p.expand(batch_size, -1)
        if batch_size % p.shape[0] == 0:
            return p.repeat_interleave(batch_size // p.shape[0], dim=0)
        raise ValueError(f"Cannot broadcast {p.shape[0]} prompt features to batch size {batch_size}.")

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor, timesteps: torch.Tensor):
        if self.clip is None:
            raise RuntimeError("Attach a CLIP image encoder first with attach_clip_from_aesthetic_scorer(...).")
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        prompt_batch = self._prompt_batch(embed.shape[0], embed.device)
        values = self.value_model(embed.float(), prompt_batch, timesteps).reshape(-1)
        return values, embed


class PromptConditionedCompressibilityCvarScorer(nn.Module):
    """Scorer wrapper compatible with calculate_mc_cost(..., reward='compressibility')."""

    def __init__(self, value_model: PromptConditionedTimeConvNet, eta_net: Optional[PromptEtaNet] = None):
        super().__init__()
        self.value_model = value_model
        self.eta_net = eta_net
        self.register_buffer("prompt_features", torch.empty(0), persistent=False)

    def set_prompt_features(self, prompt_features: torch.Tensor) -> None:
        self.prompt_features = prompt_features.detach().float()

    def _prompt_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.prompt_features.numel() == 0:
            raise RuntimeError("Call scorer.set_prompt_features(...) before sampling.")
        p = self.prompt_features.to(device=device)
        if p.shape[0] == batch_size:
            return p
        if p.shape[0] == 1:
            return p.expand(batch_size, -1)
        if batch_size % p.shape[0] == 0:
            return p.repeat_interleave(batch_size // p.shape[0], dim=0)
        raise ValueError(f"Cannot broadcast {p.shape[0]} prompt features to batch size {batch_size}.")

    @torch.inference_mode()
    def __call__(self, latents: torch.Tensor, timesteps: torch.Tensor):
        prompt_batch = self._prompt_batch(latents.shape[0], latents.device)
        values = self.value_model(latents.float(), prompt_batch, timesteps).reshape(-1)
        return values, latents


# ---------------------------------------------------------------------------
# Pipeline / data helpers
# ---------------------------------------------------------------------------


def clip_preprocess_tensor(images_01: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(images_01, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor(CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def decode_latents_to_tensor(pipe: Decoding_nonbatch_SDPipeline_CVaR, latents: torch.Tensor) -> torch.Tensor:
    scaling = getattr(pipe.vae.config, "scaling_factor", 0.18215)
    images = pipe.vae.decode(latents.to(pipe.vae.dtype) / scaling).sample
    return ((images / 2.0) + 0.5).clamp(0.0, 1.0)


def jpeg_compressibility_reward(images_01: torch.Tensor) -> np.ndarray:
    """Return reward = -JPEG_size_kb for images in [0,1], shape [B,3,H,W]."""
    arr = (images_01.detach().float().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
    arr = np.transpose(arr, (0, 2, 3, 1))
    rewards: List[float] = []
    with contextlib.ExitStack() as stack:
        buffers = [stack.enter_context(io.BytesIO()) for _ in range(arr.shape[0])]
        for image_arr, buffer in zip(arr, buffers):
            Image.fromarray(image_arr).save(buffer, format="JPEG", quality=95)
            rewards.append(-float(buffer.tell()) / 1000.0)
    return np.asarray(rewards, dtype=np.float32)


@torch.inference_mode()
def score_final_costs(
    cfg: TrainConfig,
    pipe: Decoding_nonbatch_SDPipeline_CVaR,
    aesthetic_oracle: Optional[AestheticScorerDiff],
    final_latents: torch.Tensor,
) -> torch.Tensor:
    images_01 = decode_latents_to_tensor(pipe, final_latents)
    device = final_latents.device

    if cfg.reward == "compressibility":
        rewards_np = jpeg_compressibility_reward(images_01)
        rewards = torch.as_tensor(rewards_np, device=device, dtype=torch.float32)

    elif cfg.reward == "aesthetic":
        if aesthetic_oracle is None:
            raise RuntimeError("aesthetic_oracle is required for reward='aesthetic'.")
        oracle_dtype = next(aesthetic_oracle.parameters()).dtype
        images_clip = clip_preprocess_tensor(images_01.float()).to(device=device, dtype=oracle_dtype)
        rewards, _embed = aesthetic_oracle(images_clip)
        rewards = rewards.detach().float().reshape(-1)
        del _embed

    else:
        raise ValueError(f"Unknown reward: {cfg.reward}")

    return -rewards.reshape(-1).float()


@torch.no_grad()
def encode_prompt_features(
    pipe: Decoding_nonbatch_SDPipeline_CVaR,
    prompts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Mean-pool SD text encoder hidden states into one prompt feature per prompt."""
    text_inputs = pipe.tokenizer(
        list(prompts),
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    try:
        outputs = pipe.text_encoder(input_ids, attention_mask=attention_mask)
    except TypeError:
        outputs = pipe.text_encoder(input_ids)

    hidden = outputs[0].float()
    mask = attention_mask.float().unsqueeze(-1)
    pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
    return pooled.detach().float()


def choose_record_indices(num_steps: int, cfg: TrainConfig) -> List[int]:
    last_allowed_exclusive = num_steps if cfg.include_last_noisy_state else max(1, num_steps - 1)
    candidates = list(range(last_allowed_exclusive))
    if cfg.states_per_trajectory < 0 or cfg.states_per_trajectory >= len(candidates):
        return candidates
    k = int(cfg.states_per_trajectory)
    if cfg.state_selection == "uniform":
        if k == 1:
            return [candidates[len(candidates) // 2]]
        idx = np.linspace(0, len(candidates) - 1, k).round().astype(int).tolist()
        return sorted({candidates[i] for i in idx})
    return sorted(random.sample(candidates, k))


@torch.inference_mode()
def rollout_pretrained_batch(
    cfg: TrainConfig,
    pipe: Decoding_nonbatch_SDPipeline_CVaR,
    aesthetic_oracle: Optional[AestheticScorerDiff],
    flat_prompts: List[str],
    negative_prompts: Optional[List[str]],
    generator: Optional[torch.Generator],
) -> Tuple[List[Tuple[int, torch.Tensor]], torch.Tensor]:
    """Roll out pretrained SD and collect intermediate x_t states.

    Returns:
        recorded_states:
            list of (timestep_int, latents_cpu) where latents_cpu has shape
            [total_batch, 4, 64, 64].
        final_costs:
            tensor [total_batch] on GPU, cost = -reward.
    """
    device = pipe._execution_device
    diffusion_dtype = next(pipe.unet.parameters()).dtype
    store_dtype = get_torch_dtype(cfg.store_latents_dtype)
    total_batch = len(flat_prompts)
    do_cfg = cfg.guidance_scale > 1.0

    prompt_embeds = pipe._encode_prompt(
        flat_prompts,
        device,
        1,
        do_cfg,
        negative_prompts,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    )

    pipe.scheduler.set_timesteps(cfg.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    record_indices = set(choose_record_indices(len(timesteps), cfg))

    latents = pipe.prepare_latents(
        total_batch,
        pipe.unet.config.in_channels,
        cfg.height,
        cfg.width,
        diffusion_dtype,
        device,
        generator,
        None,
    )

    recorded_states: List[Tuple[int, torch.Tensor]] = []

    for step_index, t in enumerate(timesteps):
        if step_index in record_indices:
            recorded_states.append((int(t.detach().cpu().item()), latents.detach().to("cpu", dtype=store_dtype)))

        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        noise_pred_raw = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
        ).sample

        if do_cfg:
            noise_uncond, noise_text = noise_pred_raw.chunk(2)
            noise_pred = noise_uncond + cfg.guidance_scale * (noise_text - noise_uncond)
            del noise_uncond, noise_text
        else:
            noise_pred = noise_pred_raw

        # The pretrained rollout uses the same DDIM transition utility as your
        # SVDD code. Passing noise_pred twice gives the ordinary pretrained step
        # with a zero KL-control correction.
        latents, kl_terms = ddim_step_KL(
            pipe.scheduler,
            noise_pred,
            noise_pred,
            t,
            latents,
            eta=cfg.ddim_eta,
        )

        del latent_model_input, noise_pred_raw, noise_pred, kl_terms

    final_costs = score_final_costs(cfg, pipe, aesthetic_oracle, latents)
    del latents
    return recorded_states, final_costs


@torch.no_grad()
def latent_state_to_clip_embed(
    pipe: Decoding_nonbatch_SDPipeline_CVaR,
    aesthetic_oracle: AestheticScorerDiff,
    latents: torch.Tensor,
) -> torch.Tensor:
    images_01 = decode_latents_to_tensor(pipe, latents)
    oracle_dtype = next(aesthetic_oracle.parameters()).dtype
    images_clip = clip_preprocess_tensor(images_01.float()).to(device=latents.device, dtype=oracle_dtype)
    embed = aesthetic_oracle.clip.get_image_features(pixel_values=images_clip)
    embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
    return embed.detach().clone().float()


# ---------------------------------------------------------------------------
# CVaR objective and targets
# ---------------------------------------------------------------------------


def prompt_group_eta_objective(
    costs_pk: torch.Tensor,
    eta_p: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Prompt-grouped eta objective.

    costs_pk: [P,K]
    eta_p:    [P]

    Returns [P], one empirical objective per prompt group.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if not (0.0 <= beta < 1.0):
        raise ValueError("beta must be in [0,1)")
    denom = float(alpha) * (1.0 - float(beta))
    log_terms = -F.relu(costs_pk.float() - eta_p.float().view(-1, 1)) / denom
    log_mean_exp = torch.logsumexp(log_terms, dim=1) - math.log(costs_pk.shape[1])
    return eta_p.float() - float(alpha) * log_mean_exp


@torch.no_grad()
def eta_group_diagnostics(costs_pk: torch.Tensor, eta_p: torch.Tensor, alpha: float, beta: float) -> Dict[str, float]:
    denom = float(alpha) * (1.0 - float(beta))
    log_terms = -F.relu(costs_pk.float() - eta_p.float().view(-1, 1)) / denom
    probs = torch.softmax(log_terms, dim=1)
    tail = (costs_pk.float() > eta_p.float().view(-1, 1)).float()
    tilted_tail_prob = torch.sum(probs * tail, dim=1)
    grad = 1.0 - tilted_tail_prob / (1.0 - float(beta))
    return {
        "eta_mean": float(eta_p.mean().detach().cpu()),
        "eta_std": float(eta_p.std(unbiased=False).detach().cpu()),
        "eta_min": float(eta_p.min().detach().cpu()),
        "eta_max": float(eta_p.max().detach().cpu()),
        "eta_grad_mean": float(grad.mean().detach().cpu()),
        "eta_grad_abs_mean": float(grad.abs().mean().detach().cpu()),
        "tilted_tail_prob_mean": float(tilted_tail_prob.mean().detach().cpu()),
        "target_tail_prob": float(1.0 - float(beta)),
    }


def make_targets(
    costs_b: torch.Tensor,
    eta_b_detached: torch.Tensor,
    alpha: float,
    beta: float,
    target_type: str,
) -> torch.Tensor:
    tail_cost = F.relu(costs_b.float() - eta_b_detached.float()) / (1.0 - float(beta))
    tail_reward = -tail_cost
    if target_type == "tail_reward":
        return tail_reward
    if target_type == "tail_cost":
        return tail_cost
    if target_type == "log_weight":
        return tail_reward / float(alpha)
    if target_type == "exp_weight":
        return torch.exp(tail_reward / float(alpha))
    raise ValueError(f"Unknown target_type: {target_type}")


# ---------------------------------------------------------------------------
# Build, save, resume
# ---------------------------------------------------------------------------


def build_trainable_models(cfg: TrainConfig, device: torch.device) -> Tuple[PromptEtaNet, nn.Module]:
    eta_net = PromptEtaNet(
        prompt_dim=cfg.prompt_dim,
        hidden_dim=max(128, cfg.hidden_dim // 2),
        eta_init=cfg.eta_init,
        eta_min=cfg.eta_min,
        eta_max=cfg.eta_max,
    ).to(device=device, dtype=torch.float32)

    if cfg.reward == "aesthetic":
        value_model: nn.Module = PromptConditionedAestheticValueNet(
            image_dim=768,
            prompt_dim=cfg.prompt_dim,
            time_dim=cfg.time_dim,
            hidden_dim=cfg.hidden_dim,
        )
    elif cfg.reward == "compressibility":
        value_model = PromptConditionedTimeConvNet(
            latent_channels=4,
            prompt_dim=cfg.prompt_dim,
            time_dim=cfg.time_dim,
            prompt_channels=cfg.prompt_channels,
        )
    else:
        raise ValueError(f"Unknown reward: {cfg.reward}")

    value_model = value_model.to(device=device, dtype=torch.float32)
    return eta_net, value_model


def build_aesthetic_oracle(cfg: TrainConfig, device: torch.device) -> Optional[AestheticScorerDiff]:
    if cfg.reward != "aesthetic":
        return None
    oracle_dtype = get_torch_dtype(cfg.oracle_dtype)
    scorer = AestheticScorerDiff(dtype=oracle_dtype).to(device=device)
    if oracle_dtype != torch.float32:
        scorer = scorer.to(dtype=oracle_dtype)
    scorer.requires_grad_(False)
    scorer.eval()
    return scorer


def save_checkpoint(
    run_dir: str,
    cfg: TrainConfig,
    eta_net: PromptEtaNet,
    value_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: Dict[str, Any],
) -> None:
    os.makedirs(run_dir, exist_ok=True)
    model_type = "aesthetic" if cfg.reward == "aesthetic" else "compressibility"

    latest = {
        "step": int(step),
        "config": asdict(cfg),
        "model_type": model_type,
        "target_type": cfg.target_type,
        "eta_net_state_dict": eta_net.state_dict(),
        "value_model_state_dict": value_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "model_config": {
            "prompt_dim": cfg.prompt_dim,
            "time_dim": cfg.time_dim,
            "hidden_dim": cfg.hidden_dim,
            "prompt_channels": cfg.prompt_channels,
            "eta_min": cfg.eta_min,
            "eta_max": cfg.eta_max,
        },
    }

    torch.save(latest, os.path.join(run_dir, "checkpoint_latest.pt"))
    torch.save(eta_net.state_dict(), os.path.join(run_dir, "eta_net_latest.pt"))
    torch.save(value_model.state_dict(), os.path.join(run_dir, "prompt_cvar_value_model_latest.pt"))

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    with open(os.path.join(run_dir, "latest_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_resume_if_needed(
    cfg: TrainConfig,
    eta_net: PromptEtaNet,
    value_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    if cfg.resume is None:
        return 0
    ckpt = torch.load(cfg.resume, map_location=device)
    eta_net.load_state_dict(ckpt["eta_net_state_dict"])
    value_model.load_state_dict(ckpt["value_model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt.get("step", 0))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def compute_value_loss_for_states(
    cfg: TrainConfig,
    pipe: Decoding_nonbatch_SDPipeline_CVaR,
    aesthetic_oracle: Optional[AestheticScorerDiff],
    value_model: nn.Module,
    recorded_states: List[Tuple[int, torch.Tensor]],
    prompt_features_b: torch.Tensor,
    targets_b: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    if not recorded_states:
        raise RuntimeError("No recorded states. Increase --states_per_trajectory or check timestep settings.")

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_count = 0
    value_bs = max(1, int(cfg.value_batch_size))

    for timestep_int, latents_cpu in recorded_states:
        batch_n = int(latents_cpu.shape[0])
        for start in range(0, batch_n, value_bs):
            end = min(batch_n, start + value_bs)
            latents_t = latents_cpu[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
            prompt_t = prompt_features_b[start:end].to(device=device, dtype=torch.float32)
            target_t = targets_b[start:end].to(device=device, dtype=torch.float32)
            timesteps_t = torch.full((end - start,), int(timestep_int), device=device, dtype=torch.long)

            if cfg.reward == "aesthetic":
                if aesthetic_oracle is None:
                    raise RuntimeError("aesthetic_oracle is required for reward='aesthetic'.")
                with torch.inference_mode():
                    image_embed = latent_state_to_clip_embed(pipe, aesthetic_oracle, latents_t)
                pred = value_model(image_embed, prompt_t, timesteps_t).reshape(-1)
            else:
                pred = value_model(latents_t, prompt_t, timesteps_t).reshape(-1)

            loss_sum = F.mse_loss(pred.float(), target_t.float(), reduction="sum")
            total_loss = total_loss + loss_sum
            total_count += int(end - start)

            del latents_t, prompt_t, target_t, timesteps_t, pred, loss_sum

    return total_loss / float(max(1, total_count)), total_count


def train(cfg: TrainConfig) -> None:
    if cfg.alpha <= 0:
        raise ValueError("alpha must be positive")
    if not (0.0 <= cfg.beta < 1.0):
        raise ValueError("beta must be in [0,1)")
    if cfg.num_prompt_groups < 1:
        raise ValueError("num_prompt_groups must be at least 1")
    if cfg.samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be at least 1")
    if cfg.value_batch_size < 1:
        raise ValueError("value_batch_size must be at least 1")

    set_seed(cfg.seed)
    run_dir = make_run_dir(cfg)
    device = torch.device(cfg.device)
    diffusion_dtype = get_torch_dtype(cfg.diffusion_dtype)

    print(f"Run directory: {run_dir}")
    print(f"Loading Stable Diffusion pipeline: {cfg.model_id}")

    pipe = Decoding_nonbatch_SDPipeline_CVaR.from_pretrained(
        cfg.model_id,
        torch_dtype=diffusion_dtype,
        local_files_only=cfg.local_files_only,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.unet.eval()

    aesthetic_oracle = build_aesthetic_oracle(cfg, device)
    eta_net, value_model = build_trainable_models(cfg, device)
    eta_net.train()
    value_model.train()

    optimizer = torch.optim.AdamW(
        [
            {"params": value_model.parameters(), "lr": cfg.value_lr, "weight_decay": cfg.weight_decay},
            {"params": eta_net.parameters(), "lr": cfg.eta_lr, "weight_decay": 0.0},
        ]
    )

    start_step = load_resume_if_needed(cfg, eta_net, value_model, optimizer, device)

    generator = torch.Generator(device=device)
    if cfg.seed >= 0:
        generator.manual_seed(cfg.seed + 12345)

    log_path = os.path.join(run_dir, "train_log.jsonl")
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    try:
        from tqdm.auto import trange
        iterator = trange(start_step + 1, cfg.num_train_steps + 1, desc="Training prompt-CVaR value")
    except Exception:
        iterator = range(start_step + 1, cfg.num_train_steps + 1)

    for step in iterator:
        prompt_groups = sample_prompt_groups(cfg, cfg.num_prompt_groups)
        flat_prompts, group_ids_cpu = repeat_prompts(prompt_groups, cfg.samples_per_prompt)
        negative_prompts = make_negative_prompt_batch(cfg, len(flat_prompts))
        group_ids = group_ids_cpu.to(device=device)

        prompt_features_p = encode_prompt_features(pipe, prompt_groups, device=device)
        prompt_features_p = prompt_features_p.detach().clone().to(
            device=device,
            dtype=torch.float32,
        )
        if prompt_features_p.shape[1] != cfg.prompt_dim:
            raise ValueError(
                f"Prompt feature dimension is {prompt_features_p.shape[1]}, but cfg.prompt_dim={cfg.prompt_dim}. "
                "Set --prompt_dim to match the text encoder hidden size."
            )
        prompt_features_b = prompt_features_p.index_select(0, group_ids)

        recorded_states, final_costs = rollout_pretrained_batch(
            cfg=cfg,
            pipe=pipe,
            aesthetic_oracle=aesthetic_oracle,
            flat_prompts=flat_prompts,
            negative_prompts=negative_prompts,
            generator=generator,
        )
        final_costs = final_costs.detach().clone().to(device=device, dtype=torch.float32)
        costs_pk = final_costs.reshape(cfg.num_prompt_groups, cfg.samples_per_prompt)

        optimizer.zero_grad(set_to_none=True)

        eta_p = eta_net(prompt_features_p)
        eta_obj_p = prompt_group_eta_objective(costs_pk, eta_p, cfg.alpha, cfg.beta)
        eta_loss = eta_obj_p.mean()

        eta_b_detached = eta_p.detach().index_select(0, group_ids)
        targets_b = make_targets(
            costs_b=final_costs,
            eta_b_detached=eta_b_detached,
            alpha=cfg.alpha,
            beta=cfg.beta,
            target_type=cfg.target_type,
        )

        value_loss, num_value_terms = compute_value_loss_for_states(
            cfg=cfg,
            pipe=pipe,
            aesthetic_oracle=aesthetic_oracle,
            value_model=value_model,
            recorded_states=recorded_states,
            prompt_features_b=prompt_features_b,
            targets_b=targets_b,
            device=device,
        )

        loss = value_loss + float(cfg.eta_loss_weight) * eta_loss
        loss.backward()

        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), float(cfg.grad_clip))
            torch.nn.utils.clip_grad_norm_(eta_net.parameters(), float(cfg.grad_clip))

        optimizer.step()

        with torch.no_grad():
            diag = eta_group_diagnostics(costs_pk, eta_p, cfg.alpha, cfg.beta)
            metrics: Dict[str, Any] = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "eta_loss": float(eta_loss.detach().cpu()),
                "cost_mean": float(final_costs.mean().detach().cpu()),
                "cost_std": float(final_costs.std(unbiased=False).detach().cpu()),
                "cost_min": float(final_costs.min().detach().cpu()),
                "cost_max": float(final_costs.max().detach().cpu()),
                "target_mean": float(targets_b.mean().detach().cpu()),
                "target_std": float(targets_b.std(unbiased=False).detach().cpu()),
                "num_prompt_groups": int(cfg.num_prompt_groups),
                "samples_per_prompt": int(cfg.samples_per_prompt),
                "total_rollout_batch": int(len(flat_prompts)),
                "num_recorded_state_batches": int(len(recorded_states)),
                "num_value_terms": int(num_value_terms),
                "example_prompt": prompt_groups[0] if prompt_groups else "",
            }
            metrics.update(diag)

        if step == 1 or step % cfg.log_every == 0:
            print(
                "step={step} loss={loss:.4f} value_loss={value_loss:.4f} "
                "eta_loss={eta_loss:.4f} eta_mean={eta_mean:.4f} "
                "cost_mean={cost_mean:.4f} tail_p_tilted={tilted_tail_prob_mean:.3f}".format(**metrics)
            )
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")

        if step % cfg.save_every == 0 or step == cfg.num_train_steps:
            save_checkpoint(
                run_dir=run_dir,
                cfg=cfg,
                eta_net=eta_net,
                value_model=value_model,
                optimizer=optimizer,
                step=step,
                metrics=metrics,
            )

        del (
            prompt_features_p,
            prompt_features_b,
            group_ids,
            recorded_states,
            final_costs,
            costs_pk,
            eta_p,
            eta_obj_p,
            eta_loss,
            eta_b_detached,
            targets_b,
            value_loss,
            loss,
        )

    print("Training finished.")
    print(f"Saved checkpoint: {os.path.join(run_dir, 'checkpoint_latest.pt')}")
    print(f"Saved value state_dict: {os.path.join(run_dir, 'prompt_cvar_value_model_latest.pt')}")
    print(f"Saved eta state_dict: {os.path.join(run_dir, 'eta_net_latest.pt')}")
    print("")
    print("Inference reminder:")
    print("  The default target is tail_reward = -relu(c-eta(prompt))/(1-beta).")
    print("  Use pipe.set_cvar_lambda(0.0) and pipe.set_cvar_eta(0.0) to avoid a double CVaR hinge.")
    print("  Before sampling, set prompt features on the prompt-conditioned scorer.")


# ---------------------------------------------------------------------------
# Loading helpers for inference scripts
# ---------------------------------------------------------------------------


def build_models_from_checkpoint(ckpt_path: str, device: str = "cuda") -> Tuple[PromptEtaNet, nn.Module, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["config"]
    model_cfg = ckpt.get("model_config", {})
    prompt_dim = int(model_cfg.get("prompt_dim", cfg_dict.get("prompt_dim", 768)))
    time_dim = int(model_cfg.get("time_dim", cfg_dict.get("time_dim", 128)))
    hidden_dim = int(model_cfg.get("hidden_dim", cfg_dict.get("hidden_dim", 512)))
    prompt_channels = int(model_cfg.get("prompt_channels", cfg_dict.get("prompt_channels", 32)))
    eta_min = model_cfg.get("eta_min", cfg_dict.get("eta_min", None))
    eta_max = model_cfg.get("eta_max", cfg_dict.get("eta_max", None))

    eta_net = PromptEtaNet(
        prompt_dim=prompt_dim,
        hidden_dim=max(128, hidden_dim // 2),
        eta_min=eta_min,
        eta_max=eta_max,
    ).to(device=device, dtype=torch.float32)
    eta_net.load_state_dict(ckpt["eta_net_state_dict"])
    eta_net.eval()

    reward = cfg_dict["reward"]
    if reward == "aesthetic":
        value_model: nn.Module = PromptConditionedAestheticValueNet(
            image_dim=768,
            prompt_dim=prompt_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
        )
    elif reward == "compressibility":
        value_model = PromptConditionedTimeConvNet(
            latent_channels=4,
            prompt_dim=prompt_dim,
            time_dim=time_dim,
            prompt_channels=prompt_channels,
        )
    else:
        raise ValueError(f"Unknown reward in checkpoint: {reward}")

    value_model = value_model.to(device=device, dtype=torch.float32)
    value_model.load_state_dict(ckpt["value_model_state_dict"])
    value_model.eval()
    return eta_net, value_model, ckpt


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
