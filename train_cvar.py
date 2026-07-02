#!/usr/bin/env python
"""
train_cvar.py

Trainer for a CVaR-tail value function for SVDD-image.

This file is meant to live at the root of the SVDD-image repo, next to
sd_pipeline_cvar.py, aesthetic_scorer.py, and compressibility_scorer.py.

It follows the Monte Carlo value-function idea used by SVDD-MC:
    1. Roll out the pretrained diffusion model.
    2. Store intermediate noisy states x_t.
    3. Score the final image x_0 with the true reward oracle.
    4. Regress a time-dependent value model from x_t to a terminal target.

For the CVaR formulation, the default target is reward-like:

    y_eta(x_0) = - relu(c(x_0) - eta) / (1 - beta),

where c(x_0) = -reward(x_0). Larger is better, and y_eta <= 0.

Eta is trained jointly using the empirical pretrained objective:

    J(eta) = eta - alpha * log E_pre[ exp(-relu(c(x_0)-eta)
                                          / (alpha * (1-beta))) ].

Important inference note:
    Your current sd_pipeline_cvar.py calculate_mc_cost() interprets the MC
    scorer output as a reward and then calculate_weighted_value() applies the
    CVaR hinge again. If you train this file with --target_type tail_reward,
    the model already outputs the CVaR tail reward. To use it without a double
    hinge in the current pipeline, run MC with cvar_lambda=0 and cvar_eta=0, or
    add a separate pipeline branch that directly uses this tail value as the
    log-potential. See the README-style note printed at the end of training.
"""

import argparse
import datetime as _datetime
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler

from diffusers_patch.ddim_with_kl import ddim_step_KL
from sd_pipeline_cvar import Decoding_nonbatch_SDPipeline_CVaR

from aesthetic_scorer import AestheticScorerDiff, SinusoidalTimeMLP
from compressibility_scorer import (
    SinusoidalTimeConvNet,
    jpeg_compressibility,
)


ArrayLike = Union[np.ndarray, Sequence[float], torch.Tensor]


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


@dataclass
class TrainConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    local_files_only: bool = False
    device: str = "cuda:0"
    dtype: str = "bfloat16" # float32, float16, bfloat16

    reward: str = "compressibility"  # aesthetic or compressibility
    prompt: Optional[str] = None
    prompt_fn: str = "eval_aesthetic_animals"
    negative_prompt: Optional[str] = None

    output_dir: str = "cvar_value_runs"
    run_name: Optional[str] = None
    seed: int = 42

    num_train_steps: int = 1000
    batch_size: int = 4
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    ddim_eta: float = 1.0
    height: int = 512
    width: int = 512

    alpha: float = 10.0
    beta: float = 0.8
    eta_init: float = 120.0
    eta_lr: float = 1.0e-3
    value_lr: float = 1.0e-4
    weight_decay: float = 0.0
    eta_bounds: Optional[Tuple[float, float]] = None
    grad_clip: Optional[float] = 1.0

    # tail_reward is the recommended default for the new CVaR value.
    # Other modes are included for experiments.
    target_type: str = "tail_reward"  # tail_reward, tail_cost, log_weight, exp_weight

    # Store this many random intermediate states per trajectory.
    # Use -1 to use every diffusion state.
    states_per_trajectory: int = 4
    include_t0_state: bool = False

    save_every: int = 100
    log_every: int = 10
    num_workers: int = 0


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a CVaR SVDD-MC value function.")

    parser.add_argument("--model_id", type=str, default=TrainConfig.model_id)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--dtype", type=str, default=TrainConfig.dtype, choices=["float32", "float16", "bfloat16"])

    parser.add_argument("--reward", type=str, default=TrainConfig.reward, choices=["aesthetic", "compressibility"])
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_fn", type=str, default=TrainConfig.prompt_fn)
    parser.add_argument("--negative_prompt", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)

    parser.add_argument("--num_train_steps", type=int, default=TrainConfig.num_train_steps)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--num_inference_steps", type=int, default=TrainConfig.num_inference_steps)
    parser.add_argument("--guidance_scale", type=float, default=TrainConfig.guidance_scale)
    parser.add_argument("--ddim_eta", type=float, default=TrainConfig.ddim_eta)
    parser.add_argument("--height", type=int, default=TrainConfig.height)
    parser.add_argument("--width", type=int, default=TrainConfig.width)

    parser.add_argument("--alpha", type=float, default=TrainConfig.alpha)
    parser.add_argument("--beta", type=float, default=TrainConfig.beta)
    parser.add_argument("--eta_init", type=float, default=TrainConfig.eta_init)
    parser.add_argument("--eta_lr", type=float, default=TrainConfig.eta_lr)
    parser.add_argument("--value_lr", type=float, default=TrainConfig.value_lr)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--eta_min", type=float, default=None)
    parser.add_argument("--eta_max", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=TrainConfig.grad_clip)

    parser.add_argument(
        "--target_type",
        type=str,
        default=TrainConfig.target_type,
        choices=["tail_reward", "tail_cost", "log_weight", "exp_weight"],
    )
    parser.add_argument("--states_per_trajectory", type=int, default=TrainConfig.states_per_trajectory)
    parser.add_argument("--include_t0_state", action="store_true")

    parser.add_argument("--save_every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--log_every", type=int, default=TrainConfig.log_every)

    args = parser.parse_args()

    eta_bounds = None
    if args.eta_min is not None or args.eta_max is not None:
        if args.eta_min is None or args.eta_max is None:
            raise ValueError("Pass both --eta_min and --eta_max, or neither.")
        lo, hi = float(args.eta_min), float(args.eta_max)
        if lo > hi:
            lo, hi = hi, lo
        eta_bounds = (lo, hi)

    return TrainConfig(
        model_id=args.model_id,
        local_files_only=bool(args.local_files_only),
        device=args.device,
        dtype=args.dtype,
        reward=args.reward,
        prompt=args.prompt,
        prompt_fn=args.prompt_fn,
        negative_prompt=args.negative_prompt,
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        ddim_eta=args.ddim_eta,
        height=args.height,
        width=args.width,
        alpha=args.alpha,
        beta=args.beta,
        eta_init=args.eta_init,
        eta_lr=args.eta_lr,
        value_lr=args.value_lr,
        weight_decay=args.weight_decay,
        eta_bounds=eta_bounds,
        grad_clip=args.grad_clip,
        target_type=args.target_type,
        states_per_trajectory=args.states_per_trajectory,
        include_t0_state=bool(args.include_t0_state),
        save_every=args.save_every,
        log_every=args.log_every,
    )


def get_torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
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
        cfg.run_name = f"cvar_{cfg.reward}_{cfg.target_type}_{stamp}"
    run_dir = os.path.join(cfg.output_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def make_prompt_batch(cfg: TrainConfig, batch_size: int) -> List[str]:
    if cfg.prompt is not None:
        return [cfg.prompt] * batch_size

    try:
        import prompts as prompts_file
    except Exception as exc:
        raise RuntimeError(
            "Could not import prompts.py. Pass --prompt 'your prompt' or make sure "
            "prompts.py is importable from the repo root."
        ) from exc

    prompt_fn = getattr(prompts_file, cfg.prompt_fn)
    prompts: List[str] = []
    for _ in range(batch_size):
        out = prompt_fn()
        if isinstance(out, (tuple, list)):
            prompts.append(str(out[0]))
        else:
            prompts.append(str(out))
    return prompts


def make_negative_prompt_batch(cfg: TrainConfig, batch_size: int) -> Optional[List[str]]:
    if cfg.negative_prompt is None:
        return None
    return [cfg.negative_prompt] * batch_size


def clip_preprocess_tensor(images_01: torch.Tensor) -> torch.Tensor:
    """Convert [B,3,H,W] in [0,1] to CLIP-normalized [B,3,224,224]."""
    x = F.interpolate(images_01, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor(CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def decode_latents_to_tensor(pipe: Decoding_nonbatch_SDPipeline_CVaR, latents: torch.Tensor) -> torch.Tensor:
    """Decode SD latents to a torch image tensor [B,3,H,W] in [0,1]."""
    image = pipe.vae.decode(latents.to(pipe.vae.dtype) / 0.18215).sample
    return ((image / 2.0) + 0.5).clamp(0.0, 1.0)


@torch.inference_mode()
def score_final_costs(
    cfg: TrainConfig,
    pipe: Decoding_nonbatch_SDPipeline_CVaR,
    reward_scorer: Any,
    final_latents: torch.Tensor,
) -> torch.Tensor:
    """Return cost = -reward for final latents."""
    images_01 = decode_latents_to_tensor(pipe, final_latents)

    if cfg.reward == "compressibility":
        rewards_np = jpeg_compressibility(images_01)
        rewards = torch.as_tensor(rewards_np, device=final_latents.device, dtype=torch.float32)

    elif cfg.reward == "aesthetic":
        images_clip = clip_preprocess_tensor(images_01).to(dtype=next(reward_scorer.parameters()).dtype)
        rewards, _ = reward_scorer(images_clip)
        rewards = rewards.detach().float()

    else:
        raise ValueError(f"Unknown reward: {cfg.reward}")

    costs = -rewards.reshape(-1).float()
    return costs


def choose_record_indices(num_steps: int, cfg: TrainConfig) -> List[int]:
    """Choose diffusion step indices whose x_t states become MC training data."""
    last_allowed = num_steps if cfg.include_t0_state else max(1, num_steps - 1)
    all_indices = list(range(last_allowed))
    if cfg.states_per_trajectory < 0 or cfg.states_per_trajectory >= len(all_indices):
        return all_indices
    return sorted(random.sample(all_indices, int(cfg.states_per_trajectory)))


@torch.inference_mode()
def rollout_pretrained_batch(
    cfg: TrainConfig,
    pipe: Decoding_nonbatch_SDPipeline_CVaR,
    reward_scorer: Any,
    prompts: List[str],
    negative_prompts: Optional[List[str]],
    generator: Optional[torch.Generator] = None,
) -> Tuple[List[Tuple[int, torch.Tensor]], torch.Tensor]:
    """
    Roll out the pretrained diffusion model and return:
        recorded_states: list of (timestep_int, latents_cpu), each latents_cpu is [B,4,64,64]
        final_costs: torch tensor [B] on device
    """
    device = pipe._execution_device
    dtype = next(pipe.unet.parameters()).dtype
    batch_size = len(prompts)
    do_cfg = cfg.guidance_scale > 1.0

    prompt_embeds = pipe._encode_prompt(
        prompts,
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
        batch_size,
        pipe.unet.config.in_channels,
        cfg.height,
        cfg.width,
        dtype,
        device,
        generator,
        None,
    )

    recorded_states: List[Tuple[int, torch.Tensor]] = []

    for step_index, t in enumerate(timesteps):
        if step_index in record_indices:
            recorded_states.append((int(t.detach().cpu().item()), latents.detach().cpu()))

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

        latents, _kl_terms = ddim_step_KL(
            pipe.scheduler,
            noise_pred,
            noise_pred,
            t,
            latents,
            eta=cfg.ddim_eta,
        )

        del latent_model_input, noise_pred_raw, noise_pred, _kl_terms

    final_costs = score_final_costs(cfg, pipe, reward_scorer, latents)
    del latents
    return recorded_states, final_costs


def cvar_eta_objective(costs: torch.Tensor, eta: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    J(eta) = eta - alpha * log mean exp(-relu(c-eta)/(alpha*(1-beta))).
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if not (0.0 <= beta < 1.0):
        raise ValueError("beta must be in [0, 1).")
    denom = float(alpha) * (1.0 - float(beta))
    log_terms = -F.relu(costs.float() - eta.float()) / denom
    return eta.float() - float(alpha) * (torch.logsumexp(log_terms, dim=0) - math.log(log_terms.numel()))


@torch.no_grad()
def eta_grad_info(costs: torch.Tensor, eta: torch.Tensor, alpha: float, beta: float) -> Dict[str, float]:
    """
    Diagnostic derivative for the pretrained eta objective.

    For log_terms_i = -relu(c_i-eta)/(alpha*(1-beta)), the derivative is:
        1 - P_tilted(c > eta)/(1-beta),
    where P_tilted uses weights proportional to exp(log_terms_i).
    """
    denom = float(alpha) * (1.0 - float(beta))
    costs_f = costs.detach().float()
    eta_f = eta.detach().float()
    log_terms = -F.relu(costs_f - eta_f) / denom
    probs = torch.softmax(log_terms, dim=0)
    tail_prob_tilted = torch.sum(probs * (costs_f > eta_f).float())
    grad = 1.0 - tail_prob_tilted / (1.0 - float(beta))
    return {
        "eta_grad": float(grad.detach().cpu()),
        "tail_prob_tilted": float(tail_prob_tilted.detach().cpu()),
        "target_tail_prob": float(1.0 - float(beta)),
    }


def make_targets(
    costs: torch.Tensor,
    eta_value: torch.Tensor,
    alpha: float,
    beta: float,
    target_type: str,
) -> torch.Tensor:
    """
    Build regression targets from terminal costs.

    tail_reward:  -relu(c-eta)/(1-beta), recommended for the new CVaR value.
    tail_cost:     relu(c-eta)/(1-beta).
    log_weight:   -relu(c-eta)/(alpha*(1-beta)).
    exp_weight:    exp(log_weight).
    """
    tail_cost = F.relu(costs.float() - eta_value.detach().float()) / (1.0 - float(beta))
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


@torch.inference_mode()
def latent_state_to_clip_embed(
    pipe: Decoding_nonbatch_SDPipeline_CVaR,
    reward_scorer: AestheticScorerDiff,
    latents: torch.Tensor,
) -> torch.Tensor:
    """Feature map used by AestheticScorerDiff_Time during MC inference."""
    images_01 = decode_latents_to_tensor(pipe, latents)
    images_clip = clip_preprocess_tensor(images_01).to(dtype=next(reward_scorer.parameters()).dtype)
    embed = reward_scorer.clip.get_image_features(pixel_values=images_clip)
    embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
    return embed.detach().float()


def build_models(cfg: TrainConfig, device: torch.device, dtype: torch.dtype) -> Tuple[Any, torch.nn.Module]:
    """Build final reward scorer and trainable CVaR value model."""
    if cfg.reward == "aesthetic":
        reward_scorer = AestheticScorerDiff(dtype=dtype).to(device)
        reward_scorer.requires_grad_(False)
        reward_scorer.eval()

        value_model = SinusoidalTimeMLP().to(device)

    elif cfg.reward == "compressibility":
        reward_scorer = None
        # Current MC compressibility inference passes latents [B,4,64,64]
        # to scorer(latents, timesteps). Train the same input convention.
        value_model = SinusoidalTimeConvNet(num_channels=4, num_classes=1, dtype=dtype).to(device)

    else:
        raise ValueError(f"Unknown reward: {cfg.reward}")

    value_model.train()
    return reward_scorer, value_model


def save_outputs(
    run_dir: str,
    cfg: TrainConfig,
    value_model: torch.nn.Module,
    eta_param: torch.Tensor,
    value_optimizer: torch.optim.Optimizer,
    eta_optimizer: torch.optim.Optimizer,
    step: int,
    metrics: Dict[str, Any],
) -> None:
    os.makedirs(run_dir, exist_ok=True)

    # This file is directly loadable by your existing aesthetic
    # AestheticScorerDiff_Time.set_valuefunction(path). For compressibility,
    # you can assign scorer.model = torch.load(path).
    model_path = os.path.join(run_dir, "cvar_value_model_latest.pth")
    torch.save(value_model, model_path)

    ckpt_path = os.path.join(run_dir, "checkpoint_latest.pt")
    torch.save(
        {
            "step": int(step),
            "config": asdict(cfg),
            "eta": float(eta_param.detach().cpu()),
            "model_state_dict": value_model.state_dict(),
            "value_optimizer_state_dict": value_optimizer.state_dict(),
            "eta_optimizer_state_dict": eta_optimizer.state_dict(),
            "metrics": metrics,
        },
        ckpt_path,
    )

    with open(os.path.join(run_dir, "cvar_eta.json"), "w", encoding="utf-8") as f:
        json.dump({"eta": float(eta_param.detach().cpu()), "step": int(step), "metrics": metrics}, f, indent=2)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def train(cfg: TrainConfig) -> None:
    if cfg.alpha <= 0:
        raise ValueError("alpha must be positive.")
    if not (0.0 <= cfg.beta < 1.0):
        raise ValueError("beta must be in [0, 1).")
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    set_seed(cfg.seed)
    run_dir = make_run_dir(cfg)
    device = torch.device(cfg.device)
    dtype = get_torch_dtype(cfg.dtype)

    print(f"Run directory: {run_dir}")
    print(f"Loading pipeline: {cfg.model_id}")

    pipe = Decoding_nonbatch_SDPipeline_CVaR.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
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

    reward_scorer, value_model = build_models(cfg, device, dtype)

    value_optimizer = torch.optim.AdamW(
        value_model.parameters(),
        lr=cfg.value_lr,
        weight_decay=cfg.weight_decay,
    )

    eta_param = torch.nn.Parameter(torch.tensor(float(cfg.eta_init), device=device, dtype=torch.float32))
    eta_optimizer = torch.optim.Adam([eta_param], lr=cfg.eta_lr)

    if cfg.eta_bounds is not None:
        with torch.no_grad():
            eta_param.clamp_(cfg.eta_bounds[0], cfg.eta_bounds[1])

    log_path = os.path.join(run_dir, "train_log.jsonl")
    generator = torch.Generator(device=device)
    if cfg.seed >= 0:
        generator.manual_seed(cfg.seed + 12345)

    try:
        from tqdm.auto import trange
        step_iter = trange(1, cfg.num_train_steps + 1, desc="Training CVaR value")
    except Exception:
        step_iter = range(1, cfg.num_train_steps + 1)

    for step in step_iter:
        prompts = make_prompt_batch(cfg, cfg.batch_size)
        negative_prompts = make_negative_prompt_batch(cfg, cfg.batch_size)

        recorded_states, final_costs = rollout_pretrained_batch(
            cfg=cfg,
            pipe=pipe,
            reward_scorer=reward_scorer,
            prompts=prompts,
            negative_prompts=negative_prompts,
            generator=generator,
        )

        final_costs = final_costs.to(device=device, dtype=torch.float32)

        # 1. Eta update from the pretrained terminal costs.
        eta_optimizer.zero_grad(set_to_none=True)
        eta_loss = cvar_eta_objective(final_costs, eta_param, cfg.alpha, cfg.beta)
        eta_loss.backward()
        eta_optimizer.step()

        if cfg.eta_bounds is not None:
            with torch.no_grad():
                eta_param.clamp_(cfg.eta_bounds[0], cfg.eta_bounds[1])

        # 2. Value update. Detach eta so the regression loss cannot move eta
        # just to make the targets easier; eta is defined by eta_loss above.
        targets = make_targets(
            costs=final_costs,
            eta_value=eta_param.detach(),
            alpha=cfg.alpha,
            beta=cfg.beta,
            target_type=cfg.target_type,
        )

        value_optimizer.zero_grad(set_to_none=True)
        value_loss_total = torch.zeros((), device=device, dtype=torch.float32)
        n_value_terms = 0

        for timestep_int, latents_cpu in recorded_states:
            latents_t = latents_cpu.to(device=device, dtype=dtype, non_blocking=True)
            timesteps_t = torch.full(
                (latents_t.shape[0],),
                int(timestep_int),
                device=device,
                dtype=torch.long,
            )

            if cfg.reward == "aesthetic":
                with torch.inference_mode():
                    embed_t = latent_state_to_clip_embed(pipe, reward_scorer, latents_t)
                pred = value_model(embed_t, timesteps_t).squeeze(1).float()
            else:
                pred = value_model(latents_t, timesteps_t).squeeze(1).float()

            loss = F.mse_loss(pred, targets)
            value_loss_total = value_loss_total + loss
            n_value_terms += 1

            del latents_t, timesteps_t, pred, loss

        if n_value_terms == 0:
            raise RuntimeError("No recorded states were collected; check states_per_trajectory.")

        value_loss = value_loss_total / float(n_value_terms)
        value_loss.backward()

        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), float(cfg.grad_clip))

        value_optimizer.step()

        with torch.no_grad():
            info = eta_grad_info(final_costs, eta_param, cfg.alpha, cfg.beta)
            metrics = {
                "step": int(step),
                "eta": float(eta_param.detach().cpu()),
                "eta_loss": float(eta_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "cost_mean": float(final_costs.mean().detach().cpu()),
                "cost_std": float(final_costs.std(unbiased=False).detach().cpu()),
                "cost_min": float(final_costs.min().detach().cpu()),
                "cost_max": float(final_costs.max().detach().cpu()),
                "target_mean": float(targets.mean().detach().cpu()),
                "target_std": float(targets.std(unbiased=False).detach().cpu()),
                "num_recorded_states": int(n_value_terms),
            }
            metrics.update(info)

        if step % cfg.log_every == 0 or step == 1:
            msg = (
                f"step={step} eta={metrics['eta']:.4f} "
                f"eta_loss={metrics['eta_loss']:.4f} "
                f"value_loss={metrics['value_loss']:.4f} "
                f"cost_mean={metrics['cost_mean']:.4f} "
                f"tail_p_tilted={metrics['tail_prob_tilted']:.3f}"
            )
            print(msg)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")

        if step % cfg.save_every == 0 or step == cfg.num_train_steps:
            save_outputs(
                run_dir=run_dir,
                cfg=cfg,
                value_model=value_model,
                eta_param=eta_param,
                value_optimizer=value_optimizer,
                eta_optimizer=eta_optimizer,
                step=step,
                metrics=metrics,
            )

        del recorded_states, final_costs, targets, value_loss_total, value_loss, eta_loss

    print("Training finished.")
    print(f"Saved model: {os.path.join(run_dir, 'cvar_value_model_latest.pth')}")
    print(f"Saved eta:   {os.path.join(run_dir, 'cvar_eta.json')}")
    print("")
    print("Inference note:")
    if cfg.target_type == "tail_reward":
        print("  This model outputs tail_reward = -relu(c-eta)/(1-beta).")
        print("  In your current pipeline, avoid applying the CVaR hinge a second time.")
        print("  Easiest current-code usage: set cvar_lambda=0 and cvar_eta=0 when this")
        print("  scorer is used as the MC scorer, so log_weight = tail_reward / alpha.")
    elif cfg.target_type == "log_weight":
        print("  This model outputs log_weight directly. Use a pipeline branch that treats")
        print("  scorer output as log_weight, not as reward/cost.")
    elif cfg.target_type == "exp_weight":
        print("  This model outputs exp(log_weight). This is usually less stable than")
        print("  tail_reward or log_weight because probabilities can have high dynamic range.")


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
