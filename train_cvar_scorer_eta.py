#!/usr/bin/env python

import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from PIL import Image
from torchvision.transforms import Normalize, Resize

from aesthetic_scorer import AestheticScorerDiff
from compressibility_scorer import jpeg_compressibility
from diffusers_patch.ddim_with_kl import ddim_step_KL
from sd_pipeline_cvar import Decoding_nonbatch_SDPipeline_CVaR
from aesthetic_scorer_cvar import MLPDiff_CVaR
from compressibility_scorer_cvar import SinusoidalTimeEtaConvNet


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def parse():
    parser = argparse.ArgumentParser(description="Train eta-conditioned CVaR MC scorer")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--reward", type=str, default="aesthetic", choices=["aesthetic", "compressibility"])
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output_dir", type=str, default="cvar_value_eta")
    parser.add_argument("--valuefunction", type=str, default="")

    parser.add_argument("--prompt_fn", type=str, default="eval_aesthetic_animals")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt_file", type=str, default="assets/simple_animals.txt")
    parser.add_argument("--negative_prompt", type=str, default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--states_per_trajectory", type=int, default=4)

    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--eta0", type=float, default=None)
    parser.add_argument("--eta_radius", type=float, default=None)
    parser.add_argument("--eta_base_samples", type=int, default=256)
    parser.add_argument("--eta_search_iters", type=int, default=80)
    parser.add_argument("--eta_lower", type=float, default=None)
    parser.add_argument("--eta_upper", type=float, default=None)
    parser.add_argument("--eta_verify_grid", type=int, default=101)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--time_encoding_dim", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompt_pool(args):
    """Return the fixed prompt set used by the shared value network."""
    if args.prompt != "":
        return [args.prompt]

    if args.prompt_file == "":
        raise ValueError("Use --prompt for one prompt or --prompt_file for shared training.")

    with open(args.prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Preserve file order while removing accidental duplicates.
    prompts = list(dict.fromkeys(prompts))
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompt_file}.")
    return prompts


def sample_prompts(prompt_pool, batch_size):
    """Sample prompts for one shared-network minibatch."""
    return random.choices(prompt_pool, k=batch_size)


def make_negative_prompts(args, batch_size):
    if args.negative_prompt == "":
        return None
    return [args.negative_prompt] * batch_size


def clip_preprocess(images):
    images = Resize(224, antialias=False)(images)
    images = Normalize(mean=CLIP_MEAN, std=CLIP_STD)(images)
    return images


@torch.no_grad()
def decode_latents(pipe, latents):
    scaling_factor = getattr(pipe.vae.config, "scaling_factor", 0.18215)
    images = pipe.vae.decode(latents.to(pipe.vae.dtype) / scaling_factor).sample
    return ((images / 2.0) + 0.5).clamp(0.0, 1.0)


@torch.no_grad()
def score_final_costs(args, pipe, aesthetic_oracle, latents):
    images = decode_latents(pipe, latents)

    if args.reward == "compressibility":
        rewards = jpeg_compressibility(images)
        rewards = torch.tensor(rewards, device=latents.device, dtype=torch.float32)
    else:
        images_clip = clip_preprocess(images.float())
        oracle_dtype = next(aesthetic_oracle.parameters()).dtype
        rewards, _ = aesthetic_oracle(images_clip.to(dtype=oracle_dtype))
        rewards = rewards.float()

    return -rewards.reshape(-1)


def choose_record_indices(num_steps, states_per_trajectory):
    if states_per_trajectory < 0 or states_per_trajectory >= num_steps - 1:
        return list(range(max(1, num_steps - 1)))
    indices = np.linspace(0, max(0, num_steps - 2), states_per_trajectory)
    return sorted(set(indices.round().astype(int).tolist()))


@torch.no_grad()
def rollout_base(args, pipe, aesthetic_oracle, prompts, record_states=True):
    device = pipe._execution_device
    batch_size = len(prompts)
    do_cfg = args.guidance_scale > 1.0
    negative_prompts = make_negative_prompts(args, batch_size)

    prompt_embeds = pipe._encode_prompt(
        prompts,
        device,
        1,
        do_cfg,
        negative_prompts,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    )

    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    latent_dtype = next(pipe.unet.parameters()).dtype
    latents = pipe.prepare_latents(
        batch_size,
        pipe.unet.config.in_channels,
        512,
        512,
        latent_dtype,
        device,
        None,
        None,
    )

    record_indices = set(
        choose_record_indices(len(timesteps), args.states_per_trajectory)
        if record_states
        else []
    )
    recorded_states = []

    for step_index, t in enumerate(timesteps):
        if step_index in record_indices:
            recorded_states.append((int(t.item()), latents.detach().clone()))

        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        noise_pred_raw = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
        ).sample

        if do_cfg:
            noise_uncond, noise_text = noise_pred_raw.chunk(2)
            noise_pred = noise_uncond + args.guidance_scale * (noise_text - noise_uncond)
        else:
            noise_pred = noise_pred_raw

        latents, _ = ddim_step_KL(
            pipe.scheduler,
            noise_pred,
            noise_pred,
            t,
            latents,
            eta=args.ddim_eta,
        )

    final_costs = score_final_costs(args, pipe, aesthetic_oracle, latents)
    return recorded_states, final_costs


def logmeanexp(values):
    values = np.asarray(values, dtype=np.float64)
    vmax = float(np.max(values))
    return vmax + math.log(float(np.mean(np.exp(values - vmax))))


def eta_objective(eta, costs, alpha, beta):
    costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    log_weights = -np.maximum(costs - float(eta), 0.0) / (alpha * (1.0 - beta))
    return float(eta - alpha * logmeanexp(log_weights))


def ternary_search_eta(costs, alpha, beta, lower=None, upper=None, num_iters=80):
    costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    if lower is None:
        lower = float(np.min(costs))
    if upper is None:
        upper = float(np.max(costs))
    if lower > upper:
        lower, upper = upper, lower

    left = float(lower)
    right = float(upper)
    for _ in range(num_iters):
        m1 = left + (right - left) / 3.0
        m2 = right - (right - left) / 3.0
        f1 = eta_objective(m1, costs, alpha, beta)
        f2 = eta_objective(m2, costs, alpha, beta)
        if f1 <= f2:
            right = m2
        else:
            left = m1

    eta0 = 0.5 * (left + right)
    return eta0, eta_objective(eta0, costs, alpha, beta)


@torch.no_grad()
def estimate_eta0_for_prompt(args, pipe, aesthetic_oracle, prompt):
    """Roll out one fixed prompt and estimate its local eta center."""
    costs = []
    while len(costs) < args.eta_base_samples:
        batch_size = min(args.batch_size, args.eta_base_samples - len(costs))
        prompts = [prompt] * batch_size
        _, batch_costs = rollout_base(
            args,
            pipe,
            aesthetic_oracle,
            prompts,
            record_states=False,
        )
        costs.extend(batch_costs.detach().cpu().tolist())
        print(
            "eta samples for ",
            repr(prompt),
            ": ",
            len(costs),
            "/",
            args.eta_base_samples,
        )

    costs = np.asarray(costs, dtype=np.float64)
    eta0, objective = ternary_search_eta(
        costs,
        args.alpha,
        args.beta,
        lower=args.eta_lower,
        upper=args.eta_upper,
        num_iters=args.eta_search_iters,
    )

    if args.eta_verify_grid > 1:
        lower = float(np.min(costs)) if args.eta_lower is None else float(args.eta_lower)
        upper = float(np.max(costs)) if args.eta_upper is None else float(args.eta_upper)
        grid = np.linspace(lower, upper, args.eta_verify_grid)
        grid_obj = np.asarray(
            [eta_objective(x, costs, args.alpha, args.beta) for x in grid]
        )
        grid_idx = int(np.argmin(grid_obj))
        if grid_obj[grid_idx] + 1e-5 < objective:
            print(
                "warning: coarse grid found a smaller objective than ternary search. "
                "The empirical eta objective may not be unimodal on this sample."
            )
            print("ternary eta/objective: ", eta0, objective)
            print("grid eta/objective: ", float(grid[grid_idx]), float(grid_obj[grid_idx]))

    info = {
        "prompt": prompt,
        "eta0": float(eta0),
        "objective": float(objective),
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "cost_min": float(np.min(costs)),
        "cost_max": float(np.max(costs)),
        "num_samples": int(costs.size),
    }
    return float(eta0), costs, info


@torch.no_grad()
def estimate_eta_centers(args, pipe, aesthetic_oracle, prompt_pool):
    """Estimate one eta center from base rollouts for every prompt."""
    eta_centers = {}
    base_costs = {}
    eta_info = {}

    for prompt in prompt_pool:
        eta0, costs, info = estimate_eta0_for_prompt(
            args,
            pipe,
            aesthetic_oracle,
            prompt,
        )
        eta_centers[prompt] = float(eta0)
        base_costs[prompt] = costs
        eta_info[prompt] = info
        print("eta_0 for ", repr(prompt), ": ", eta0)

    return eta_centers, base_costs, eta_info


def sample_eta(args, eta_centers, prompts, device):
    """Sample eta locally around the center belonging to each prompt."""
    centers = torch.tensor(
        [eta_centers[prompt] for prompt in prompts],
        dtype=torch.float32,
        device=device,
    )
    offsets = 2.0 * torch.rand(len(prompts), device=device) - 1.0
    return centers + args.eta_radius * offsets


def make_target(costs, eta):
    """Unscaled eta-dependent value target: (cost - eta)^+."""
    return F.relu(costs.float() - eta.float())


@torch.no_grad()
def latent_to_aesthetic_embed(pipe, aesthetic_oracle, latents):
    images = decode_latents(pipe, latents)
    images = clip_preprocess(images.float())
    clip_dtype = next(aesthetic_oracle.clip.parameters()).dtype
    embed = aesthetic_oracle.clip.get_image_features(pixel_values=images.to(dtype=clip_dtype))
    embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
    return embed.float()


def save_checkpoint(
    args,
    model,
    eta_normalization_center,
    eta_normalization_radius,
    eta_centers,
    step,
    path,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        # These two values are consumed by the scorer to normalize absolute eta.
        "eta0": float(eta_normalization_center),
        "eta_radius": float(eta_normalization_radius),
        "eta_normalization_center": float(eta_normalization_center),
        "eta_normalization_radius": float(eta_normalization_radius),
        # Prompt-specific centers define the locally trained eta neighborhoods.
        "eta_centers": {key: float(value) for key, value in eta_centers.items()},
        "training_eta_radius": float(args.eta_radius),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "reward": args.reward,
        "time_encoding_dim": int(args.time_encoding_dim),
        "target": "positive_part_cost",
        "target_formula": "relu(cost - eta)",
        "step": int(step),
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def main():
    args = parse()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    prompt_pool = load_prompt_pool(args)
    print("shared training prompts: ", prompt_pool)

    pipe = Decoding_nonbatch_SDPipeline_CVaR.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
    )
    pipe.to(args.device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.unet.eval()

    aesthetic_oracle = None
    if args.reward == "aesthetic":
        aesthetic_oracle = AestheticScorerDiff(dtype=torch.float32).to(args.device)
        aesthetic_oracle.requires_grad_(False)
        aesthetic_oracle.eval()

    if args.eta_radius is None:
        if args.reward == "aesthetic":
            args.eta_radius = 0.5
        else:
            args.eta_radius = 10.0

    if args.eta0 is None:
        eta_centers, base_costs, eta_info = estimate_eta_centers(
            args,
            pipe,
            aesthetic_oracle,
            prompt_pool,
        )
    else:
        # A command-line eta0 intentionally forces the same center for all prompts.
        eta_centers = {prompt: float(args.eta0) for prompt in prompt_pool}
        base_costs = None
        eta_info = {
            prompt: {
                "prompt": prompt,
                "eta0": float(args.eta0),
                "source": "command_line",
            }
            for prompt in prompt_pool
        }

    # The network receives absolute eta. Normalize it with one range covering
    # every prompt-specific local interval [eta0(prompt)-r, eta0(prompt)+r].
    center_values = np.asarray(list(eta_centers.values()), dtype=np.float64)
    eta_normalization_center = float(np.mean(center_values))
    eta_normalization_radius = float(
        np.max(np.abs(center_values - eta_normalization_center)) + args.eta_radius
    )

    if args.time_encoding_dim is None:
        # Match the existing repo: aesthetic uses a 768-D time encoding,
        # compressibility uses a 64-D spatial time encoding.
        args.time_encoding_dim = 768 if args.reward == "aesthetic" else 64

    print("prompt eta centers: ", eta_centers)
    print("local training eta radius: ", args.eta_radius)
    print("eta normalization center: ", eta_normalization_center)
    print("eta normalization radius: ", eta_normalization_radius)

    with open(os.path.join(args.output_dir, "eta0.json"), "w") as f:
        json.dump(
            {
                "eta_centers": eta_centers,
                "training_eta_radius": float(args.eta_radius),
                "eta_normalization_center": eta_normalization_center,
                "eta_normalization_radius": eta_normalization_radius,
                "per_prompt": eta_info,
            },
            f,
            indent=2,
        )
    if base_costs is not None:
        for prompt_index, prompt in enumerate(prompt_pool):
            np.save(
                os.path.join(args.output_dir, f"base_costs_{prompt_index:03d}.npy"),
                base_costs[prompt],
            )

    if args.reward == "aesthetic":
        model = MLPDiff_CVaR(
            eta0=eta_normalization_center,
            eta_radius=eta_normalization_radius,
            time_encoding_dim=args.time_encoding_dim,
        ).to(args.device)
    else:
        model = SinusoidalTimeEtaConvNet(
            num_channels=4,
            num_classes=1,
            time_encoding_dim=args.time_encoding_dim,
            eta0=eta_normalization_center,
            eta_radius=eta_normalization_radius,
            dtype=torch.float32,
        ).to(args.device)

    if args.valuefunction != "":
        checkpoint = torch.load(args.valuefunction, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Value function loaded: ", args.valuefunction)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model.train()
    for step in range(1, args.num_train_steps + 1):
        prompts = sample_prompts(prompt_pool, args.batch_size)
        states, final_costs = rollout_base(
            args,
            pipe,
            aesthetic_oracle,
            prompts,
            record_states=True,
        )

        if len(states) == 0:
            raise RuntimeError("No noisy states were recorded for value training.")

        # One eta per trajectory/sample, shared over all recorded t for that
        # trajectory.  This mirrors an eta-conditioned value V(x_t, t, eta).
        eta = sample_eta(args, eta_centers, prompts, final_costs.device)
        # Learn E[(c(X_0) - eta)^+ | X_t] under the base-model rollout.
        # beta is used to determine eta0, but it does not scale this target.
        target = make_target(final_costs, eta)

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=final_costs.device, dtype=torch.float32)
        for timestep, latents in states:
            batch_size = latents.shape[0]
            timesteps = torch.full(
                (batch_size,),
                timestep,
                dtype=torch.long,
                device=latents.device,
            )

            if args.reward == "aesthetic":
                with torch.no_grad():
                    embed = latent_to_aesthetic_embed(pipe, aesthetic_oracle, latents)
                prediction = model(embed, timesteps, eta).squeeze(1)
            else:
                prediction = model(latents.float(), timesteps, eta).squeeze(1)

            total_loss = total_loss + F.mse_loss(
                prediction.float(), target.float(), reduction="mean"
            )

        loss = total_loss / float(len(states))
        loss.backward()
        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            print(
                "step: ",
                step,
                " loss: ",
                float(loss.detach().cpu()),
                " cost_mean: ",
                float(final_costs.mean().item()),
                " eta_mean: ",
                float(eta.mean().item()),
            )

        if step % args.save_every == 0:
            path = os.path.join(args.output_dir, f"cvar_{args.reward}_eta_step_{step}.pth")
            save_checkpoint(
                args,
                model,
                eta_normalization_center,
                eta_normalization_radius,
                eta_centers,
                step,
                path,
            )
            print("saved: ", path)

    final_path = os.path.join(args.output_dir, f"cvar_{args.reward}_eta_final.pth")
    save_checkpoint(
        args,
        model,
        eta_normalization_center,
        eta_normalization_radius,
        eta_centers,
        args.num_train_steps,
        final_path,
    )
    print("saved: ", final_path)


if __name__ == "__main__":
    main()
