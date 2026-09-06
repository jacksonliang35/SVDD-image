"""Shared training for raw-reward and positive-part aesthetic latent values.

Frozen base rollouts supply final-image oracle labels. Intermediate states are
never decoded or passed through CLIP. Checkpoints include optimizer and RNG.
"""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from aesthetic_scorer_latent import FORMAT, AestheticLatentValueNet
from aesthetic_latent_sampling import encode_prompt_features


def parse(target):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", default=f"aesthetic_latent/{target}")
    p.add_argument("--resume_from_checkpoint", default="")
    p.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--prompt_file", default="assets/simple_animals.txt")
    p.add_argument("--prompt", default="")
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--oracle_batch_size", type=int, default=8)
    p.add_argument("--num_train_steps", type=int, default=1000,
                   help="Final global step, including steps already completed on resume")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--states_per_trajectory", type=int, default=8)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--ddim_eta", type=float, default=1.0)
    p.add_argument("--base_dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--no_prompt_conditioning", action="store_true")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--time_encoding_dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--validation_every", type=int, default=100)
    p.add_argument("--validation_samples_per_prompt", type=int, default=32)
    p.add_argument("--validation_seed", type=int, default=2027)
    if target == "positive_part_cost":
        p.add_argument("--alpha", type=float, default=10.0)
        p.add_argument("--beta", type=float, default=0.8)
        p.add_argument("--eta_radius", type=float, default=0.5)
        p.add_argument("--eta_base_samples", type=int, default=256)
        p.add_argument("--eta_search_iters", type=int, default=100)
        p.add_argument("--eta_verify_grid", type=int, default=201)
    args = p.parse_args()
    for key in ("batch_size", "oracle_batch_size", "states_per_trajectory", "num_train_steps",
                "save_every", "log_every", "validation_every", "validation_samples_per_prompt"):
        if getattr(args, key) < 1:
            p.error(f"--{key} must be positive")
    if args.num_inference_steps < 2 or args.lr <= 0 or args.weight_decay < 0 or args.grad_clip < 0:
        p.error("Invalid diffusion-step count or optimizer setting")
    if target == "positive_part_cost" and not (args.alpha > 0 and 0 <= args.beta < 1
            and args.eta_radius > 0 and args.eta_base_samples > 0
            and args.eta_search_iters > 0 and args.eta_verify_grid >= 2):
        p.error("Invalid eta-calibration settings")
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same number of visible GPUs for RNG restoration")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def make_targets(costs, eta, target):
    if target == "expected_reward":
        return -costs.float()
    if target == "positive_part_cost":
        return F.relu(costs.float() - eta.float())
    raise ValueError(target)


@torch.no_grad()
def rollout(args, pipe, oracle, prompts, record=True):
    from torchvision.transforms import Normalize, Resize
    from diffusers_patch.ddim_with_kl import ddim_step_KL
    cfg = args.guidance_scale > 1
    negative_prompts = [args.negative_prompt] * len(prompts) if args.negative_prompt else None
    if hasattr(pipe, "encode_prompt"):
        positive, negative = pipe.encode_prompt(
            prompts, pipe._execution_device, 1, cfg, negative_prompt=negative_prompts,
        )
        embeddings = torch.cat((negative, positive)) if cfg else positive
    else:
        embeddings = pipe._encode_prompt(
            prompts, pipe._execution_device, 1, cfg, negative_prompt=negative_prompts,
        )
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=pipe._execution_device)
    timesteps = pipe.scheduler.timesteps
    latents = pipe.prepare_latents(len(prompts), 4, 512, 512, pipe.unet.dtype,
                                   pipe._execution_device, None, None)
    # Candidate values are queried at the NEXT timestep. Cover indices 1..T-1,
    # including the last noisy state, with a new uniform subset each rollout.
    count = min(args.states_per_trajectory, len(timesteps) - 1)
    indices = set(np.random.choice(np.arange(1, len(timesteps)), count, replace=False)) if record else set()
    states = []
    for i, t in enumerate(timesteps):
        if i in indices:
            states.append((int(t.item()), latents.detach().clone()))
        x = torch.cat((latents, latents)) if cfg else latents
        noise = pipe.unet(pipe.scheduler.scale_model_input(x, t), t,
                          encoder_hidden_states=embeddings).sample
        if cfg:
            uncond, cond = noise.chunk(2)
            noise = uncond + args.guidance_scale * (cond - uncond)
        latents, _ = ddim_step_KL(pipe.scheduler, noise, noise, t, latents, eta=args.ddim_eta)
    costs = []
    for chunk in latents.split(args.oracle_batch_size):
        images = pipe.vae.decode(chunk.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor).sample
        images = ((images.float() / 2) + .5).clamp(0, 1)
        images = Resize(224, antialias=False)(images)
        images = Normalize([.48145466, .4578275, .40821073], [.26862954, .26130258, .27577711])(images)
        rewards, _ = oracle(images.to(next(oracle.parameters()).dtype))
        costs.append(-rewards.float().reshape(-1))
    costs = torch.cat(costs)
    if not torch.isfinite(costs).all():
        raise ValueError("Non-finite terminal aesthetic rewards")
    return states, costs


def calibrate_eta(args, pipe, oracle, prompts):
    from sd_pipeline_cvar import solve_cvar_eta_time0_ternary_from_costs, solve_cvar_eta_time0_from_costs
    centers, metadata = {}, {}
    for index, prompt in enumerate(prompts):
        chunks = []
        for start in range(0, args.eta_base_samples, args.batch_size):
            batch = [prompt] * min(args.batch_size, args.eta_base_samples - start)
            _, costs = rollout(args, pipe, oracle, batch, record=False)
            chunks.append(costs.cpu().numpy())
            print(f"eta calibration {prompt}: {start + len(batch)}/{args.eta_base_samples}", flush=True)
        costs = np.concatenate(chunks)
        eta, info = solve_cvar_eta_time0_ternary_from_costs(costs, args.alpha, args.beta, args.eta_search_iters)
        grid_eta, grid_info = solve_cvar_eta_time0_from_costs(costs, args.alpha, args.beta, args.eta_verify_grid)
        # The empirical objective need not be unimodal. Retain the better result.
        if grid_info["objective"] < info["objective"]:
            eta, info = grid_eta, dict(grid_info, method="grid_verification")
        centers[prompt], metadata[prompt] = float(eta), info
        np.save(Path(args.output_dir) / f"base_costs_{index:03d}.npy", costs)
    (Path(args.output_dir) / "eta_calibration.json").write_text(json.dumps(
        dict(eta_centers=centers, training_eta_radius=args.eta_radius,
             args=vars(args), per_prompt=metadata), indent=2))
    return centers


def sample_eta(args, centers, prompts, device):
    if not centers:
        return None
    center = torch.tensor([centers[p] for p in prompts], device=device, dtype=torch.float32)
    return center + (2 * torch.rand(len(prompts), device=device) - 1) * args.eta_radius


def train_update(model, optimizer, states, costs, features, eta, target, grad_clip):
    if not states:
        raise ValueError("No training states")
    labels = make_targets(costs, eta, target)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.
    for t, latents in states:
        prediction = model(latents, t, eta=eta, prompt_embedding=features)
        loss = F.mse_loss(prediction.float(), labels)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        # Backprop one state at a time, avoiding retained graphs for all states.
        (loss / len(states)).backward()
        loss_sum += loss.detach().item()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return loss_sum / len(states)


def save_checkpoint(path, args, target, model, optimizer, prompts, centers, step, best_mse):
    state = dict(format=FORMAT, reward="aesthetic", target=target,
                 model_config=model.config, model_state_dict=model.state_dict(),
                 optimizer_state_dict=optimizer.state_dict(), rng_state=rng_state(),
                 prompt_pool=prompts, eta_centers=centers, step=step, args=vars(args),
                 training_eta_radius=getattr(args, "eta_radius", None),
                 alpha=getattr(args, "alpha", None), beta=getattr(args, "beta", None),
                 best_validation_mse=best_mse,
                 prompt_embedding_source="stable_diffusion_text_encoder_pooled_l2_normalized")
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def validate_resume(checkpoint, args, target, prompts):
    if checkpoint.get("format") != FORMAT or checkpoint.get("target") != target:
        raise ValueError("Resume requires a latent checkpoint with the same target; legacy checkpoints cannot resume here")
    if checkpoint["prompt_pool"] != prompts:
        raise ValueError("Resume prompt pool differs from checkpoint")
    keys = ["model_id", "negative_prompt", "num_inference_steps", "guidance_scale", "ddim_eta",
            "base_dtype", "no_prompt_conditioning", "width", "time_encoding_dim",
            "batch_size", "oracle_batch_size", "states_per_trajectory", "seed",
            "validation_seed", "validation_samples_per_prompt"]
    if target == "positive_part_cost":
        keys += ["alpha", "beta", "eta_radius"]
    for key in keys:
        if checkpoint["args"][key] != getattr(args, key):
            raise ValueError(f"Resume --{key} must match saved value {checkpoint['args'][key]!r}")
    if checkpoint["step"] >= args.num_train_steps:
        raise ValueError("--num_train_steps must exceed the checkpoint's completed step")


@torch.no_grad()
def validation_rows(model, validation, features, target):
    rows = []
    device = next(model.parameters()).device
    for prompt, states, costs, eta in validation:
        costs = costs.to(device)
        eta = eta.to(device) if eta is not None else None
        feature = features[prompt].expand(len(costs), -1) if features else None
        labels = make_targets(costs, eta, target)
        for t, latents in states:
            prediction = model(latents, t, eta=eta, prompt_embedding=feature).float()
            error = prediction - labels
            rows.append(dict(prompt=prompt, timestep=t, n=len(costs), mse=error.square().mean().item(),
                             bias=error.mean().item(), zero_mse=labels.square().mean().item(),
                             constant_mse=((labels-labels.mean())**2).mean().item()))
    return rows


def main(target):
    args = parse(target)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prompts = [args.prompt] if args.prompt else list(dict.fromkeys(
        s.strip() for s in Path(args.prompt_file).read_text().splitlines() if s.strip()))
    if not prompts:
        raise ValueError("Empty prompt pool")
    checkpoint = (torch.load(args.resume_from_checkpoint, map_location="cpu", weights_only=False)
                  if args.resume_from_checkpoint else None)
    if checkpoint is not None:
        validate_resume(checkpoint, args, target, prompts)
    elif (output / "latest.pth").exists():
        raise ValueError("Output already contains latest.pth; resume it or choose a new output_dir")
    seed_all(args.seed)
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from aesthetic_scorer import AestheticScorerDiff
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=getattr(torch, args.base_dtype),
                                                   local_files_only=args.local_files_only).to(args.device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    for module in (pipe.unet, pipe.vae, pipe.text_encoder):
        module.requires_grad_(False).eval()
    oracle = AestheticScorerDiff(dtype=torch.float32).to(args.device).requires_grad_(False).eval()
    features = {}
    if not args.no_prompt_conditioning:
        for prompt in prompts:
            features[prompt] = encode_prompt_features(pipe, [prompt])
    centers = checkpoint["eta_centers"] if checkpoint else (
        calibrate_eta(args, pipe, oracle, prompts) if target == "positive_part_cost" else {})
    eta0 = float(np.mean(list(centers.values()))) if centers else 0.
    radius = max(abs(x - eta0) for x in centers.values()) + args.eta_radius if centers else 1.
    config = dict(eta_conditioned=target == "positive_part_cost",
                  prompt_conditioned=not args.no_prompt_conditioning,
                  text_embedding_dim=pipe.text_encoder.config.hidden_size,
                  width=args.width, time_encoding_dim=args.time_encoding_dim, eta0=eta0, eta_radius=radius)
    if checkpoint:
        config = checkpoint["model_config"]
        if config["text_embedding_dim"] != pipe.text_encoder.config.hidden_size:
            raise ValueError("Resume text encoder dimension mismatch")
    model = AestheticLatentValueNet(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start, best_mse = 0, float("inf")
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group.update(lr=args.lr, weight_decay=args.weight_decay)
        start, best_mse = checkpoint["step"], checkpoint["best_validation_mse"]
    # Fixed held-out trajectories on CPU; generation cannot perturb training RNG.
    training_rng = checkpoint["rng_state"] if checkpoint else rng_state()
    seed_all(args.validation_seed)
    validation = []
    for prompt in prompts:
        for i in range(0, args.validation_samples_per_prompt, args.batch_size):
            batch = [prompt] * min(args.batch_size, args.validation_samples_per_prompt - i)
            states, costs = rollout(args, pipe, oracle, batch)
            eta = sample_eta(args, centers, batch, costs.device)
            validation.append((prompt, [(t, z.cpu()) for t,z in states], costs.cpu(),
                               eta.cpu() if eta is not None else None))
    restore_rng(training_rng)
    print(f"Training {target}; {sum(p.numel() for p in model.parameters()):,} parameters; prompts={prompts}", flush=True)
    for step in range(start + 1, args.num_train_steps + 1):
        model.train()
        batch = random.choices(prompts, k=args.batch_size)
        states, costs = rollout(args, pipe, oracle, batch)
        feature = torch.cat([features[p] for p in batch]) if features else None
        eta = sample_eta(args, centers, batch, costs.device)
        loss = train_update(model, optimizer, states, costs, feature, eta, target, args.grad_clip)
        if step == start + 1 or step % args.log_every == 0:
            print(f"step={step} loss={loss:.6f} reward_mean={-costs.mean().item():.4f}", flush=True)
        if step % args.validation_every == 0 or step == args.num_train_steps:
            model.eval()
            rows = validation_rows(model, validation, features, target)
            mse = sum(row['n'] * row['mse'] for row in rows) / sum(row['n'] for row in rows)
            with (output / "validation.jsonl").open("a") as f:
                f.write(json.dumps(dict(step=step, mse=mse, rows=rows)) + "\n")
            print(f"validation step={step} mse={mse:.6f}", flush=True)
            if mse < best_mse:
                best_mse = mse
                save_checkpoint(output / "best.pth", args, target, model, optimizer, prompts, centers, step, best_mse)
        if step % args.save_every == 0 or step == args.num_train_steps:
            for name in (f"step_{step}.pth", "latest.pth"):
                save_checkpoint(output / name, args, target, model, optimizer, prompts, centers, step, best_mse)
