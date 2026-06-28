from diffusers import DDIMScheduler
import torch
import numpy as np
import random
import PIL
from typing import Callable, List, Optional, Union, Dict, Any
import os
from tqdm import tqdm
import datetime
from types import SimpleNamespace

from aesthetic_scorer import AestheticScorerDiff_Time, MLPDiff
from compressibility_scorer import CompressibilityScorerDiff, jpeg_compressibility, CompressibilityScorer_modified
from aesthetic_scorer import AestheticScorerDiff

from sd_pipeline_cvar import Decoding_nonbatch_SDPipeline_CVaR


args = {
    "device": "cuda:1",
    "reward": "compressibility",
    "num_images": 100,
    "bs": 10,
    "val_bs": 10,
    "seed": 19274,
    "duplicate_size": 20,
    "alpha": 10,
    "cvar_beta": 0.8,
    "cvar_lambda": 0.95,
    "cvar_eta": 130.,
    "variant": "PM"
}
args = SimpleNamespace(**args)

if args.reward == 'compressibility':
    if args.variant == 'PM':
        scorer = CompressibilityScorer_modified(dtype=torch.float32)#.to(device)
    elif args.variant == 'MC':
        scorer = CompressibilityScorerDiff(dtype=torch.float32).to(args.device)
elif args.reward == 'aesthetic':
    if args.variant == 'PM':
        scorer = AestheticScorerDiff(dtype=torch.float32).to(args.device)
    elif args.variant == 'MC':
        scorer = AestheticScorerDiff_Time(dtype=torch.float32).to(args.device)
        # if args.valuefunction != "":
        #     scorer.set_valuefunction(args.valuefunction)
        #     scorer = scorer.to(args.device)
scorer.requires_grad_(False)
scorer.eval()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
shape = (args.num_images//args.bs, args.bs, 4, 64, 64)
init_latents = torch.randn(shape, device=args.device)


sd_model = Decoding_nonbatch_SDPipeline_CVaR.from_pretrained("runwayml/stable-diffusion-v1-5", local_files_only=False)
sd_model.to(args.device)

# switch to DDIM scheduler
sd_model.scheduler = DDIMScheduler.from_config(sd_model.scheduler.config)
sd_model.scheduler.set_timesteps(50, device=args.device)

sd_model.vae.requires_grad_(False)
sd_model.text_encoder.requires_grad_(False)
sd_model.unet.requires_grad_(False)

sd_model.vae.eval()
sd_model.text_encoder.eval()
sd_model.unet.eval()

sd_model.setup_scorer(scorer)
sd_model.set_variant(args.variant)
sd_model.set_reward(args.reward)
sd_model.set_parameters(args.bs, args.duplicate_size, args.alpha)

sd_model.set_cvar_beta(args.cvar_beta)
sd_model.set_cvar_lambda(args.cvar_lambda)
sd_model.set_cvar_eta(args.cvar_eta)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
initial_memory = torch.cuda.memory_allocated()

prompt = "a crowded street market at night with hundreds of signs, lanterns, people, food stalls, fabrics, reflections, and details."

image = []
eval_prompt_list = []
KL_list = []

for i in tqdm(range(args.num_images // args.bs), desc="Generating Images"):
    init_i = init_latents[i]
    eval_prompts = [prompt] * args.bs
    eval_prompt_list.extend(eval_prompts)

    # image_, kl_loss = sd_model(eval_prompts, num_images_per_prompt=1, eta=1.0, latents=init_i) # List of PIL.Image objects
    image_, kl_loss = sd_model.sample_max(eval_prompts, num_images_per_prompt=1, eta=1.0, latents=init_i) # List of PIL.Image objects
    image.extend(image_)
    KL_list.append(kl_loss)

# KL_entropy = torch.mean(torch.stack(KL_list))
end_event.record()
torch.cuda.synchronize() # Wait for the events to complete
gpu_time = start_event.elapsed_time(end_event)/1000 # Time in seconds
max_memory = torch.cuda.max_memory_allocated()
max_memory_used = (max_memory - initial_memory) / (1024 ** 2)

print("GPUTimeInS:", gpu_time)
print("MaxMemoryInMb:", max_memory_used)

from dataset import AVACompressibilityDataset, AVACLIPDataset

if args.reward == 'compressibility':
    gt_dataset= AVACompressibilityDataset(image)
elif args.reward == 'aesthetic':
    from importlib import resources
    ASSETS_PATH = resources.files("assets")
    eval_model = MLPDiff().to(args.device)
    eval_model.requires_grad_(False)
    eval_model.eval()
    s = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"), map_location=device, weights_only=True)
    eval_model.load_state_dict(s)
    gt_dataset= AVACLIPDataset(image)

gt_dataloader = torch.utils.data.DataLoader(gt_dataset, batch_size=args.val_bs, shuffle=False)

with torch.no_grad():
    eval_rewards = []

    for inputs in gt_dataloader:
        inputs = inputs.to(args.device)

        if args.reward == 'compressibility':
            jpeg_compressibility_scores = jpeg_compressibility(inputs)
            scores = torch.tensor(jpeg_compressibility_scores, dtype=inputs.dtype, device=inputs.device)

        elif args.reward == 'aesthetic':
            scores = eval_model(inputs)
            scores = scores.squeeze(1)

        eval_rewards.extend(scores.tolist())

    eval_rewards = torch.tensor(eval_rewards)

    print(f"eval_{args.reward}_rewards:", eval_rewards)
    print(f"eval_{args.reward}_rewards_mean:", torch.mean(eval_rewards))
    print(f"eval_{args.reward}_rewards_std:", torch.std(eval_rewards))

np.save('./compressibility_pm_m20_street_cvar_max.npy', np.array(eval_rewards))
