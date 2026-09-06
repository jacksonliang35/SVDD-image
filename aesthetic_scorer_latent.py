"""Small aesthetic value networks on noisy SD latents. No VAE or CLIP image encoder."""
import math

import torch
from torch import nn
from torch.nn import functional as F


FORMAT = "aesthetic_latent_v1"


def batch_scalar(value, batch, device, name):
    value = torch.as_tensor(value, device=device, dtype=torch.float32).reshape(-1)
    if value.numel() == 1:
        return value.expand(batch)
    if value.numel() != batch:
        raise ValueError(f"{name} must be scalar or have {batch} entries")
    return value


class ConditionedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.affine = nn.Linear(condition_dim, 2 * out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=2)

    def forward(self, x, condition):
        h = F.silu(self.norm1(self.conv1(x)))
        scale, shift = self.affine(condition).chunk(2, dim=-1)
        h = self.norm2(self.conv2(h))
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return F.silu(h + self.skip(x))


class AestheticLatentValueNet(nn.Module):
    def __init__(self, eta_conditioned=False, prompt_conditioned=True,
                 text_embedding_dim=768, time_encoding_dim=64, width=32,
                 condition_dim=128, eta0=0.0, eta_radius=1.0):
        super().__init__()
        if width < 8 or width % 8 or time_encoding_dim < 2 or time_encoding_dim % 2:
            raise ValueError("width must be a positive multiple of 8; time dimension must be positive and even")
        if not math.isfinite(eta_radius) or eta_radius <= 0 or not math.isfinite(eta0):
            raise ValueError("eta normalization requires a finite center and positive radius")
        self.config = dict(eta_conditioned=eta_conditioned, prompt_conditioned=prompt_conditioned,
                           text_embedding_dim=text_embedding_dim, time_encoding_dim=time_encoding_dim,
                           width=width, condition_dim=condition_dim, eta0=eta0, eta_radius=eta_radius)
        self.eta_conditioned = eta_conditioned
        self.prompt_conditioned = prompt_conditioned
        self.text_embedding_dim = text_embedding_dim
        self.time_encoding_dim = time_encoding_dim
        self.register_buffer("eta_center", torch.tensor(float(eta0)))
        self.register_buffer("eta_scale", torch.tensor(float(eta_radius)))
        self.time_mlp = nn.Sequential(nn.Linear(time_encoding_dim, condition_dim), nn.SiLU(),
                                      nn.Linear(condition_dim, condition_dim))
        self.prompt_mlp = (nn.Linear(text_embedding_dim, condition_dim)
                           if prompt_conditioned else None)
        self.eta_mlp = (nn.Sequential(nn.Linear(1, condition_dim), nn.SiLU(),
                                     nn.Linear(condition_dim, condition_dim))
                        if eta_conditioned else None)
        self.stem = nn.Conv2d(4, width, 3, padding=1)
        self.blocks = nn.ModuleList([
            ConditionedBlock(width, width, condition_dim),
            ConditionedBlock(width, width * 2, condition_dim),
            ConditionedBlock(width * 2, width * 4, condition_dim),
        ])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                                  nn.Linear(width * 4 * 16, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, latents, timesteps, eta=None, prompt_embedding=None):
        if latents.ndim != 4 or latents.shape[1] != 4:
            raise ValueError("Expected noisy SD latents of shape [B, 4, H, W]")
        parameter = self.stem.weight
        x = latents.to(device=parameter.device, dtype=parameter.dtype)
        batch = x.shape[0]
        t = batch_scalar(timesteps, batch, x.device, "timesteps")
        frequency = torch.exp(-math.log(10000) * torch.arange(
            self.time_encoding_dim // 2, device=x.device, dtype=torch.float32
        ) / max(self.time_encoding_dim // 2 - 1, 1))
        angles = t[:, None] * frequency[None, :]
        condition = self.time_mlp(torch.cat((angles.sin(), angles.cos()), -1).to(x.dtype))
        if self.prompt_conditioned:
            if prompt_embedding is None or tuple(prompt_embedding.shape) != (batch, self.text_embedding_dim):
                raise ValueError(f"Provide prompt_embedding with shape {(batch, self.text_embedding_dim)}")
            condition = condition + self.prompt_mlp(prompt_embedding.to(device=x.device, dtype=x.dtype))
        if self.eta_conditioned:
            if eta is None:
                raise ValueError("The CVaR latent model requires eta")
            eta = batch_scalar(eta, batch, x.device, "eta")
            scaled_eta = (eta - self.eta_center) / self.eta_scale
            condition = condition + self.eta_mlp(scaled_eta[:, None].to(x.dtype))
        h = self.stem(x)
        for block in self.blocks:
            h = block(h, condition)
        prediction = self.head(h).squeeze(-1)
        return F.softplus(prediction) if self.eta_conditioned else prediction


class AestheticScorerLatent(nn.Module):
    """Predict E[r(X0) | z_t, t, prompt] in raw aesthetic-reward units."""
    input_is_latent = True
    is_eta_conditioned_cvar = False
    output_is_cvar_cost = False
    expected_target = "expected_reward"

    def __init__(self, dtype=torch.float32, pathtomodel=None):
        super().__init__()
        self.dtype = dtype
        self.model = None
        if pathtomodel is not None:
            self.set_valuefunction(pathtomodel)

    def set_valuefunction(self, pathtomodel):
        checkpoint = torch.load(pathtomodel, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != FORMAT or checkpoint.get("reward") != "aesthetic":
            raise ValueError("Expected a new latent aesthetic checkpoint; use the obsolete scorer for CLIP checkpoints")
        if checkpoint.get("target") != self.expected_target:
            raise ValueError(f"Expected target {self.expected_target!r}; got {checkpoint.get('target')!r}")
        config = checkpoint["model_config"]
        if bool(config["eta_conditioned"]) != self.is_eta_conditioned_cvar:
            raise ValueError("Checkpoint architecture and target disagree about eta conditioning")
        device = next(self.model.parameters()).device if self.model is not None else torch.device("cpu")
        self.model = AestheticLatentValueNet(**config)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(device=device, dtype=self.dtype)
        self.prompt_conditioned = config["prompt_conditioned"]
        self.eta_centers = checkpoint.get("eta_centers", {})
        self.training_eta_radius = checkpoint.get("training_eta_radius")
        self.eta0 = config["eta0"]
        self.eta_radius = config["eta_radius"]
        self.alpha = checkpoint.get("alpha")
        self.beta = checkpoint.get("beta")
        self.target = checkpoint["target"]
        self.training_args = checkpoint.get("args", {})
        self.requires_grad_(False)
        self.eval()
        return self

    def forward(self, latents, timesteps, eta=None, prompt_embedding=None):
        if self.model is None:
            raise RuntimeError("Load a trained latent checkpoint with pathtomodel or set_valuefunction")
        return self.model(latents, timesteps, eta=eta, prompt_embedding=prompt_embedding), None


class AestheticScorerLatent_CVaR(AestheticScorerLatent):
    """Predict E[(cost(X0)-eta)^+ | z_t,t,prompt], with cost=-reward; no beta scaling."""
    is_eta_conditioned_cvar = True
    output_is_cvar_cost = True
    output_is_unscaled_positive_part = True
    expected_target = "positive_part_cost"
