import math

import torch
import torch.nn as nn
from transformers import CLIPModel


class MLPDiff_CVaR(nn.Module):
    """Aesthetic CVaR value model with explicit eta conditioning."""

    def __init__(self, eta0, eta_radius, time_encoding_dim=768):
        super().__init__()
        self.time_encoding_dim = time_encoding_dim
        self.register_buffer("eta0", torch.tensor(float(eta0)), persistent=True)
        self.register_buffer("eta_radius", torch.tensor(float(eta_radius)), persistent=True)

        self.layers = nn.Sequential(
            nn.Linear(768 + time_encoding_dim + 1, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def sinusoidal_encoding(self, timesteps):
        timesteps = timesteps.float() / 1000.0
        frequencies = torch.exp(
            torch.arange(
                0,
                self.time_encoding_dim,
                2,
                dtype=torch.float32,
                device=timesteps.device,
            )
            * -(math.log(10000.0) / self.time_encoding_dim)
        )
        arguments = timesteps[:, None] * frequencies[None, :]
        return torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=1)

    def normalize_eta(self, eta, batch_size, device):
        if not isinstance(eta, torch.Tensor):
            eta = torch.tensor(eta, dtype=torch.float32, device=device)
        eta = eta.to(device=device, dtype=torch.float32)
        if eta.ndim == 0:
            eta = eta.repeat(batch_size)
        elif eta.numel() == 1:
            eta = eta.reshape(1).repeat(batch_size)
        else:
            eta = eta.reshape(batch_size)
        radius = torch.clamp(self.eta_radius.abs(), min=1e-8)
        return ((eta - self.eta0) / radius).unsqueeze(1)

    def forward(self, embed, timesteps, eta):
        timestep_embed = self.sinusoidal_encoding(timesteps).to(embed.dtype)
        eta_embed = self.normalize_eta(eta, embed.shape[0], embed.device).to(embed.dtype)
        combined_input = torch.cat([embed, timestep_embed, eta_embed], dim=1)
        return self.layers(combined_input.float())


class AestheticScorerDiff_CVaR(torch.nn.Module):
    """
    MC scorer for CVaR aesthetic guidance.

    The returned prediction is a *cost-like* tail value
        E[(c(x_0) - eta)^+ / (1 - beta) | x_t]
    rather than a reward.  sd_pipeline_cvar.py should therefore consume this
    value directly and must not negate it or apply another CVaR hinge.
    """

    is_eta_conditioned_cvar = True
    output_is_cvar_cost = True

    def __init__(self, dtype=torch.float32, pathtomodel=None):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.dtype = dtype
        self.mlp = None
        self.eta0 = None
        self.eta_radius = None
        self.alpha = None
        self.beta = None
        if pathtomodel is not None:
            self.set_valuefunction(pathtomodel)
        self.eval()

    def set_valuefunction(self, pathtomodel):
        checkpoint = torch.load(pathtomodel, map_location="cpu", weights_only=False)
        self.eta0 = float(checkpoint["eta0"])
        self.eta_radius = float(checkpoint["eta_radius"])
        self.alpha = float(checkpoint.get("alpha", 10.0))
        self.beta = float(checkpoint["beta"])
        time_encoding_dim = int(checkpoint.get("time_encoding_dim", 768))

        self.mlp = MLPDiff_CVaR(
            eta0=self.eta0,
            eta_radius=self.eta_radius,
            time_encoding_dim=time_encoding_dim,
        )
        self.mlp.load_state_dict(checkpoint["model_state_dict"])
        self.mlp.eval()
        print("CVaR value function loaded: ", pathtomodel)
        print("eta_0: ", self.eta0, " eta_radius: ", self.eta_radius)
        return self

    def __call__(self, images, timesteps, eta):
        if self.mlp is None:
            raise RuntimeError("Call set_valuefunction(...) before using the CVaR scorer.")
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        predictions = self.mlp(embed, timesteps, eta).squeeze(1)
        return predictions, embed

    def generate_feats(self, images):
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return embed
