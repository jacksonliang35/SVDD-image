import math

import torch
import torch.nn as nn


class ResidualBlock_CVaR(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class SinusoidalTimeEtaConvNet(nn.Module):
    """Latent-space CVaR value network with time, eta, and prompt inputs."""

    def __init__(
        self,
        num_channels=4,
        num_classes=1,
        time_encoding_dim=64,
        eta0=0.0,
        eta_radius=1.0,
        dtype=torch.float32,
        prompt_conditioned=False,
        text_embedding_dim=768,
        prompt_encoding_dim=64,
    ):
        super().__init__()
        self.dtype = dtype
        self.time_encoding_dim = time_encoding_dim
        self.prompt_conditioned = bool(prompt_conditioned)
        self.text_embedding_dim = int(text_embedding_dim)
        self.prompt_encoding_dim = int(prompt_encoding_dim)
        self.register_buffer("eta0", torch.tensor(float(eta0)), persistent=True)
        self.register_buffer("eta_radius", torch.tensor(float(eta_radius)), persistent=True)

        if self.prompt_conditioned:
            self.prompt_projection = nn.Sequential(
                nn.Linear(self.text_embedding_dim, self.prompt_encoding_dim),
                nn.SiLU(),
            )
        else:
            self.prompt_projection = None

        layer2_input_channels = 64 + time_encoding_dim + 1
        if self.prompt_conditioned:
            layer2_input_channels += self.prompt_encoding_dim

        self.layer1 = ResidualBlock_CVaR(num_channels, 64, stride=1)
        self.layer2 = ResidualBlock_CVaR(layer2_input_channels, 128, stride=2)
        self.layer3 = ResidualBlock_CVaR(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def sinusoidal_encoding(self, timesteps, height, width):
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
        encoding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=1)
        return encoding[:, :, None, None].repeat(1, 1, height, width)

    def eta_encoding(self, eta, batch_size, height, width, device):
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
        eta = (eta - self.eta0) / radius
        return eta[:, None, None, None].repeat(1, 1, height, width)

    def encode_prompt(self, prompt_embedding, batch_size, height, width, device):
        if not self.prompt_conditioned:
            return None
        if prompt_embedding is None:
            raise ValueError(
                "This compressibility CVaR model is prompt-conditioned; "
                "provide prompt_embedding."
            )

        prompt_embedding = prompt_embedding.to(device=device, dtype=torch.float32)
        expected_shape = (batch_size, self.text_embedding_dim)
        if prompt_embedding.ndim != 2 or prompt_embedding.shape != expected_shape:
            raise ValueError(
                f"Expected prompt_embedding shape {expected_shape}, got "
                f"{tuple(prompt_embedding.shape)}."
            )

        prompt_embedding = self.prompt_projection(prompt_embedding)
        return prompt_embedding[:, :, None, None].expand(-1, -1, height, width)

    def forward(self, x, timesteps, eta, prompt_embedding=None):
        out = self.layer1(x.to(self.dtype))
        timestep_embed = self.sinusoidal_encoding(timesteps, out.size(2), out.size(3))
        eta_embed = self.eta_encoding(
            eta, out.shape[0], out.size(2), out.size(3), out.device
        )
        inputs = [out, timestep_embed.to(out.dtype), eta_embed.to(out.dtype)]
        prompt_embed = self.encode_prompt(
            prompt_embedding,
            out.shape[0],
            out.size(2),
            out.size(3),
            out.device,
        )
        if prompt_embed is not None:
            inputs.append(prompt_embed.to(out.dtype))
        combined_input = torch.cat(inputs, dim=1)
        out = self.layer2(combined_input)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class CompressibilityScorerDiff_CVaR(torch.nn.Module):
    """
    MC scorer for CVaR compressibility guidance.

    Returns E[(c(x_0) - eta)^+ | x_t, prompt], without the 1 / (1 - beta)
    scaling, and not a reward.
    """

    is_eta_conditioned_cvar = True
    output_is_cvar_cost = True
    output_is_unscaled_positive_part = True

    def __init__(self, dtype=torch.float32, pathtomodel=None):
        super().__init__()
        self.dtype = dtype
        self.model = None
        self.eta0 = None
        self.eta_radius = None
        self.alpha = None
        self.beta = None
        self.eta_centers = {}
        self.training_eta_radius = None
        self.target = None
        self.prompt_conditioned = False
        self.text_embedding_dim = 768
        self.prompt_encoding_dim = 64
        if pathtomodel is not None:
            self.set_valuefunction(pathtomodel)
        self.eval()

    def set_valuefunction(self, pathtomodel):
        checkpoint = torch.load(pathtomodel, map_location="cpu", weights_only=False)
        self.eta0 = float(checkpoint["eta0"])
        self.eta_radius = float(checkpoint["eta_radius"])
        self.alpha = float(checkpoint.get("alpha", 10.0))
        self.beta = float(checkpoint["beta"])
        self.eta_centers = {
            str(key): float(value)
            for key, value in checkpoint.get("eta_centers", {}).items()
        }
        self.training_eta_radius = checkpoint.get("training_eta_radius")
        if self.training_eta_radius is not None:
            self.training_eta_radius = float(self.training_eta_radius)
        self.target = checkpoint.get("target")
        if self.target != "positive_part_cost":
            raise ValueError(
                f"Unsupported checkpoint target {self.target!r}; expected "
                "'positive_part_cost'. Retrain with the unscaled target before "
                "using this sampler."
            )
        time_encoding_dim = int(checkpoint.get("time_encoding_dim", 64))
        self.prompt_conditioned = bool(checkpoint.get("prompt_conditioned", False))
        self.text_embedding_dim = int(checkpoint.get("text_embedding_dim", 768))
        self.prompt_encoding_dim = int(checkpoint.get("prompt_encoding_dim", 64))

        self.model = SinusoidalTimeEtaConvNet(
            num_channels=4,
            num_classes=1,
            time_encoding_dim=time_encoding_dim,
            eta0=self.eta0,
            eta_radius=self.eta_radius,
            dtype=self.dtype,
            prompt_conditioned=self.prompt_conditioned,
            text_embedding_dim=self.text_embedding_dim,
            prompt_encoding_dim=self.prompt_encoding_dim,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print("CVaR value function loaded: ", pathtomodel)
        print("eta_0: ", self.eta0, " eta_radius: ", self.eta_radius)
        return self

    def __call__(self, images, timesteps, eta, prompt_embedding=None):
        if self.model is None:
            raise RuntimeError("Call set_valuefunction(...) before using the CVaR scorer.")
        predictions = self.model(
            images,
            timesteps,
            eta,
            prompt_embedding=prompt_embedding,
        ).squeeze(1)
        return predictions, images
