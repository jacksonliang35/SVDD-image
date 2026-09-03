import math

import torch
import torch.nn as nn


class ResidualBlock_CVaR(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class SinusoidalTimeEtaConvNet(nn.Module):
    """Latent-space CVaR value network with time and eta as inputs."""

    def __init__(
        self,
        num_channels=4,
        num_classes=1,
        time_encoding_dim=64,
        eta0=0.0,
        eta_radius=1.0,
        dtype=torch.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.time_encoding_dim = time_encoding_dim
        self.register_buffer("eta0", torch.tensor(float(eta0)), persistent=True)
        self.register_buffer("eta_radius", torch.tensor(float(eta_radius)), persistent=True)

        self.layer1 = ResidualBlock_CVaR(num_channels, 64, stride=1)
        self.layer2 = ResidualBlock_CVaR(64 + time_encoding_dim + 1, 128, stride=2)
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

    def forward(self, x, timesteps, eta):
        out = self.layer1(x.to(self.dtype))
        timestep_embed = self.sinusoidal_encoding(timesteps, out.size(2), out.size(3))
        eta_embed = self.eta_encoding(
            eta, out.shape[0], out.size(2), out.size(3), out.device
        )
        combined_input = torch.cat(
            [out, timestep_embed.to(out.dtype), eta_embed.to(out.dtype)], dim=1
        )
        out = self.layer2(combined_input)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class CompressibilityScorerDiff_CVaR(torch.nn.Module):
    """
    MC scorer for CVaR compressibility guidance.

    Returns a positive, cost-like conditional tail value and not a reward.
    """

    is_eta_conditioned_cvar = True
    output_is_cvar_cost = True

    def __init__(self, dtype=torch.float32, pathtomodel=None):
        super().__init__()
        self.dtype = dtype
        self.model = None
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
        time_encoding_dim = int(checkpoint.get("time_encoding_dim", 64))

        self.model = SinusoidalTimeEtaConvNet(
            num_channels=4,
            num_classes=1,
            time_encoding_dim=time_encoding_dim,
            eta0=self.eta0,
            eta_radius=self.eta_radius,
            dtype=self.dtype,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print("CVaR value function loaded: ", pathtomodel)
        print("eta_0: ", self.eta0, " eta_radius: ", self.eta_radius)
        return self

    def __call__(self, images, timesteps, eta):
        if self.model is None:
            raise RuntimeError("Call set_valuefunction(...) before using the CVaR scorer.")
        predictions = self.model(images, timesteps, eta).squeeze(1)
        return predictions, images
