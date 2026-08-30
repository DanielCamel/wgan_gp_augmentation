from __future__ import annotations

import torch
from torch import nn


def _mlp(input_dim: int, hidden_layers: tuple[int, ...], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_layers:
        layers.append(nn.Linear(previous, width))
        layers.append(nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.2))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    network = nn.Sequential(*layers)
    for module in network.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return network


class Generator(nn.Module):
    """MLP generator for one class in normalized PCA space."""

    def __init__(self, latent_dim: int, output_dim: int, hidden_layers: tuple[int, ...]) -> None:
        super().__init__()
        self.network = _mlp(latent_dim, hidden_layers, output_dim, "relu")

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


class Critic(nn.Module):
    """Unbounded scalar critic for one class."""

    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...]) -> None:
        super().__init__()
        self.network = _mlp(input_dim, hidden_layers, 1, "leaky_relu")

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return self.network(samples).reshape(-1)


def gradient_penalty(
    critic: Critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return WGAN-GP penalty and per-sample interpolation gradient norms."""
    epsilon_shape = (real.shape[0],) + (1,) * (real.ndim - 1)
    epsilon = torch.rand(epsilon_shape, device=real.device, dtype=real.dtype)
    interpolated = epsilon * real + (1.0 - epsilon) * fake
    interpolated.requires_grad_(True)
    scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    norms = gradients.flatten(start_dim=1).norm(2, dim=1)
    return coefficient * ((norms - 1.0) ** 2).mean(), norms

