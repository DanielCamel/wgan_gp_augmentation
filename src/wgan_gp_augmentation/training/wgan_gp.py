from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from ..config import WganConfig
from ..models.wgan_gp import Critic, Generator, gradient_penalty
from ..reproducibility import set_torch_seed


@dataclass(frozen=True)
class WganFit:
    generator: Generator
    critic: Critic
    history: list[dict[str, float | int]]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    elapsed_seconds: float
    device: str


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_wgan(real_samples: np.ndarray, config: WganConfig, seed: int) -> WganFit:
    """Train a WGAN-GP on real training observations from one class."""
    if real_samples.ndim != 2 or len(real_samples) < 2:
        raise ValueError("WGAN training requires a two-dimensional array with at least two rows")
    set_torch_seed(seed)
    device = _device()
    mean = real_samples.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = real_samples.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    normalized = ((real_samples - mean) / scale).astype(np.float32)
    real_tensor = torch.from_numpy(normalized).to(device)
    dimension = real_samples.shape[1]
    generator = Generator(config.latent_dim, dimension, config.hidden_layers).to(device)
    critic = Critic(dimension, config.hidden_layers).to(device)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2)
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2)
    )
    torch_rng = torch.Generator(device=device).manual_seed(seed)
    batch_size = min(config.batch_size, len(real_tensor))
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, config.train_steps + 1):
        critic_loss_value = 0.0
        wasserstein_value = 0.0
        penalty_value = 0.0
        norm_value = 0.0
        for _ in range(config.critic_steps):
            indices = torch.randint(len(real_tensor), (batch_size,), generator=torch_rng, device=device)
            real_batch = real_tensor[indices]
            noise = torch.rand(
                (batch_size, config.latent_dim), generator=torch_rng, device=device
            ) * 2.0 - 1.0
            fake_batch = generator(noise).detach()
            real_score = critic(real_batch).mean()
            fake_score = critic(fake_batch).mean()
            penalty, gradient_norms = gradient_penalty(
                critic, real_batch, fake_batch, config.gradient_penalty
            )
            critic_loss = fake_score - real_score + penalty
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_optimizer.step()
            critic_loss_value = float(critic_loss.detach().cpu())
            wasserstein_value = float((real_score - fake_score).detach().cpu())
            penalty_value = float(penalty.detach().cpu())
            norm_value = float(gradient_norms.mean().detach().cpu())
        noise = torch.rand(
            (batch_size, config.latent_dim), generator=torch_rng, device=device
        ) * 2.0 - 1.0
        generator_loss = -critic(generator(noise)).mean()
        generator_optimizer.zero_grad(set_to_none=True)
        generator_loss.backward()
        generator_optimizer.step()
        if step == 1 or step == config.train_steps or step % config.log_interval == 0:
            history.append(
                {
                    "step": step,
                    "critic_loss": critic_loss_value,
                    "generator_loss": float(generator_loss.detach().cpu()),
                    "wasserstein_estimate": wasserstein_value,
                    "gradient_penalty": penalty_value,
                    "mean_gradient_norm": norm_value,
                }
            )
    return WganFit(
        generator=generator.cpu(),
        critic=critic.cpu(),
        history=history,
        feature_mean=mean,
        feature_scale=scale,
        elapsed_seconds=time.perf_counter() - started,
        device=str(device),
    )


def sample_generator(
    generator: Generator,
    count: int,
    latent_dim: int,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    seed: int,
    batch_size: int = 4096,
) -> np.ndarray:
    """Generate deterministic samples and return them in global PCA space."""
    if count < 0:
        raise ValueError("Sample count cannot be negative")
    if count == 0:
        return np.empty((0, len(feature_mean)), dtype=np.float32)
    set_torch_seed(seed)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    generator = generator.cpu().eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, count, batch_size):
            size = min(batch_size, count - start)
            noise = torch.rand((size, latent_dim), generator=rng) * 2.0 - 1.0
            batches.append(generator(noise).numpy())
    normalized = np.concatenate(batches, axis=0)
    return (normalized * feature_scale + feature_mean).astype(np.float32)
