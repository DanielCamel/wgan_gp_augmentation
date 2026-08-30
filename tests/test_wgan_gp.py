import numpy as np
import torch

from wgan_gp_augmentation.config import WganConfig
from wgan_gp_augmentation.models.wgan_gp import Critic, Generator, gradient_penalty
from wgan_gp_augmentation.training.wgan_gp import fit_wgan, sample_generator


def test_model_shapes_and_linear_outputs() -> None:
    generator = Generator(5, 3, (8,))
    critic = Critic(3, (8,))
    fake = generator(torch.zeros(4, 5))
    assert fake.shape == (4, 3)
    assert critic(fake).shape == (4,)


def test_gradient_penalty_is_scalar_and_differentiable() -> None:
    critic = Critic(3, (8,))
    real = torch.randn(6, 3)
    fake = torch.randn(6, 3)
    penalty, norms = gradient_penalty(critic, real, fake, 10.0)
    penalty.backward()
    assert penalty.ndim == 0
    assert norms.shape == (6,)
    weight_gradients = [
        parameter.grad for name, parameter in critic.named_parameters() if name.endswith("weight")
    ]
    assert all(gradient is not None for gradient in weight_gradients)


def test_training_and_generation_are_reproducible() -> None:
    real = np.random.default_rng(7).normal(size=(24, 4)).astype(np.float32)
    config = WganConfig((8,), 8, 3, 2, 1, 0.0001, 0.0, 0.9, 10.0, 30, 1)
    first = fit_wgan(real, config, 11)
    second = fit_wgan(real, config, 11)
    first_generated = sample_generator(first.generator, 5, 3, first.feature_mean, first.feature_scale, 12)
    second_generated = sample_generator(second.generator, 5, 3, second.feature_mean, second.feature_scale, 12)
    assert np.array_equal(first_generated, second_generated)
