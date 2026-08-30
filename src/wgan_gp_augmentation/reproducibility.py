from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed random number generators available in the baseline environment."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_torch_seed(seed: int, deterministic: bool = True) -> None:
    """Seed PyTorch and request deterministic algorithms when installed."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
