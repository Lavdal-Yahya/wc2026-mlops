"""Centralized seeding for reproducibility."""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)
