from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedState:
    seed: int
    python_random_seeded: bool
    numpy_seeded: bool
    torch_seeded: bool
    torch_cuda_seeded: bool
    deterministic_algorithms: bool
    cudnn_deterministic: bool | None
    cudnn_benchmark: bool | None


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return seed


def seed_summary(seed: int) -> dict[str, int | bool]:
    validated = _validate_seed(seed)
    return {
        "python": validated,
        "numpy": validated,
        "torch": validated,
        "cuda": validated,
        "cuda_all": validated,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }


def fix_seed(seed: int) -> dict[str, int | bool]:
    validated = _validate_seed(seed)
    summary = seed_summary(validated)
    random.seed(validated)
    np.random.seed(validated)
    torch.manual_seed(validated)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(validated)
        torch.cuda.manual_seed_all(validated)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return summary


def set_global_seed(seed: int, *, deterministic_algorithms: bool = False) -> SeedState:
    validated = _validate_seed(seed)
    random.seed(validated)
    np.random.seed(validated)
    torch.manual_seed(validated)
    cuda_seeded = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(validated)
        cuda_seeded = True

    cudnn_deterministic: bool | None = None
    cudnn_benchmark: bool | None = None
    if deterministic_algorithms:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
            cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    elif hasattr(torch.backends, "cudnn"):
        cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
        cudnn_benchmark = bool(torch.backends.cudnn.benchmark)

    return SeedState(
        seed=validated,
        python_random_seeded=True,
        numpy_seeded=True,
        torch_seeded=True,
        torch_cuda_seeded=cuda_seeded,
        deterministic_algorithms=deterministic_algorithms,
        cudnn_deterministic=cudnn_deterministic,
        cudnn_benchmark=cudnn_benchmark,
    )


def make_python_rng(seed: int) -> random.Random:
    return random.Random(_validate_seed(seed))


def make_numpy_rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(_validate_seed(seed))


def seed_worker(worker_id: int, base_seed: int) -> int:
    if not isinstance(worker_id, int):
        raise ValueError("worker_id must be an integer")
    derived = _validate_seed(base_seed) + worker_id
    random.seed(derived)
    np.random.seed(derived)
    return derived
