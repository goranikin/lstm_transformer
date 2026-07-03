import random
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def move_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    payload = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield payload
    finally:
        payload["elapsed"] = time.perf_counter() - start
