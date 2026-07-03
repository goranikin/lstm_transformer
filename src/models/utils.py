import math
from collections.abc import Callable
from typing import cast

import torch
from torch import nn


def init_uniform_(param: torch.Tensor, d: int) -> None:
    bound = 1.0 / math.sqrt(d)
    nn.init.uniform_(param, -bound, bound)


def set_decode_type_if_supported(model: torch.nn.Module, decode_type: str) -> None:
    set_decode_type = getattr(model, "set_decode_type", None)
    if callable(set_decode_type):
        cast(Callable[[str], None], set_decode_type)(decode_type)
