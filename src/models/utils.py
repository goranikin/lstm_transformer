import math

import torch
from torch import nn


def init_uniform_(param: torch.Tensor, d: int) -> None:
    bound = 1.0 / math.sqrt(d)
    nn.init.uniform_(param, -bound, bound)
