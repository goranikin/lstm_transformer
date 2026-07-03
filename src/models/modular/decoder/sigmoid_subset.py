import torch
from torch import nn

from src.models.modular.common import init_linear


class SigmoidSubsetHead(nn.Module):
    """Independent node/item membership head for subset prediction."""

    def __init__(self, d_h: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(d_h, 1)
        init_linear(self.classifier)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.classifier(h).squeeze(-1)
