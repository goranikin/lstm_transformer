from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class EncoderOutput:
    node_embeddings: torch.Tensor
    graph_embedding: torch.Tensor


@dataclass
class ProblemState:
    batch: dict[str, Any]
    selected_mask: torch.Tensor
    done: torch.Tensor
    prev_action: torch.Tensor
    first_action: torch.Tensor
    actions: list[torch.Tensor] = field(default_factory=list)
    step_count: int = 0
    aux: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.done.size(0))

    @property
    def device(self) -> torch.device:
        return self.done.device

    def stacked_actions(self, pad_value: int = -1) -> torch.Tensor:
        if not self.actions:
            return torch.full(
                (self.batch_size, 0),
                pad_value,
                dtype=torch.long,
                device=self.device,
            )
        return torch.stack(self.actions, dim=1)


@dataclass
class ProblemDecodeState:
    problem: Any
    batch: dict[str, Any]
    state: ProblemState
    target_actions: torch.Tensor | None = None


@dataclass(frozen=True)
class SupervisedTarget:
    actions: torch.Tensor | None = None
    selected_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class SolutionOutput:
    actions: torch.Tensor
    log_probs: torch.Tensor | None
    selected_mask: torch.Tensor | None
    objective: torch.Tensor
    feasible: torch.Tensor
    logits: torch.Tensor | None = None
    reward: torch.Tensor | None = None


def stack_or_empty(actions: list[torch.Tensor], batch_size: int, device: torch.device):
    if actions:
        return torch.stack(actions, dim=1)
    return torch.empty(batch_size, 0, dtype=torch.long, device=device)
