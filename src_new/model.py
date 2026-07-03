from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src_new.constants import DecodeType, DecoderKind, EncoderKind, ProblemName
from src_new.core import ProblemDecodeState, SolutionOutput, SupervisedTarget
from src_new.models.decoders import (
    AttentionPointerDecoder,
    GRUPointerDecoder,
    LSTMPointerDecoder,
    SigmoidSubsetDecoder,
)
from src_new.models.encoders import AttentionEncoder, LSTMEncoder
from src_new.problems import Problem, make_problem


class NCOModel(nn.Module):
    def __init__(
        self,
        *,
        problem: ProblemName,
        encoder_kind: EncoderKind,
        decoder_kind: DecoderKind,
        input_dim: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.0,
        tanh_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.problem_name = problem
        self.problem: Problem = make_problem(problem)
        self.encoder_kind = encoder_kind
        self.decoder_kind = decoder_kind
        if encoder_kind == "lstm":
            self.encoder = LSTMEncoder(
                input_dim=input_dim,
                d_model=d_model,
                num_layers=1,
                dropout=dropout,
            )
        elif encoder_kind in ("attention", "graph_attention"):
            self.encoder = AttentionEncoder(
                input_dim=input_dim,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported encoder_kind: {encoder_kind}")

        if decoder_kind == "attention_pointer":
            self.decoder = AttentionPointerDecoder(
                d_model=d_model,
                context_dim=self.problem.context_dim,
                tanh_clip=tanh_clip,
            )
        elif decoder_kind == "lstm_pointer":
            self.decoder = LSTMPointerDecoder(
                d_model=d_model,
                context_dim=self.problem.context_dim,
                tanh_clip=tanh_clip,
            )
        elif decoder_kind == "gru_pointer":
            self.decoder = GRUPointerDecoder(
                d_model=d_model,
                context_dim=self.problem.context_dim,
                tanh_clip=tanh_clip,
            )
        elif decoder_kind == "sigmoid_subset":
            self.decoder = SigmoidSubsetDecoder(d_model=d_model)
        else:
            raise ValueError(f"Unsupported decoder_kind: {decoder_kind}")

    def encode(self, batch: dict[str, Any]):
        node_features, adjacency, edge_features = self.problem.build_features(batch)
        return self.encoder(node_features, adjacency=adjacency, edge_features=edge_features)

    def forward(
        self,
        batch: dict[str, Any],
        *,
        decode_type: DecodeType = "greedy",
        target_actions: torch.Tensor | None = None,
    ) -> SolutionOutput:
        encoder_output = self.encode(batch)
        decode_state = ProblemDecodeState(
            problem=self.problem,
            batch=batch,
            state=self.problem.make_state(batch),
            target_actions=target_actions,
        )
        return self.decoder.decode(encoder_output, decode_state, decode_type)

    def supervised_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        target = self.problem.get_supervised_target(batch)
        if self.decoder_kind == "sigmoid_subset":
            return self._sigmoid_supervised_loss(batch, target)
        if target.actions is None:
            raise ValueError(f"{self.problem_name} is missing target_actions")
        output = self.forward(batch, decode_type="greedy", target_actions=target.actions)
        if output.log_probs is None:
            raise RuntimeError("Autoregressive decoder did not return log_probs")
        valid_steps = (target.actions >= 0).sum(dim=1).clamp_min(1).to(output.log_probs.dtype)
        return -(output.log_probs / valid_steps).mean()

    def _sigmoid_supervised_loss(
        self,
        batch: dict[str, Any],
        target: SupervisedTarget,
    ) -> torch.Tensor:
        encoder_output = self.encode(batch)
        if not isinstance(self.decoder, SigmoidSubsetDecoder):
            raise RuntimeError("Expected SigmoidSubsetDecoder")
        logits = self.decoder.logits(encoder_output)
        target_mask = target.selected_mask
        if target_mask is None:
            target_mask = _mask_from_actions(target.actions, logits.size(1), logits.device)
        if target_mask is None:
            raise ValueError(f"{self.problem_name} is missing target_mask")
        target_mask = target_mask.to(device=logits.device, dtype=logits.dtype)
        if target_mask.shape != logits.shape:
            if target_mask.size(1) == logits.size(1) - 1:
                pad = torch.zeros(target_mask.size(0), 1, dtype=target_mask.dtype, device=target_mask.device)
                target_mask = torch.cat([pad, target_mask], dim=1)
            else:
                raise ValueError(
                    f"target_mask shape {tuple(target_mask.shape)} does not match "
                    f"logits shape {tuple(logits.shape)}"
                )
        return F.binary_cross_entropy_with_logits(logits, target_mask)


def _mask_from_actions(
    actions: torch.Tensor | None,
    node_count: int,
    device: torch.device,
) -> torch.Tensor | None:
    if actions is None:
        return None
    mask = torch.zeros(actions.size(0), node_count, dtype=torch.float32, device=device)
    safe = actions.to(device=device)
    for row in range(safe.size(0)):
        valid = safe[row][(safe[row] >= 0) & (safe[row] < node_count)]
        if valid.numel():
            mask[row, valid.long()] = 1.0
    return mask
