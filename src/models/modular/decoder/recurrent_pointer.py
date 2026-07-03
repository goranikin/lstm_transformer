from typing import Literal

import torch
from torch import nn

from src.models.modular.common import (
    DecodeType,
    attention_logits,
    init_gru_cell,
    init_linear,
    init_lstm_cell,
    log_prob,
    select,
)
from src.models.utils import init_uniform_


class RecurrentPointerDecoder(nn.Module):
    """LSTM/GRU pointer decoder over attention-encoder node embeddings."""

    def __init__(
        self,
        d_h: int,
        tanh_clip: float,
        cell_kind: Literal["lstm", "gru"],
    ) -> None:
        super().__init__()
        self.d_h = d_h
        self.tanh_clip = tanh_clip
        self.cell_kind = cell_kind
        self.logit_key_proj = nn.Linear(d_h, d_h, bias=False)
        self.initial_hidden = nn.Linear(d_h, d_h)
        self.start_input = nn.Parameter(torch.empty(d_h))
        if cell_kind == "lstm":
            self.initial_cell = nn.Linear(d_h, d_h)
            self.cell = nn.LSTMCell(input_size=d_h, hidden_size=d_h)
            init_linear(self.initial_cell)
            init_lstm_cell(self.cell, d_h, d_h)
        else:
            self.initial_cell = None
            self.cell = nn.GRUCell(input_size=d_h, hidden_size=d_h)
            init_gru_cell(self.cell, d_h, d_h)
        init_linear(self.logit_key_proj)
        init_linear(self.initial_hidden)
        init_uniform_(self.start_input, d_h)

    def forward(
        self,
        h: torch.Tensor,
        h_bar: torch.Tensor,
        *,
        decode_type: DecodeType,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = h.shape
        device = h.device
        logit_key = self.logit_key_proj(h)
        selected_mask = torch.zeros(
            batch_size,
            num_nodes,
            dtype=torch.bool,
            device=device,
        )
        batch_idx = torch.arange(batch_size, device=device)
        hidden = torch.tanh(self.initial_hidden(h_bar))
        cell = (
            torch.tanh(self.initial_cell(h_bar))
            if self.cell_kind == "lstm" and self.initial_cell is not None
            else None
        )
        decoder_input = self.start_input.expand(batch_size, -1)
        actions: list[torch.Tensor] = []
        log_p_steps: list[torch.Tensor] = []

        for step in range(num_nodes):
            if self.cell_kind == "lstm":
                if cell is None or not isinstance(self.cell, nn.LSTMCell):
                    raise RuntimeError("LSTM pointer decoder is misconfigured")
                hidden, cell = self.cell(decoder_input, (hidden, cell))
            else:
                if not isinstance(self.cell, nn.GRUCell):
                    raise RuntimeError("GRU pointer decoder is misconfigured")
                hidden = self.cell(decoder_input, hidden)

            logits = attention_logits(hidden, logit_key, selected_mask, self.tanh_clip)
            if target is None:
                selected, log_p = select(logits, decode_type)
            else:
                selected = target[:, step]
                log_p = log_prob(logits, selected)
            selected_mask = selected_mask.scatter(1, selected.unsqueeze(1), True)
            decoder_input = h[batch_idx, selected]
            actions.append(selected)
            log_p_steps.append(log_p)

        return torch.stack(actions, dim=1), torch.stack(log_p_steps, dim=1).sum(dim=1)
