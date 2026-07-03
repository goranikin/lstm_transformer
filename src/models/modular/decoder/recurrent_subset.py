from typing import Literal

import torch
from torch import nn

from src.models.modular.common import (
    attention_logits,
    init_gru_cell,
    init_linear,
    init_lstm_cell,
)
from src.models.utils import init_uniform_


class RecurrentSubsetDecoder(nn.Module):
    """LSTM/GRU autoregressive decoder for partial-set construction."""

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
        self.prev_placeholder = nn.Parameter(torch.empty(d_h))
        self.stop_logit_key = nn.Parameter(torch.empty(d_h))
        if cell_kind == "lstm":
            self.initial_cell = nn.Linear(d_h, d_h)
            self.cell = nn.LSTMCell(input_size=d_h + 2, hidden_size=d_h)
            init_linear(self.initial_cell)
            init_lstm_cell(self.cell, d_h + 2, d_h)
        else:
            self.initial_cell = None
            self.cell = nn.GRUCell(input_size=d_h + 2, hidden_size=d_h)
            init_gru_cell(self.cell, d_h + 2, d_h)
        init_linear(self.logit_key_proj)
        init_linear(self.initial_hidden)
        init_uniform_(self.prev_placeholder, d_h)
        init_uniform_(self.stop_logit_key, d_h)

    def initial_state(
        self,
        h_bar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = torch.tanh(self.initial_hidden(h_bar))
        cell = (
            torch.tanh(self.initial_cell(h_bar))
            if self.cell_kind == "lstm" and self.initial_cell is not None
            else None
        )
        return hidden, cell

    def step(
        self,
        h: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor | None,
        decoder_input: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if self.cell_kind == "lstm":
            if cell is None or not isinstance(self.cell, nn.LSTMCell):
                raise RuntimeError("LSTM subset decoder is misconfigured")
            hidden, cell = self.cell(decoder_input, (hidden, cell))
        else:
            if not isinstance(self.cell, nn.GRUCell):
                raise RuntimeError("GRU subset decoder is misconfigured")
            hidden = self.cell(decoder_input, hidden)
        batch_size = h.size(0)
        stop_key = self.stop_logit_key.view(1, 1, -1).expand(batch_size, 1, -1)
        logits = attention_logits(
            hidden,
            torch.cat([self.logit_key_proj(h), stop_key], dim=1),
            action_mask,
            self.tanh_clip,
        )
        return hidden, cell, logits
