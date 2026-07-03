import math

import torch
import torch.nn.functional as F
from torch import nn

from src.constants import DecodeType
from src.models.utils import init_uniform_


def attention_logits(
    query: torch.Tensor,
    keys: torch.Tensor,
    mask: torch.Tensor,
    tanh_clip: float,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(keys.size(-1))
    logits = scale * torch.matmul(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
    if tanh_clip > 0:
        logits = tanh_clip * torch.tanh(logits)
    return logits.masked_fill(mask, float("-inf"))


def select(
    logits: torch.Tensor, decode_type: DecodeType
) -> tuple[torch.Tensor, torch.Tensor]:
    if decode_type == "greedy":
        selected = logits.argmax(dim=-1)
    else:
        probs = F.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        selected = torch.multinomial(probs, 1).squeeze(-1)
    return selected, log_prob(logits, selected)


def log_prob(logits: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)
    log_p = torch.nan_to_num(log_p, nan=0.0, neginf=0.0)
    return log_p.gather(1, selected.unsqueeze(1)).squeeze(1)


def closed_route_length(loc: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    ordered = loc.gather(1, order.unsqueeze(-1).expand(-1, -1, loc.size(-1)))
    return (ordered[:, 1:] - ordered[:, :-1]).norm(p=2, dim=-1).sum(dim=1) + (
        ordered[:, 0] - ordered[:, -1]
    ).norm(p=2, dim=-1)


def init_linear(linear: nn.Linear) -> None:
    init_uniform_(linear.weight, linear.in_features)
    if linear.bias is not None:
        init_uniform_(linear.bias, linear.bias.numel())


def init_lstm(lstm: nn.LSTM, input_size: int, hidden_size: int) -> None:
    for name, parameter in lstm.named_parameters():
        if "weight_ih" in name:
            init_uniform_(parameter, input_size)
        elif "weight_hh" in name:
            init_uniform_(parameter, hidden_size)
        else:
            init_uniform_(parameter, parameter.numel())


def init_lstm_cell(
    lstm_cell: nn.LSTMCell,
    input_size: int,
    hidden_size: int,
) -> None:
    init_uniform_(lstm_cell.weight_ih, input_size)
    init_uniform_(lstm_cell.weight_hh, hidden_size)
    init_uniform_(lstm_cell.bias_ih, lstm_cell.bias_ih.numel())
    init_uniform_(lstm_cell.bias_hh, lstm_cell.bias_hh.numel())


def init_gru_cell(
    gru_cell: nn.GRUCell,
    input_size: int,
    hidden_size: int,
) -> None:
    init_uniform_(gru_cell.weight_ih, input_size)
    init_uniform_(gru_cell.weight_hh, hidden_size)
    init_uniform_(gru_cell.bias_ih, gru_cell.bias_ih.numel())
    init_uniform_(gru_cell.bias_hh, gru_cell.bias_hh.numel())
