import torch
from torch import nn

from src.models.attention_model.attention_layer import MultiHeadAttention
from src.constants import DecodeType
from src.models.modular.common import (
    attention_logits,
    init_linear,
    log_prob,
    select,
)
from src.models.utils import init_uniform_


class AttentionPointerDecoder(nn.Module):
    """AM-style attention decoder for total-order construction."""

    def __init__(self, d_h: int, n_heads: int, tanh_clip: float) -> None:
        super().__init__()
        self.d_h = d_h
        self.tanh_clip = tanh_clip
        self.node_proj = nn.Linear(d_h, 3 * d_h, bias=False)
        self.step_proj = nn.Linear(3 * d_h, d_h, bias=False)
        self.glimpse_mha = MultiHeadAttention(
            n_heads=n_heads,
            d_h=d_h,
            input_dim=d_h,
        )
        self.prev_placeholder = nn.Parameter(torch.empty(d_h))
        self.first_placeholder = nn.Parameter(torch.empty(d_h))
        init_linear(self.node_proj)
        init_linear(self.step_proj)
        init_uniform_(self.prev_placeholder, d_h)
        init_uniform_(self.first_placeholder, d_h)

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
        glimpse_key, glimpse_value, logit_key = self.node_proj(h).chunk(3, dim=-1)
        selected_mask = torch.zeros(
            batch_size,
            num_nodes,
            dtype=torch.bool,
            device=device,
        )
        batch_idx = torch.arange(batch_size, device=device)
        first = torch.zeros(batch_size, dtype=torch.long, device=device)
        previous = torch.zeros(batch_size, dtype=torch.long, device=device)
        actions: list[torch.Tensor] = []
        log_p_steps: list[torch.Tensor] = []

        for step in range(num_nodes):
            if step == 0:
                context = torch.cat(
                    [
                        h_bar,
                        self.prev_placeholder.expand(batch_size, -1),
                        self.first_placeholder.expand(batch_size, -1),
                    ],
                    dim=-1,
                )
            else:
                context = torch.cat(
                    [h_bar, h[batch_idx, previous], h[batch_idx, first]],
                    dim=-1,
                )

            query = self.step_proj(context)
            glimpse = self.glimpse_mha(
                query.unsqueeze(1),
                glimpse_key,
                value=glimpse_value,
                mask=selected_mask.unsqueeze(1),
            ).squeeze(1)
            logits = attention_logits(
                glimpse,
                logit_key,
                selected_mask,
                self.tanh_clip,
            )
            if target is None:
                selected, log_p = select(logits, decode_type)
            else:
                selected = target[:, step]
                log_p = log_prob(logits, selected)

            if step == 0:
                first = selected
            previous = selected
            selected_mask = selected_mask.scatter(1, selected.unsqueeze(1), True)
            actions.append(selected)
            log_p_steps.append(log_p)

        return torch.stack(actions, dim=1), torch.stack(log_p_steps, dim=1).sum(dim=1)
