import torch
from torch import nn

from src.models.attention_model.attention_layer import MultiHeadAttention
from src.models.modular.common import attention_logits, init_linear
from src.models.utils import init_uniform_


class AttentionSubsetDecoder(nn.Module):
    """Attention autoregressive decoder for partial-set construction."""

    def __init__(self, d_h: int, n_heads: int, tanh_clip: float) -> None:
        super().__init__()
        self.d_h = d_h
        self.tanh_clip = tanh_clip
        self.node_proj = nn.Linear(d_h, 3 * d_h, bias=False)
        self.step_proj = nn.Linear(2 * d_h + 2, d_h, bias=False)
        self.glimpse_mha = MultiHeadAttention(
            n_heads=n_heads,
            d_h=d_h,
            input_dim=d_h,
        )
        self.prev_placeholder = nn.Parameter(torch.empty(d_h))
        self.stop_embedding = nn.Parameter(torch.empty(d_h))
        self.stop_logit_key = nn.Parameter(torch.empty(d_h))
        init_linear(self.node_proj)
        init_linear(self.step_proj)
        init_uniform_(self.prev_placeholder, d_h)
        init_uniform_(self.stop_embedding, d_h)
        init_uniform_(self.stop_logit_key, d_h)

    def step(
        self,
        h: torch.Tensor,
        h_bar: torch.Tensor,
        prev_emb: torch.Tensor,
        context: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = h.size(0)
        node_glimpse_key, node_glimpse_value, node_logit_key = self.node_proj(h).chunk(
            3,
            dim=-1,
        )
        stop_embedding = self.stop_embedding.view(1, 1, -1).expand(batch_size, 1, -1)
        stop_glimpse_key = self.stop_logit_key.view(1, 1, -1).expand(batch_size, 1, -1)
        stop_logit_key = self.stop_logit_key.view(1, 1, -1).expand(batch_size, 1, -1)
        query = self.step_proj(torch.cat([h_bar, prev_emb, context], dim=-1))
        glimpse = self.glimpse_mha(
            query.unsqueeze(1),
            torch.cat([node_glimpse_key, stop_glimpse_key], dim=1),
            value=torch.cat([node_glimpse_value, stop_embedding], dim=1),
            mask=action_mask.unsqueeze(1),
        ).squeeze(1)
        return attention_logits(
            glimpse,
            torch.cat([node_logit_key, stop_logit_key], dim=1),
            action_mask,
            self.tanh_clip,
        )
