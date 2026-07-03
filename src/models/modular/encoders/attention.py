import torch
from torch import nn

from src.models.attention_model.attention_layer import GraphAttentionEncoder


class AttentionEncoder(nn.Module):
    """Attention-model encoder used by the modular NCO variants.

    This block is the Transformer-style graph attention encoder from the
    original Attention Model family. It performs repeated multi-head
    self-attention and feed-forward updates over embedded problem nodes, then
    returns both per-node embeddings and a mean graph embedding. Passing a mask
    makes the same block usable as a graph-attention encoder for graph problems,
    while leaving the mask unset gives the fully connected AM encoder for
    routing and item-selection problems.
    """

    def __init__(
        self,
        *,
        n_layers: int,
        n_heads: int,
        d_h: int,
        d_ff: int,
        normalization: str,
    ) -> None:
        super().__init__()
        self.encoder = GraphAttentionEncoder(
            n_layers=n_layers,
            n_heads=n_heads,
            d_h=d_h,
            d_ff=d_ff,
            normalization=normalization,
        )

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(h, mask=mask)
