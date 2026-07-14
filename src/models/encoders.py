import torch
from torch import nn

from src.core import EncoderOutput


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int = 3,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    batch_first=True,
                    activation="relu",
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
    ) -> EncoderOutput:
        # For now, we don't use the adjacency and edge features
        del adjacency, edge_features
        h = self.input_proj(node_features.float())
        for layer in self.layers:
            h = layer(h)
        return EncoderOutput(node_embeddings=h, graph_embedding=h.mean(dim=1))
