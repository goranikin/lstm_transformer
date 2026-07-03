import torch
from torch import nn

from src.core import EncoderOutput


class AttentionEncoder(nn.Module):
    """Graph-attention style encoder with the unified EncoderOutput contract."""

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
        del adjacency, edge_features
        h = self.input_proj(node_features.float())
        for layer in self.layers:
            h = layer(h)
        return EncoderOutput(node_embeddings=h, graph_embedding=h.mean(dim=1))


class LSTMEncoder(nn.Module):
    """Pointer-Network-style sequence encoder.

    The bidirectional LSTM keeps the same output dimensionality as the attention
    encoder. For graph/set problems, callers should shuffle input order during
    training if they want to reduce order bias.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if bidirectional and d_model % 2 != 0:
            raise ValueError("d_model must be even for bidirectional LSTMEncoder")
        hidden_size = d_model // 2 if bidirectional else d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
    ) -> EncoderOutput:
        del adjacency, edge_features
        h, _ = self.lstm(self.input_proj(node_features.float()))
        return EncoderOutput(node_embeddings=h, graph_embedding=h.mean(dim=1))
