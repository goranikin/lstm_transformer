import torch
from torch import nn

from src.models.modular.common import init_lstm


class LSTMEncoder(nn.Module):
    """Sequential encoder baseline for the modular NCO pipeline.

    This encoder reads the embedded nodes in their input order with a single
    LSTM. It is used to instantiate the Pointer-Network-style candidate in the
    comparison matrix, giving a recurrent encoder that can be paired with the
    same modular decoders as the attention-based variants. The returned node
    embeddings keep one output per input item, and the graph embedding is their
    mean, matching the interface of the attention encoder.
    """

    def __init__(self, d_h: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=d_h, hidden_size=d_h, batch_first=True)
        init_lstm(self.lstm, d_h, d_h)

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del mask
        encoded, _ = self.lstm(h)
        return encoded, encoded.mean(dim=1)
