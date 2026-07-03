from typing import Literal, TypeAlias

DecodeType: TypeAlias = Literal["greedy", "sampling"]

EncoderKind: TypeAlias = Literal["lstm", "attention", "graph_attention"]

DecoderKind: TypeAlias = Literal[
    "attention_pointer",
    "lstm_pointer",
    "gru_pointer",
    "attention_subset",
    "lstm_subset",
    "gru_subset",
    "sigmoid_subset",
]

RecurrentCellKind: TypeAlias = Literal["lstm", "gru"]

ObjectiveSense: TypeAlias = Literal["min", "max"]
