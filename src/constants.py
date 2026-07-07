from typing import Literal, TypeAlias

ProblemName: TypeAlias = Literal[
    "tsp",
    "cvrp",
    "orienteering",
    "knapsack",
    "mis",
    "max_clique",
    "vertex_cover",
]

EncoderKind: TypeAlias = Literal["lstm", "attention", "graph_attention"]

DecoderKind: TypeAlias = Literal[
    "attention_pointer",
    "lstm_pointer",
    "gru_pointer",
    "sigmoid_subset",
]

DecodeType: TypeAlias = Literal["greedy", "sampling"]
TrainingMode: TypeAlias = Literal["supervised", "rl"]
ObjectiveSense: TypeAlias = Literal["min", "max"]

PROBLEM_NAMES: tuple[ProblemName, ...] = (
    "tsp",
    "cvrp",
    "orienteering",
    "knapsack",
    "mis",
    "max_clique",
    "vertex_cover",
)

ENCODER_KINDS: tuple[EncoderKind, ...] = ("attention", "lstm", "graph_attention")

MATRIX_ENCODERS: tuple[EncoderKind, ...] = ("attention",)

DECODER_KINDS: tuple[DecoderKind, ...] = (
    "attention_pointer",
    "lstm_pointer",
    "gru_pointer",
    "sigmoid_subset",
)

AUTOREGRESSIVE_DECODERS = frozenset(
    {"attention_pointer", "lstm_pointer", "gru_pointer"}
)

MINIMIZATION_PROBLEMS = frozenset({"tsp", "cvrp", "vertex_cover"})
MAXIMIZATION_PROBLEMS = frozenset({"orienteering", "knapsack", "mis", "max_clique"})

DEFAULT_SEEDS = (1234, 4321, 9999)

DEFAULT_SPLIT_INSTANCES: dict[str, int] = {
    "train": 64_000,
    "val": 10_000,
    "test": 10_000,
}

DEFAULT_TARGET_ALGORITHM: dict[ProblemName, str] = {
    "tsp": "concorde",
    "cvrp": "gurobi",
    "orienteering": "gurobi",
    "knapsack": "dynamic_programming",
    "mis": "gurobi",
    "max_clique": "gurobi",
    "vertex_cover": "gurobi",
}

PROBLEM_PATH_DIR: dict[ProblemName, str] = {
    "tsp": "tsp",
    "cvrp": "cvrp",
    "orienteering": "orienteering",
    "knapsack": "knapsack",
    "mis": "mis",
    "max_clique": "max_clique",
    "vertex_cover": "vertex_cover",
}

PROBLEM_FILE_PREFIX: dict[ProblemName, str] = {
    "tsp": "tsp50",
    "cvrp": "cvrp50",
    "orienteering": "orienteering50",
    "knapsack": "knapsack100",
    "mis": "mis100_p015",
    "max_clique": "max_clique100_p050",
    "vertex_cover": "vertex_cover100_p015",
}
