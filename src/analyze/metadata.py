"""Stable experiment metadata used to label analysis outputs."""

from dataclasses import dataclass

from src.constants import DEFAULT_TARGET_ALGORITHM


@dataclass(frozen=True)
class ProblemMetadata:
    family: str
    topology: str
    solver: str
    objective_sense: str


PROBLEM_METADATA: dict[str, ProblemMetadata] = {
    "tsp": ProblemMetadata(
        family="sequential_construction",
        topology="euclidean_tour",
        solver=DEFAULT_TARGET_ALGORITHM["tsp"],
        objective_sense="min",
    ),
    "cvrp": ProblemMetadata(
        family="sequential_construction",
        topology="capacitated_euclidean_routes",
        solver=DEFAULT_TARGET_ALGORITHM["cvrp"],
        objective_sense="min",
    ),
    "orienteering": ProblemMetadata(
        family="sequential_construction",
        topology="prize_collecting_route",
        solver=DEFAULT_TARGET_ALGORITHM["orienteering"],
        objective_sense="max",
    ),
    "knapsack": ProblemMetadata(
        family="subset_selection",
        topology="capacity_constrained_items",
        solver=DEFAULT_TARGET_ALGORITHM["knapsack"],
        objective_sense="max",
    ),
    "mis": ProblemMetadata(
        family="subset_selection",
        topology="sparse_graph_independence",
        solver=DEFAULT_TARGET_ALGORITHM["mis"],
        objective_sense="max",
    ),
    "max_clique": ProblemMetadata(
        family="subset_selection",
        topology="dense_graph_clique",
        solver=DEFAULT_TARGET_ALGORITHM["max_clique"],
        objective_sense="max",
    ),
    "vertex_cover": ProblemMetadata(
        family="subset_selection",
        topology="sparse_graph_cover",
        solver=DEFAULT_TARGET_ALGORITHM["vertex_cover"],
        objective_sense="min",
    ),
}


DECODER_FAMILY: dict[str, str] = {
    "attention_pointer": "autoregressive_attention",
    "lstm_pointer": "autoregressive_lstm",
    "gru_pointer": "autoregressive_gru",
    "transformer_pointer": "autoregressive_causal_transformer",
    "sigmoid_subset": "parallel_sigmoid_with_repair",
}


DECODER_DESCRIPTION: dict[str, str] = {
    "attention_pointer": (
        "Autoregressive pointer with a learned query, attention glimpse, and "
        "masked pointer scores."
    ),
    "lstm_pointer": (
        "Autoregressive pointer whose recurrent decoder state is an LSTMCell."
    ),
    "gru_pointer": (
        "Autoregressive pointer whose recurrent decoder state is a GRUCell."
    ),
    "transformer_pointer": (
        "Autoregressive pointer whose decoder state is the newest output of a "
        "causal Transformer over the complete decoder-input history."
    ),
    "sigmoid_subset": (
        "Independent node logits followed by problem-specific repair; it is not "
        "an autoregressive sequence decoder."
    ),
}


def problem_metadata(problem: str) -> ProblemMetadata:
    try:
        return PROBLEM_METADATA[problem]
    except KeyError as error:
        raise ValueError(f"No analysis metadata registered for problem {problem!r}") from error
