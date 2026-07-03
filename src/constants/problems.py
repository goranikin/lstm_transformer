from typing import Literal, TypeAlias

ProblemName: TypeAlias = Literal[
    "tsp",
    "cvrp",
    "mis",
    "maximum_clique",
    "minimum_vertex_cover",
    "knapsack",
    "orienteering",
]
ProblemType = ProblemName

PROBLEM_NAMES: tuple[ProblemName, ...] = (
    "tsp",
    "cvrp",
    "mis",
    "maximum_clique",
    "minimum_vertex_cover",
    "knapsack",
    "orienteering",
)

TOTAL_PROBLEMS = frozenset({"tsp", "cvrp"})
SUBSET_PROBLEMS = frozenset(
    {
        "mis",
        "maximum_clique",
        "minimum_vertex_cover",
        "knapsack",
        "orienteering",
    }
)
MINIMIZE_PROBLEMS = frozenset({"tsp", "cvrp", "minimum_vertex_cover"})
MAXIMIZE_PROBLEMS = frozenset(
    {
        "mis",
        "maximum_clique",
        "knapsack",
        "orienteering",
    }
)

PROBLEM_FEATURE_DIMS: dict[ProblemName, int] = {
    "tsp": 2,
    "cvrp": 3,
    "mis": 1,
    "maximum_clique": 1,
    "minimum_vertex_cover": 1,
    "knapsack": 2,
    "orienteering": 3,
}
