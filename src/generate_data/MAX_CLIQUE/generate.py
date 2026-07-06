import argparse
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.generate_data.common import instance_seed, iter_instance_indices, write_jsonl
from src.generate_data.graph_utils import (
    adjacency_to_edges,
    generate_erdos_renyi_graph,
)
from src.generate_data.MAX_CLIQUE.algorithms import solve_gurobi


class MaxCliqueGenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_instances: int = Field(gt=0)
    start_index: int = Field(default=0, ge=0)
    num_nodes: int = Field(gt=0)
    edge_probability: float = Field(ge=0.0, le=1.0)
    seed: int
    output_path: str
    solver_time_limit_sec: float | None = Field(default=None, gt=0.0)


def generate_max_clique_instance(
    num_nodes: int,
    edge_probability: float,
    seed: int,
):
    return generate_erdos_renyi_graph(num_nodes, edge_probability, seed)


def iter_max_clique_records(
    config: MaxCliqueGenerationConfig,
) -> Iterator[dict[str, Any]]:
    for index in iter_instance_indices(config.start_index, config.num_instances):
        seed = instance_seed(config.seed, index)
        adjacency = generate_max_clique_instance(
            config.num_nodes,
            config.edge_probability,
            seed,
        )
        record: dict[str, Any] = {
            "problem": "maximum_clique",
            "category": "partial_subset",
            "index": index,
            "seed": seed,
            "num_nodes": config.num_nodes,
            "edge_probability": config.edge_probability,
            "edges": [[u, v] for u, v in adjacency_to_edges(adjacency)],
        }
        solution = solve_gurobi(
            adjacency,
            seed=seed,
            time_limit_sec=config.solver_time_limit_sec,
        )
        record["solutions"] = {solution.algorithm: solution.to_record()}
        yield record


def generate_max_clique_dataset(config: MaxCliqueGenerationConfig) -> int:
    return write_jsonl(iter_max_clique_records(config), config.output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Maximum Clique JSONL data"
    )
    parser.add_argument("--num-instances", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--edge-probability", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--solver-time-limit-sec", type=float, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = MaxCliqueGenerationConfig(
        num_instances=args.num_instances,
        start_index=args.start_index,
        num_nodes=args.num_nodes,
        edge_probability=args.edge_probability,
        seed=args.seed,
        output_path=args.output_path,
        solver_time_limit_sec=args.solver_time_limit_sec,
    )
    written = generate_max_clique_dataset(config)
    print(f"Wrote {written} Maximum Clique instances to {config.output_path}")


if __name__ == "__main__":
    main()
