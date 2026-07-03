import argparse
from collections.abc import Iterator
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.generate_data.common import instance_seed, write_jsonl
from src.generate_data.MIS.algorithms import adjacency_to_edges, solve_kamis


class MISGenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_instances: int = Field(gt=0)
    num_nodes: int = Field(gt=0)
    edge_probability: float = Field(ge=0.0, le=1.0)
    seed: int
    output_path: str
    kamis_executable: str | None = None
    solver_time_limit_sec: float | None = None


def generate_mis_instance(
    num_nodes: int,
    edge_probability: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    upper = rng.random((num_nodes, num_nodes)) < edge_probability
    upper = np.triu(upper, k=1)
    adjacency = upper | upper.T
    return adjacency.astype(np.bool_)


def iter_mis_records(config: MISGenerationConfig) -> Iterator[dict[str, Any]]:
    for index in range(config.num_instances):
        seed = instance_seed(config.seed, index)
        adjacency = generate_mis_instance(
            config.num_nodes,
            config.edge_probability,
            seed,
        )
        edges = adjacency_to_edges(adjacency)
        record: dict[str, Any] = {
            "problem": "mis",
            "index": index,
            "seed": seed,
            "num_nodes": config.num_nodes,
            "edge_probability": config.edge_probability,
            "edges": [[u, v] for u, v in edges],
        }
        solution = solve_kamis(
            adjacency,
            executable=config.kamis_executable,
            seed=seed,
            time_limit_sec=config.solver_time_limit_sec,
        )
        record["solutions"] = {solution.algorithm: solution.to_record()}
        yield record


def generate_mis_dataset(config: MISGenerationConfig) -> int:
    return write_jsonl(iter_mis_records(config), config.output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic MIS JSONL data"
    )
    parser.add_argument("--num-instances", type=int, required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--edge-probability", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--kamis-executable", type=str, default=None)
    parser.add_argument("--solver-time-limit-sec", type=float, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = MISGenerationConfig(
        num_instances=args.num_instances,
        num_nodes=args.num_nodes,
        edge_probability=args.edge_probability,
        seed=args.seed,
        output_path=args.output_path,
        kamis_executable=args.kamis_executable,
        solver_time_limit_sec=args.solver_time_limit_sec,
    )
    written = generate_mis_dataset(config)
    print(f"Wrote {written} MIS instances to {config.output_path}")


if __name__ == "__main__":
    main()
