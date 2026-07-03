import argparse
from collections.abc import Iterator
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.generate_data.common import instance_seed, write_jsonl
from src.generate_data.TSP.algorithms import solve_concorde


class TSPGenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_instances: int = Field(gt=0)
    num_nodes: int = Field(gt=0)
    seed: int
    output_path: str
    concorde_executable: str | None = None
    solver_timeout_sec: float | None = None


def generate_tsp_instance(num_nodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((num_nodes, 2), dtype=np.float64)


def iter_tsp_records(config: TSPGenerationConfig) -> Iterator[dict[str, Any]]:
    for index in range(config.num_instances):
        seed = instance_seed(config.seed, index)
        coords = generate_tsp_instance(config.num_nodes, seed)
        record: dict[str, Any] = {
            "problem": "tsp",
            "index": index,
            "seed": seed,
            "num_nodes": config.num_nodes,
            "coordinates": coords.tolist(),
        }
        solution = solve_concorde(
            coords,
            executable=config.concorde_executable,
            timeout_sec=config.solver_timeout_sec,
        )
        record["solutions"] = {solution.algorithm: solution.to_record()}
        yield record


def generate_tsp_dataset(config: TSPGenerationConfig) -> int:
    return write_jsonl(iter_tsp_records(config), config.output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic TSP JSONL data"
    )
    parser.add_argument("--num-instances", type=int, required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--concorde-executable", type=str, default=None)
    parser.add_argument("--solver-timeout-sec", type=float, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = TSPGenerationConfig(
        num_instances=args.num_instances,
        num_nodes=args.num_nodes,
        seed=args.seed,
        output_path=args.output_path,
        concorde_executable=args.concorde_executable,
        solver_timeout_sec=args.solver_timeout_sec,
    )
    written = generate_tsp_dataset(config)
    print(f"Wrote {written} TSP instances to {config.output_path}")


if __name__ == "__main__":
    main()
