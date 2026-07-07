import argparse
from collections.abc import Iterator
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.generate_data.common import (
    instance_seed,
    iter_instance_indices,
    resolve_output_path,
    write_jsonl,
)
from src.generate_data.ORIENTEERING.algorithms import solve_gurobi


class OrienteeringGenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_instances: int = Field(gt=0)
    start_index: int = Field(default=0, ge=0)
    num_nodes: int = Field(gt=0)
    min_prize: int = Field(default=1, ge=0)
    max_prize: int = Field(default=100, ge=0)
    travel_budget: float | None = Field(default=None, gt=0.0)
    budget_ratio: float = Field(default=0.4, gt=0.0, le=1.0)
    seed: int
    output_path: str
    solver_time_limit_sec: float = Field(default=30.0, gt=0.0)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.max_prize < self.min_prize:
            raise ValueError("max_prize must be >= min_prize")
        return self


def generate_orienteering_instance(
    num_nodes: int,
    min_prize: int,
    max_prize: int,
    travel_budget: float | None,
    budget_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    depot = rng.random(2, dtype=np.float64)
    coordinates = rng.random((num_nodes, 2), dtype=np.float64)
    prizes = rng.integers(min_prize, max_prize + 1, size=num_nodes, dtype=np.int64)
    if travel_budget is None:
        full_tour = _nearest_neighbor_full_tour_length(depot, coordinates)
        shortest_round_trip = float(
            2.0 * np.linalg.norm(coordinates - depot, axis=1).min()
        )
        budget = max(shortest_round_trip, full_tour * budget_ratio)
    else:
        budget = float(travel_budget)
    return depot, coordinates, prizes, budget


def iter_orienteering_records(
    config: OrienteeringGenerationConfig,
) -> Iterator[dict[str, Any]]:
    for index in iter_instance_indices(config.start_index, config.num_instances):
        seed = instance_seed(config.seed, index)
        depot, coordinates, prizes, travel_budget = generate_orienteering_instance(
            config.num_nodes,
            config.min_prize,
            config.max_prize,
            config.travel_budget,
            config.budget_ratio,
            seed,
        )
        record: dict[str, Any] = {
            "problem": "orienteering",
            "category": "hybrid_subset_sequence",
            "index": index,
            "seed": seed,
            "num_nodes": config.num_nodes,
            "depot": depot.tolist(),
            "coordinates": coordinates.tolist(),
            "prizes": prizes.tolist(),
            "travel_budget": travel_budget,
            "budget_ratio": config.budget_ratio,
        }
        solution = solve_gurobi(
            depot,
            coordinates,
            prizes,
            travel_budget,
            seed=seed,
            time_limit_sec=config.solver_time_limit_sec,
        )
        record["solutions"] = {solution.algorithm: solution.to_record()}
        yield record


def generate_orienteering_dataset(config: OrienteeringGenerationConfig) -> int:
    return write_jsonl(iter_orienteering_records(config), config.output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Orienteering JSONL data"
    )
    parser.add_argument("--num-instances", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output JSONL path (default: ~/local_db/lstm_transformer/<problem>/... from seed).",
    )
    parser.add_argument("--min-prize", type=int, default=1)
    parser.add_argument("--max-prize", type=int, default=100)
    parser.add_argument("--travel-budget", type=float, default=None)
    parser.add_argument("--budget-ratio", type=float, default=0.4)
    parser.add_argument("--solver-time-limit-sec", type=float, default=30.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = OrienteeringGenerationConfig(
        num_instances=args.num_instances,
        start_index=args.start_index,
        num_nodes=args.num_nodes,
        min_prize=args.min_prize,
        max_prize=args.max_prize,
        travel_budget=args.travel_budget,
        budget_ratio=args.budget_ratio,
        seed=args.seed,
        output_path=resolve_output_path(
            "orienteering", seed=args.seed, output_path=args.output_path
        ),
        solver_time_limit_sec=args.solver_time_limit_sec,
    )
    written = generate_orienteering_dataset(config)
    print(f"Wrote {written} Orienteering instances to {config.output_path}")


def _nearest_neighbor_full_tour_length(
    depot: np.ndarray,
    coordinates: np.ndarray,
) -> float:
    remaining = set(range(len(coordinates)))
    current = depot
    length = 0.0
    while remaining:
        node = min(
            remaining,
            key=lambda item: float(np.linalg.norm(coordinates[item] - current)),
        )
        length += float(np.linalg.norm(coordinates[node] - current))
        current = coordinates[node]
        remaining.remove(node)
    length += float(np.linalg.norm(current - depot))
    return length


if __name__ == "__main__":
    main()
