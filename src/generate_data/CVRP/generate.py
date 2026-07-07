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
from src.generate_data.CVRP.algorithms import solve_gurobi


class CVRPGenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_instances: int = Field(gt=0)
    start_index: int = Field(default=0, ge=0)
    num_customers: int = Field(gt=0)
    min_demand: int = Field(default=1, gt=0)
    max_demand: int = Field(default=9, gt=0)
    vehicle_capacity: int | None = Field(default=None, gt=0)
    capacity_ratio: float = Field(default=0.25, gt=0.0, le=1.0)
    max_vehicles: int | None = Field(default=None, gt=0)
    seed: int
    output_path: str
    solver_time_limit_sec: float = Field(default=30.0, gt=0.0)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.max_demand < self.min_demand:
            raise ValueError("max_demand must be >= min_demand")
        return self


def generate_cvrp_instance(
    num_customers: int,
    min_demand: int,
    max_demand: int,
    vehicle_capacity: int | None,
    capacity_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    depot = rng.random(2, dtype=np.float64)
    coordinates = rng.random((num_customers, 2), dtype=np.float64)
    demands = rng.integers(
        min_demand,
        max_demand + 1,
        size=num_customers,
        dtype=np.int64,
    )
    if vehicle_capacity is None:
        capacity = max(
            int(demands.max()),
            int(round(float(demands.sum()) * capacity_ratio)),
        )
    else:
        capacity = int(vehicle_capacity)
    if int(demands.max()) > capacity:
        raise ValueError("vehicle_capacity must cover the largest generated demand")
    return depot, coordinates, demands, capacity


def iter_cvrp_records(config: CVRPGenerationConfig) -> Iterator[dict[str, Any]]:
    for index in iter_instance_indices(config.start_index, config.num_instances):
        seed = instance_seed(config.seed, index)
        depot, coordinates, demands, vehicle_capacity = generate_cvrp_instance(
            config.num_customers,
            config.min_demand,
            config.max_demand,
            config.vehicle_capacity,
            config.capacity_ratio,
            seed,
        )
        record: dict[str, Any] = {
            "problem": "cvrp",
            "category": "total_set_with_routes",
            "index": index,
            "seed": seed,
            "num_customers": config.num_customers,
            "depot": depot.tolist(),
            "coordinates": coordinates.tolist(),
            "demands": demands.tolist(),
            "vehicle_capacity": vehicle_capacity,
            "capacity_ratio": config.capacity_ratio,
        }
        if config.max_vehicles is not None:
            record["max_vehicles"] = config.max_vehicles
        solution = solve_gurobi(
            depot,
            coordinates,
            demands,
            vehicle_capacity,
            max_vehicles=config.max_vehicles,
            seed=seed,
            time_limit_sec=config.solver_time_limit_sec,
        )
        record["solutions"] = {solution.algorithm: solution.to_record()}
        yield record


def generate_cvrp_dataset(config: CVRPGenerationConfig) -> int:
    return write_jsonl(iter_cvrp_records(config), config.output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic CVRP JSONL data"
    )
    parser.add_argument("--num-instances", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-customers", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output JSONL path (default: ~/local_db/lstm_transformer/<problem>/... from seed).",
    )
    parser.add_argument("--min-demand", type=int, default=1)
    parser.add_argument("--max-demand", type=int, default=9)
    parser.add_argument("--vehicle-capacity", type=int, default=None)
    parser.add_argument("--capacity-ratio", type=float, default=0.25)
    parser.add_argument("--max-vehicles", type=int, default=None)
    parser.add_argument("--solver-time-limit-sec", type=float, default=30.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = CVRPGenerationConfig(
        num_instances=args.num_instances,
        start_index=args.start_index,
        num_customers=args.num_customers,
        min_demand=args.min_demand,
        max_demand=args.max_demand,
        vehicle_capacity=args.vehicle_capacity,
        capacity_ratio=args.capacity_ratio,
        max_vehicles=args.max_vehicles,
        seed=args.seed,
        output_path=resolve_output_path("cvrp", seed=args.seed, output_path=args.output_path),
        solver_time_limit_sec=args.solver_time_limit_sec,
    )
    written = generate_cvrp_dataset(config)
    print(f"Wrote {written} CVRP instances to {config.output_path}")


if __name__ == "__main__":
    main()
