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
from src.generate_data.KNAPSACK.algorithms import solve_dynamic_programming


class KnapsackGenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_instances: int = Field(gt=0)
    start_index: int = Field(default=0, ge=0)
    num_items: int = Field(gt=0)
    min_weight: int = Field(default=1, gt=0)
    max_weight: int = Field(default=100, gt=0)
    min_value: int = Field(default=1, ge=0)
    max_value: int = Field(default=100, ge=0)
    capacity_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    seed: int
    output_path: str

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.max_weight < self.min_weight:
            raise ValueError("max_weight must be >= min_weight")
        if self.max_value < self.min_value:
            raise ValueError("max_value must be >= min_value")
        return self


def generate_knapsack_instance(
    num_items: int,
    min_weight: int,
    max_weight: int,
    min_value: int,
    max_value: int,
    capacity_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    weights = rng.integers(min_weight, max_weight + 1, size=num_items, dtype=np.int64)
    values = rng.integers(min_value, max_value + 1, size=num_items, dtype=np.int64)
    capacity = max(1, int(round(float(weights.sum()) * capacity_ratio)))
    capacity = max(capacity, int(weights.min()))
    return weights, values, capacity


def iter_knapsack_records(
    config: KnapsackGenerationConfig,
) -> Iterator[dict[str, Any]]:
    for index in iter_instance_indices(config.start_index, config.num_instances):
        seed = instance_seed(config.seed, index)
        weights, values, capacity = generate_knapsack_instance(
            config.num_items,
            config.min_weight,
            config.max_weight,
            config.min_value,
            config.max_value,
            config.capacity_ratio,
            seed,
        )
        record: dict[str, Any] = {
            "problem": "knapsack",
            "category": "partial_subset",
            "index": index,
            "seed": seed,
            "num_items": config.num_items,
            "weights": weights.tolist(),
            "values": values.tolist(),
            "capacity": capacity,
            "capacity_ratio": config.capacity_ratio,
        }
        solution = solve_dynamic_programming(weights, values, capacity)
        record["solutions"] = {solution.algorithm: solution.to_record()}
        yield record


def generate_knapsack_dataset(config: KnapsackGenerationConfig) -> int:
    return write_jsonl(iter_knapsack_records(config), config.output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic 0/1 Knapsack JSONL data"
    )
    parser.add_argument("--num-instances", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-items", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output JSONL path (default: <prefix>_<split>_<num-instances>.jsonl under the local data root).",
    )
    parser.add_argument("--min-weight", type=int, default=1)
    parser.add_argument("--max-weight", type=int, default=100)
    parser.add_argument("--min-value", type=int, default=1)
    parser.add_argument("--max-value", type=int, default=100)
    parser.add_argument("--capacity-ratio", type=float, default=0.5)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = KnapsackGenerationConfig(
        num_instances=args.num_instances,
        start_index=args.start_index,
        num_items=args.num_items,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        min_value=args.min_value,
        max_value=args.max_value,
        capacity_ratio=args.capacity_ratio,
        seed=args.seed,
        output_path=resolve_output_path(
            "knapsack",
            seed=args.seed,
            num_instances=args.num_instances,
            output_path=args.output_path,
        ),
    )
    written = generate_knapsack_dataset(config)
    print(f"Wrote {written} Knapsack instances to {config.output_path}")


if __name__ == "__main__":
    main()
