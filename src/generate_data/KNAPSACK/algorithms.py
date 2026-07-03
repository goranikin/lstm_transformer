from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict


class KnapsackSolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    algorithm: str
    items: list[int]
    value: int
    weight: int
    is_exact: bool
    metadata: dict[str, Any] | None = None

    def to_record(self) -> dict:
        record = {
            "algorithm": self.algorithm,
            "is_exact": self.is_exact,
            "items": self.items,
            "value": self.value,
            "weight": self.weight,
        }
        if self.metadata:
            record["metadata"] = self.metadata
        return record


def validate_knapsack_inputs(
    weights: np.ndarray,
    values: np.ndarray,
    capacity: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    weights = np.asarray(weights, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)
    capacity = int(capacity)
    if weights.ndim != 1 or values.ndim != 1:
        raise ValueError("weights and values must be one-dimensional")
    if len(weights) != len(values):
        raise ValueError("weights and values must have the same length")
    if len(weights) == 0:
        raise ValueError("at least one item is required")
    if np.any(weights <= 0):
        raise ValueError("weights must be positive integers")
    if np.any(values < 0):
        raise ValueError("values must be non-negative integers")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    return weights, values, capacity


def solve_dynamic_programming(
    weights: np.ndarray,
    values: np.ndarray,
    capacity: int,
) -> KnapsackSolution:
    weights, values, capacity = validate_knapsack_inputs(weights, values, capacity)
    num_items = len(weights)
    dp = np.zeros((num_items + 1, capacity + 1), dtype=np.int64)
    keep = np.zeros((num_items + 1, capacity + 1), dtype=np.bool_)

    for item in range(1, num_items + 1):
        weight = int(weights[item - 1])
        value = int(values[item - 1])
        for cap in range(capacity + 1):
            skip = dp[item - 1, cap]
            take = -1
            if weight <= cap:
                take = dp[item - 1, cap - weight] + value
            if take > skip:
                dp[item, cap] = take
                keep[item, cap] = True
            else:
                dp[item, cap] = skip

    selected: list[int] = []
    cap = capacity
    for item in range(num_items, 0, -1):
        if keep[item, cap]:
            selected.append(item - 1)
            cap -= int(weights[item - 1])
    selected.reverse()
    solution = _solution(
        "dynamic_programming",
        selected,
        weights,
        values,
        is_exact=True,
        metadata={"capacity": capacity},
    )
    _validate_solution(solution, weights, values, capacity)
    return solution


def _solution(
    algorithm: str,
    items: list[int],
    weights: np.ndarray,
    values: np.ndarray,
    *,
    is_exact: bool,
    metadata: dict[str, Any] | None = None,
) -> KnapsackSolution:
    return KnapsackSolution(
        algorithm=algorithm,
        items=items,
        value=int(values[items].sum()) if items else 0,
        weight=int(weights[items].sum()) if items else 0,
        is_exact=is_exact,
        metadata=metadata,
    )


def _validate_solution(
    solution: KnapsackSolution,
    weights: np.ndarray,
    values: np.ndarray,
    capacity: int,
) -> None:
    if len(solution.items) != len(set(solution.items)):
        raise RuntimeError(f"{solution.algorithm} selected duplicate items")
    if any(item < 0 or item >= len(weights) for item in solution.items):
        raise RuntimeError(f"{solution.algorithm} selected an out-of-range item")
    weight = int(weights[solution.items].sum()) if solution.items else 0
    value = int(values[solution.items].sum()) if solution.items else 0
    if weight > capacity:
        raise RuntimeError(f"{solution.algorithm} exceeded capacity")
    if weight != solution.weight or value != solution.value:
        raise RuntimeError(f"{solution.algorithm} recorded inconsistent totals")
