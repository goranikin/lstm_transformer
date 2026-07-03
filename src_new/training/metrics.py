from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src_new.core import SolutionOutput
from src_new.problems import Problem


@dataclass(frozen=True)
class BatchMetrics:
    count: int
    objective: float
    feasibility_rate: float
    inference_time_sec: float
    gap: float | None = None
    gap_pct: float | None = None
    target_objective: float | None = None


@dataclass(frozen=True)
class EvaluationMetrics:
    count: int
    objective: float
    feasibility_rate: float
    inference_time_sec: float
    gap: float | None = None
    gap_pct: float | None = None
    target_objective: float | None = None

    def to_dict(self, prefix: str = "") -> dict[str, float | int]:
        key = f"{prefix}/" if prefix else ""
        metrics: dict[str, float | int] = {
            f"{key}count": self.count,
            f"{key}objective": self.objective,
            f"{key}feasibility_rate": self.feasibility_rate,
            f"{key}inference_time_sec": self.inference_time_sec,
        }
        if self.gap is not None:
            metrics[f"{key}gap"] = self.gap
        if self.gap_pct is not None:
            metrics[f"{key}gap_pct"] = self.gap_pct
        if self.target_objective is not None:
            metrics[f"{key}target_objective"] = self.target_objective
        return metrics


def batch_metrics(
    problem: Problem,
    batch: dict[str, Any],
    output: SolutionOutput,
    inference_time_sec: float,
) -> BatchMetrics:
    objective = output.objective.detach()
    feasible = output.feasible.detach().float()
    count = int(objective.numel())
    target = problem.target_value(batch)
    gap = None
    gap_pct = None
    target_mean = None
    if target is not None:
        target = target.to(device=objective.device, dtype=objective.dtype)
        if problem.objective_sense == "min":
            raw_gap = objective - target
        else:
            raw_gap = target - objective
        raw_gap_pct = 100.0 * raw_gap / target.abs().clamp_min(1e-12)
        gap = float(raw_gap.mean().item())
        gap_pct = float(raw_gap_pct.mean().item())
        target_mean = float(target.mean().item())
    return BatchMetrics(
        count=count,
        objective=float(objective.mean().item()),
        feasibility_rate=float(feasible.mean().item()),
        inference_time_sec=inference_time_sec,
        gap=gap,
        gap_pct=gap_pct,
        target_objective=target_mean,
    )


def aggregate_metrics(items: list[BatchMetrics]) -> EvaluationMetrics:
    if not items:
        return EvaluationMetrics(
            count=0,
            objective=0.0,
            feasibility_rate=0.0,
            inference_time_sec=0.0,
        )
    count = sum(item.count for item in items)
    denom = max(count, 1)

    def avg(name: str) -> float:
        return sum(getattr(item, name) * item.count for item in items) / denom

    def optional_avg(name: str) -> float | None:
        present = [item for item in items if getattr(item, name) is not None]
        if not present:
            return None
        present_count = sum(item.count for item in present)
        return (
            sum(float(getattr(item, name)) * item.count for item in present)
            / max(present_count, 1)
        )

    return EvaluationMetrics(
        count=count,
        objective=avg("objective"),
        feasibility_rate=avg("feasibility_rate"),
        inference_time_sec=sum(item.inference_time_sec for item in items),
        gap=optional_avg("gap"),
        gap_pct=optional_avg("gap_pct"),
        target_objective=optional_avg("target_objective"),
    )


def seed_variance(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
    }
