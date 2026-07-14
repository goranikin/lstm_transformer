"""Typed records shared by the analysis pipeline."""

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class ArchitectureRecord:
    run_id: str
    problem: str
    problem_family: str
    topology: str
    solver: str
    objective_sense: str
    encoder: str
    decoder: str
    decoder_family: str
    input_dim: int
    context_dim: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    transformer_decoder_layers: int
    dropout: float
    tanh_clip: float
    encoder_parameters: int
    decoder_parameters: int
    total_parameters: int
    trainable_parameters: int
    target_parameters: int | None
    parameter_delta: int | None
    parameter_delta_pct: float | None
    epochs: int
    steps_per_epoch: int | None
    batch_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GapRecord:
    run_id: str
    problem: str
    problem_family: str
    topology: str
    solver: str
    objective_sense: str
    encoder: str
    decoder: str
    decoder_family: str
    split: str
    epoch: int | None
    step: int | None
    count: int
    label_objective: float
    decoder_objective: float
    absolute_gap: float
    aggregate_gap_pct: float | None
    logged_gap: float | None
    logged_gap_pct: float | None
    feasibility_rate: float
    inference_time_sec: float
    quality_flags: tuple[str, ...]
    decoder_rank: float | None = None
    regret_to_best_decoder_pp: float | None = None

    def ranked(self, rank: float, regret: float) -> "GapRecord":
        return replace(
            self,
            decoder_rank=rank,
            regret_to_best_decoder_pp=regret,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quality_flags"] = ";".join(self.quality_flags)
        return payload


@dataclass(frozen=True)
class PairwiseRecord:
    problem: str
    problem_family: str
    topology: str
    decoder_a: str
    decoder_b: str
    decoder_a_gap_pct: float
    decoder_b_gap_pct: float
    gap_delta_a_minus_b_pp: float
    winner: str
    winner_advantage_pp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
