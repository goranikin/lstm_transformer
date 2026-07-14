"""Gap extraction, normalization, ranking, and decoder comparisons."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from itertools import combinations
from statistics import mean, median
from typing import Any
import math

from src.analyze.metadata import DECODER_FAMILY, problem_metadata
from src.analyze.records import GapRecord, PairwiseRecord
from src.constants import DECODER_KINDS, PROBLEM_NAMES


def final_gap_records(
    history: Iterable[Mapping[str, Any]],
    *,
    split: str = "test",
    fallback_split: str | None = "val",
) -> list[GapRecord]:
    rows = list(history)
    selected = _latest_rows(rows, split)
    if fallback_split is not None:
        fallback = _latest_rows(rows, fallback_split)
        for run_id, row in fallback.items():
            selected.setdefault(run_id, row)

    records = [
        _gap_record(row, split if _has_metrics(row, split) else str(fallback_split))
        for row in selected.values()
    ]
    return rank_decoders(records)


def trajectory_gap_records(
    history: Iterable[Mapping[str, Any]],
    *,
    split: str = "val",
) -> list[GapRecord]:
    selected: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in history:
        if not _has_metrics(row, split) or row.get("epoch") is None:
            continue
        key = (str(row["_run_id"]), int(row["epoch"]))
        if key not in selected or _row_order(row) > _row_order(selected[key]):
            selected[key] = row
    records = [_gap_record(row, split) for row in selected.values()]
    return sorted(
        records,
        key=lambda record: (record.problem, record.decoder, record.epoch or -1),
    )


def rank_decoders(records: list[GapRecord]) -> list[GapRecord]:
    by_problem: dict[str, list[GapRecord]] = defaultdict(list)
    for record in records:
        by_problem[record.problem].append(record)

    ranked: list[GapRecord] = []
    for problem_records in by_problem.values():
        comparable = [
            record
            for record in problem_records
            if record.aggregate_gap_pct is not None
            and math.isfinite(record.aggregate_gap_pct)
        ]
        comparable.sort(key=_required_gap)
        best = _required_gap(comparable[0]) if comparable else 0.0
        ranks: dict[str, float] = {}
        index = 0
        while index < len(comparable):
            end = index + 1
            value = _required_gap(comparable[index])
            while (
                end < len(comparable)
                and math.isclose(
                    _required_gap(comparable[end]),
                    value,
                    abs_tol=1e-9,
                )
            ):
                end += 1
            average_rank = ((index + 1) + end) / 2.0
            for record in comparable[index:end]:
                ranks[record.run_id] = average_rank
            index = end

        for record in problem_records:
            gap = record.aggregate_gap_pct
            if record.run_id in ranks and gap is not None:
                ranked.append(record.ranked(ranks[record.run_id], float(gap) - best))
            else:
                ranked.append(record)
    return sorted(ranked, key=lambda record: (record.problem, record.decoder))


def pairwise_comparisons(records: list[GapRecord]) -> list[PairwiseRecord]:
    by_problem: dict[str, dict[str, GapRecord]] = defaultdict(dict)
    for record in records:
        by_problem[record.problem][record.decoder] = record

    comparisons: list[PairwiseRecord] = []
    decoder_order = [decoder for decoder in DECODER_KINDS]
    for problem in PROBLEM_NAMES:
        problem_records = by_problem.get(problem, {})
        for decoder_a, decoder_b in combinations(decoder_order, 2):
            record_a = problem_records.get(decoder_a)
            record_b = problem_records.get(decoder_b)
            if (
                record_a is None
                or record_b is None
                or record_a.aggregate_gap_pct is None
                or record_b.aggregate_gap_pct is None
            ):
                continue
            gap_a = float(record_a.aggregate_gap_pct)
            gap_b = float(record_b.aggregate_gap_pct)
            delta = gap_a - gap_b
            if math.isclose(delta, 0.0, abs_tol=1e-9):
                winner = "tie"
            else:
                winner = decoder_a if delta < 0 else decoder_b
            comparisons.append(
                PairwiseRecord(
                    problem=problem,
                    problem_family=record_a.problem_family,
                    topology=record_a.topology,
                    decoder_a=decoder_a,
                    decoder_b=decoder_b,
                    decoder_a_gap_pct=gap_a,
                    decoder_b_gap_pct=gap_b,
                    gap_delta_a_minus_b_pp=delta,
                    winner=winner,
                    winner_advantage_pp=abs(delta),
                )
            )
    return comparisons


def decoder_summaries(records: list[GapRecord]) -> list[dict[str, Any]]:
    by_decoder: dict[str, list[GapRecord]] = defaultdict(list)
    for record in records:
        if record.aggregate_gap_pct is not None:
            by_decoder[record.decoder].append(record)

    summaries = []
    for decoder in DECODER_KINDS:
        decoder_records = by_decoder.get(decoder, [])
        if not decoder_records:
            continue
        values = [_required_gap(record) for record in decoder_records]
        ranks = [
            float(record.decoder_rank)
            for record in decoder_records
            if record.decoder_rank is not None
        ]
        sequential = [
            _required_gap(record)
            for record in decoder_records
            if record.problem_family == "sequential_construction"
        ]
        subset = [
            _required_gap(record)
            for record in decoder_records
            if record.problem_family == "subset_selection"
        ]
        best = min(decoder_records, key=_required_gap)
        worst = max(decoder_records, key=_required_gap)
        flagged = [record.problem for record in decoder_records if record.quality_flags]
        summaries.append(
            {
                "decoder": decoder,
                "decoder_family": DECODER_FAMILY.get(decoder, decoder),
                "problem_count": len(values),
                "mean_aggregate_gap_pct": mean(values),
                "median_aggregate_gap_pct": median(values),
                "mean_decoder_rank": mean(ranks) if ranks else None,
                "wins": sum(rank == 1.0 for rank in ranks),
                "sequential_mean_gap_pct": mean(sequential) if sequential else None,
                "subset_mean_gap_pct": mean(subset) if subset else None,
                "cross_category_range_pp": max(values) - min(values),
                "best_problem": best.problem,
                "best_problem_gap_pct": best.aggregate_gap_pct,
                "worst_problem": worst.problem,
                "worst_problem_gap_pct": worst.aggregate_gap_pct,
                "quality_flagged_problems": ";".join(flagged),
            }
        )
    return summaries


def _latest_rows(
    history: Iterable[Mapping[str, Any]], split: str
) -> dict[str, Mapping[str, Any]]:
    selected: dict[str, Mapping[str, Any]] = {}
    for row in history:
        if not _has_metrics(row, split):
            continue
        run_id = str(row["_run_id"])
        if run_id not in selected or _row_order(row) > _row_order(selected[run_id]):
            selected[run_id] = row
    return selected


def _has_metrics(row: Mapping[str, Any], split: str) -> bool:
    return (
        row.get(f"{split}/objective") is not None
        and row.get(f"{split}/target_objective") is not None
    )


def _row_order(row: Mapping[str, Any]) -> tuple[float, float]:
    return (float(row.get("_step", -1)), float(row.get("_timestamp", -1)))


def _gap_record(row: Mapping[str, Any], split: str) -> GapRecord:
    problem = str(row["_problem"])
    decoder = str(row["_decoder"])
    metadata = problem_metadata(problem)
    objective = float(row[f"{split}/objective"])
    target = float(row[f"{split}/target_objective"])
    absolute_gap = objective - target if metadata.objective_sense == "min" else target - objective
    aggregate_gap_pct = (
        None if math.isclose(target, 0.0, abs_tol=1e-12) else 100.0 * absolute_gap / abs(target)
    )
    logged_gap = _optional_float(row.get(f"{split}/optimal_gap"))
    logged_gap_pct = _optional_float(row.get(f"{split}/optimal_gap_pct"))
    feasibility = float(row.get(f"{split}/feasibility_rate", float("nan")))
    flags = _quality_flags(
        absolute_gap=absolute_gap,
        aggregate_gap_pct=aggregate_gap_pct,
        logged_gap=logged_gap,
        logged_gap_pct=logged_gap_pct,
        feasibility_rate=feasibility,
    )
    return GapRecord(
        run_id=str(row["_run_id"]),
        problem=problem,
        problem_family=metadata.family,
        topology=metadata.topology,
        solver=metadata.solver,
        objective_sense=metadata.objective_sense,
        encoder=str(row["_encoder"]),
        decoder=decoder,
        decoder_family=DECODER_FAMILY.get(decoder, decoder),
        split=split,
        epoch=_optional_int(row.get("epoch")),
        step=_optional_int(row.get("_step")),
        count=int(row.get(f"{split}/count", 0)),
        label_objective=target,
        decoder_objective=objective,
        absolute_gap=absolute_gap,
        aggregate_gap_pct=aggregate_gap_pct,
        logged_gap=logged_gap,
        logged_gap_pct=logged_gap_pct,
        feasibility_rate=feasibility,
        inference_time_sec=float(row.get(f"{split}/inference_time_sec", 0.0)),
        quality_flags=flags,
    )


def _quality_flags(
    *,
    absolute_gap: float,
    aggregate_gap_pct: float | None,
    logged_gap: float | None,
    logged_gap_pct: float | None,
    feasibility_rate: float,
) -> tuple[str, ...]:
    flags = []
    if logged_gap is not None and not math.isclose(
        absolute_gap, logged_gap, rel_tol=1e-5, abs_tol=1e-5
    ):
        flags.append("logged_gap_mismatch")
    if aggregate_gap_pct is None:
        flags.append("zero_mean_label_objective")
    elif aggregate_gap_pct < -1e-6:
        flags.append("decoder_better_than_label")
    if logged_gap_pct is not None and (
        not math.isfinite(logged_gap_pct)
        or abs(logged_gap_pct) > 1_000_000
        or (
            aggregate_gap_pct is not None
            and abs(logged_gap_pct - aggregate_gap_pct) > 1_000
        )
    ):
        flags.append("unstable_logged_gap_pct")
    if not math.isfinite(feasibility_rate) or feasibility_rate < 0.999999:
        flags.append("infeasible_decoder_outputs")
    return tuple(flags)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _required_gap(record: GapRecord) -> float:
    if record.aggregate_gap_pct is None:
        raise ValueError(f"Run {record.run_id} has no comparable aggregate gap")
    return float(record.aggregate_gap_pct)
