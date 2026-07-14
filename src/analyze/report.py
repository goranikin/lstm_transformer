"""Write machine-readable tables and a concise human-readable report."""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import csv
import json

from src.analyze.metadata import DECODER_DESCRIPTION
from src.analyze.records import ArchitectureRecord, GapRecord, PairwiseRecord
from src.constants import DECODER_KINDS, PROBLEM_NAMES


def write_outputs(
    output_dir: Path,
    *,
    export_dir: Path,
    architectures: list[ArchitectureRecord],
    final_gaps: list[GapRecord],
    trajectories: list[GapRecord],
    pairwise: list[PairwiseRecord],
    decoder_summary: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "architectures.csv", architectures)
    _write_csv(output_dir / "final_gaps.csv", final_gaps)
    _write_csv(output_dir / "gap_trajectories.csv", trajectories)
    _write_csv(output_dir / "pairwise_decoder_comparisons.csv", pairwise)
    _write_csv(output_dir / "decoder_summary.csv", decoder_summary)

    payload = {
        "source_export": str(export_dir),
        "methodology": {
            "primary_comparison_metric": "aggregate_gap_pct",
            "definition": (
                "100 * direction-adjusted(decoder_objective - label_objective) "
                "/ abs(label_objective), using aggregate mean objectives"
            ),
            "lower_is_better": True,
            "logged_gap_pct_retained_for_audit": True,
        },
        "architectures": [_plain(record) for record in architectures],
        "final_gaps": [_plain(record) for record in final_gaps],
        "pairwise_decoder_comparisons": [_plain(record) for record in pairwise],
        "decoder_summary": decoder_summary,
    }
    _write_json(output_dir / "analysis.json", payload)
    report = _markdown_report(
        export_dir=export_dir,
        architectures=architectures,
        final_gaps=final_gaps,
        pairwise=pairwise,
        decoder_summary=decoder_summary,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def _markdown_report(
    *,
    export_dir: Path,
    architectures: list[ArchitectureRecord],
    final_gaps: list[GapRecord],
    pairwise: list[PairwiseRecord],
    decoder_summary: list[dict[str, Any]],
) -> str:
    lines = [
        "# Decoder gap analysis",
        "",
        f"Source export: `{export_dir}`",
        "",
        "## Method",
        "",
        "The solver label is the reference objective and the decoder output is the "
        "greedy test objective. Lower gap is better for every problem. For "
        "minimization problems the signed gap is `decoder - label`; for maximization "
        "problems it is `label - decoder`.",
        "",
        "Cross-category comparisons use `aggregate_gap_pct`, computed from the two "
        "aggregate mean objectives. The W&B `logged_gap_pct` is also retained in the "
        "CSV, but it is a mean of per-instance percentages and can become unstable "
        "when an individual label objective is zero.",
        "",
        "A smaller pairwise gap difference means decoder A is closer to the solver "
        "labels than decoder B for that category. This is evidence about relative "
        "problem fit, not proof of topology understanding: training schedules, repair "
        "logic, label quality, and decoder output semantics are confounders.",
        "",
        "## Architecture inventory",
        "",
    ]
    lines.extend(_architecture_markdown(architectures))
    lines.extend(["", "Decoder mechanisms:", ""])
    for decoder in DECODER_KINDS:
        lines.append(f"- `{decoder}`: {DECODER_DESCRIPTION[decoder]}")

    lines.extend(["", "## Final solver-label gaps by problem", ""])
    lines.extend(_gap_matrix(final_gaps))
    lines.extend(
        [
            "",
            "Values are percentage points relative to the mean solver-label objective; "
            "lower is better. `*` marks a row with a quality flag.",
            "",
            "## Decoder profile across categories",
            "",
        ]
    )
    lines.extend(_decoder_summary_markdown(decoder_summary))
    lines.extend(["", "## Pairwise cross-category profile", ""])
    lines.extend(_pairwise_markdown(pairwise))

    flagged = [record for record in final_gaps if record.quality_flags]
    lines.extend(["", "## Data-quality findings", ""])
    if not flagged:
        lines.append("No final rows triggered the built-in quality checks.")
    else:
        lines.extend(
            [
                "| Problem | Decoder | Aggregate gap % | Logged gap % | Flags |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for record in flagged:
            lines.append(
                "| "
                + " | ".join(
                    [
                        record.problem,
                        record.decoder,
                        _format_number(record.aggregate_gap_pct),
                        _format_number(record.logged_gap_pct),
                        ", ".join(record.quality_flags),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Detailed, auditable rows are in `final_gaps.csv`, "
            "`pairwise_decoder_comparisons.csv`, `gap_trajectories.csv`, and "
            "`architectures.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def _architecture_markdown(records: list[ArchitectureRecord]) -> list[str]:
    by_decoder: dict[str, list[ArchitectureRecord]] = defaultdict(list)
    for record in records:
        by_decoder[record.decoder].append(record)
    lines = [
        "All exported runs use the attention encoder with three Transformer layers, "
        "eight heads, ReLU feed-forward blocks, pre-normalization, mean pooling, "
        "`dropout=0`, `context_dim=4`, and `tanh_clip=10` where applicable.",
        "",
        "| Decoder | Encoder layers | Transformer decoder layers | d_model | d_ff | Encoder params | Decoder params | Total params |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for decoder in DECODER_KINDS:
        group = by_decoder.get(decoder, [])
        if not group:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    decoder,
                    _range(record.num_layers for record in group),
                    (
                        _range(
                            record.transformer_decoder_layers for record in group
                        )
                        if decoder == "transformer_pointer"
                        else "NA"
                    ),
                    _range(record.d_model for record in group),
                    _range(record.d_ff for record in group),
                    _range(record.encoder_parameters for record in group),
                    _range(record.decoder_parameters for record in group),
                    _range(record.total_parameters for record in group),
                ]
            )
            + " |"
        )
    return lines


def _gap_matrix(records: list[GapRecord]) -> list[str]:
    lookup = {(record.problem, record.decoder): record for record in records}
    lines = [
        "| Problem | " + " | ".join(DECODER_KINDS) + " |",
        "| --- | " + " | ".join("---:" for _ in DECODER_KINDS) + " |",
    ]
    for problem in PROBLEM_NAMES:
        values = []
        for decoder in DECODER_KINDS:
            record = lookup.get((problem, decoder))
            if record is None:
                values.append("NA")
                continue
            marker = "*" if record.quality_flags else ""
            values.append(f"{_format_number(record.aggregate_gap_pct)}{marker}")
        lines.append(f"| {problem} | " + " | ".join(values) + " |")
    return lines


def _decoder_summary_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Decoder | Mean gap % | Median gap % | Mean rank | Wins | Sequential mean % | Subset mean % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["decoder"]),
                    _format_number(row["mean_aggregate_gap_pct"]),
                    _format_number(row["median_aggregate_gap_pct"]),
                    _format_number(row["mean_decoder_rank"]),
                    str(row["wins"]),
                    _format_number(row["sequential_mean_gap_pct"]),
                    _format_number(row["subset_mean_gap_pct"]),
                ]
            )
            + " |"
        )
    return lines


def _pairwise_markdown(records: list[PairwiseRecord]) -> list[str]:
    groups: dict[tuple[str, str], list[PairwiseRecord]] = defaultdict(list)
    for record in records:
        groups[(record.decoder_a, record.decoder_b)].append(record)
    lines = [
        "The detail CSV contains one row per problem and decoder pair. Positive "
        "`A-B` means decoder B is better; negative means decoder A is better.",
        "",
        "| Pair | A wins | B wins | Ties | Largest A advantage | Largest B advantage |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for (decoder_a, decoder_b), group in groups.items():
        a_wins = [row for row in group if row.winner == decoder_a]
        b_wins = [row for row in group if row.winner == decoder_b]
        ties = [row for row in group if row.winner == "tie"]
        strongest_a = max(a_wins, key=lambda row: row.winner_advantage_pp, default=None)
        strongest_b = max(b_wins, key=lambda row: row.winner_advantage_pp, default=None)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{decoder_a} vs {decoder_b}",
                    str(len(a_wins)),
                    str(len(b_wins)),
                    str(len(ties)),
                    _pair_strength(strongest_a),
                    _pair_strength(strongest_b),
                ]
            )
            + " |"
        )
    return lines


def _pair_strength(record: PairwiseRecord | None) -> str:
    if record is None:
        return "-"
    return f"{record.problem} ({record.winner_advantage_pp:.3f} pp)"


def _write_csv(path: Path, rows: Sequence[Any]) -> None:
    plain_rows = [_plain(row) for row in rows]
    if not plain_rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(plain_rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(plain_rows)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _plain(row: Any) -> dict[str, Any]:
    if hasattr(row, "to_dict"):
        return row.to_dict()
    if is_dataclass(row):
        return asdict(row)
    if isinstance(row, Mapping):
        return dict(row)
    raise TypeError(f"Cannot serialize row of type {type(row).__name__}")


def _range(values: Iterable[int]) -> str:
    unique = sorted(set(values))
    if len(unique) == 1:
        return f"{unique[0]:,}"
    return f"{unique[0]:,}-{unique[-1]:,}"


def _format_number(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"
