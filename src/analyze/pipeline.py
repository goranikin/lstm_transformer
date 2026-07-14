"""End-to-end orchestration for an exported experiment matrix."""

from dataclasses import dataclass
from pathlib import Path

from src.analyze.architecture import build_architecture_records
from src.analyze.gaps import (
    decoder_summaries,
    final_gap_records,
    pairwise_comparisons,
    trajectory_gap_records,
)
from src.analyze.loader import (
    iter_history,
    load_manifest,
    load_run_configs,
    validate_export,
)
from src.analyze.report import write_outputs


@dataclass(frozen=True)
class AnalysisResult:
    output_dir: Path
    run_count: int
    final_gap_count: int
    trajectory_count: int
    quality_flag_count: int


def analyze_export(
    export_dir: Path,
    output_dir: Path,
    *,
    evaluation_split: str = "test",
    trajectory_split: str = "val",
) -> AnalysisResult:
    export_dir = export_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    manifest = load_manifest(export_dir)
    configs = load_run_configs(export_dir)
    validate_export(manifest, configs)
    history = list(iter_history(export_dir))

    architectures = build_architecture_records(manifest, configs)
    final_gaps = final_gap_records(
        history,
        split=evaluation_split,
        fallback_split="val" if evaluation_split != "val" else None,
    )
    trajectories = trajectory_gap_records(history, split=trajectory_split)
    pairwise = pairwise_comparisons(final_gaps)
    summaries = decoder_summaries(final_gaps)

    manifest_ids = {str(entry["id"]) for entry in manifest}
    final_ids = {record.run_id for record in final_gaps}
    if final_ids != manifest_ids:
        missing = sorted(manifest_ids - final_ids)
        raise ValueError(f"No final evaluation metrics found for runs: {missing}")

    write_outputs(
        output_dir,
        export_dir=export_dir,
        architectures=architectures,
        final_gaps=final_gaps,
        trajectories=trajectories,
        pairwise=pairwise,
        decoder_summary=summaries,
    )
    return AnalysisResult(
        output_dir=output_dir,
        run_count=len(manifest),
        final_gap_count=len(final_gaps),
        trajectory_count=len(trajectories),
        quality_flag_count=sum(bool(record.quality_flags) for record in final_gaps),
    )
