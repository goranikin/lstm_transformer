"""Command-line interface for experiment analysis."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.analyze.pipeline import analyze_export


DEFAULT_EXPORT_DIR = Path("outputs/wandb_export/supervised_seed1234")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/supervised_seed1234")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze solver-label vs decoder-output gaps in an exported W&B "
            "experiment matrix."
        )
    )
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--evaluation-split",
        choices=("test", "val", "train/sl/eval"),
        default="test",
        help="Split used for the final cross-problem decoder comparison.",
    )
    parser.add_argument(
        "--trajectory-split",
        choices=("val", "train/sl/eval"),
        default="val",
        help="Epoch-level split written to gap_trajectories.csv.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    result = analyze_export(
        args.export_dir,
        args.output_dir,
        evaluation_split=args.evaluation_split,
        trajectory_split=args.trajectory_split,
    )
    print(
        f"Analyzed {result.run_count} runs and {result.final_gap_count} final gaps; "
        f"wrote {result.output_dir} ({result.quality_flag_count} flagged rows)."
    )
    return result.output_dir
