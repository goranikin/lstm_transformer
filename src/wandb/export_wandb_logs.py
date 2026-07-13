"""Export the 28 supervised W&B runs and their complete training histories.

Run with:
    uv run python -m src.wandb.export_wandb_logs
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.wandb.config import (
    DEFAULT_CUTOFF,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROJECT,
    DEFAULT_SEED,
    DEFAULT_TAG,
    ExportSettings,
)
from src.wandb.exporter import ManifestEntry, export_runs
from src.wandb.selection import fetch_runs, validate_run_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export complete W&B histories for the supervised 7x4 matrix."
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--encoder", default="attention")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--download-log-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download synchronized output.log and other .log files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    settings = ExportSettings(
        project=args.project,
        cutoff=args.cutoff,
        tag=args.tag,
        seed=args.seed,
        encoder=args.encoder,
        output_dir=args.output_dir,
        download_log_files=args.download_log_files,
    )

    import wandb

    api = wandb.Api(timeout=args.timeout)
    runs = fetch_runs(api, settings)
    print(f"Matched {len(runs)} W&B runs", flush=True)

    validation = validate_run_matrix(runs, settings)
    if not validation.is_valid:
        raise SystemExit(validation.error_message())

    output_dir = export_runs(runs, settings, progress=_print_progress)
    print(f"Export completed: {output_dir}", flush=True)
    return output_dir


def _print_progress(index: int, total: int, entry: ManifestEntry) -> None:
    print(
        f"[{index:02d}/{total:02d}] {entry.problem}/{entry.decoder}: "
        f"{entry.history_rows} history rows",
        flush=True,
    )


if __name__ == "__main__":
    main()
