"""Report JSONL generation progress under ~/local_db/lstm_transformer/<problem>/."""

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from src.paths import DATA_ROOT

TRAIN_TOTAL = 64_000
TRAIN_CHUNK = 8_000
TRAIN_PARTS = 8

VAL_TEST_TOTAL = 10_000
VAL_TEST_CHUNK = 5_000
VAL_TEST_PARTS = 2

PART_RE = re.compile(r"^(?P<base>.+)\.part(?P<part>\d{2})\.jsonl$")


@dataclass(frozen=True)
class SplitSpec:
    kind: str
    total: int
    chunk_size: int
    num_parts: int


@dataclass(frozen=True)
class ChunkStatus:
    part: int
    path: Path | None
    count: int
    expected: int

    @property
    def label(self) -> str:
        return f"part{self.part:02d}"

    @property
    def fraction(self) -> float:
        if self.expected == 0:
            return 0.0
        return self.count / self.expected


@dataclass(frozen=True)
class DatasetStatus:
    problem: str
    name: str
    spec: SplitSpec
    chunks: tuple[ChunkStatus, ...]
    is_parted: bool

    @property
    def done(self) -> int:
        return sum(chunk.count for chunk in self.chunks)

    @property
    def expected(self) -> int:
        return self.spec.total

    @property
    def fraction(self) -> float:
        if self.expected == 0:
            return 0.0
        return self.done / self.expected


def split_spec(name: str) -> SplitSpec:
    if "_test_" in name:
        return SplitSpec("test", VAL_TEST_TOTAL, VAL_TEST_CHUNK, VAL_TEST_PARTS)
    if "_val_" in name:
        return SplitSpec("val", VAL_TEST_TOTAL, VAL_TEST_CHUNK, VAL_TEST_PARTS)
    return SplitSpec("train", TRAIN_TOTAL, TRAIN_CHUNK, TRAIN_PARTS)


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
    return count


def format_bar(fraction: float, width: int = 20) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return f"[{'=' * filled}{'>' if filled < width else ''}{' ' * (width - filled - (1 if filled < width else 0))}]"


def format_count(done: int, expected: int) -> str:
    return f"{done:>6}/{expected:<6}"


def format_percent(fraction: float) -> str:
    return f"{fraction * 100:5.1f}%"


def discover_datasets(
    data_dir: Path,
    *,
    include_smoke: bool,
) -> list[DatasetStatus]:
    part_groups: dict[tuple[str, str], dict[int, Path]] = {}
    whole_files: list[tuple[str, Path]] = []

    for problem_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for path in sorted(problem_dir.glob("*.jsonl")):
            if not include_smoke and "_smoke_" in path.name:
                continue

            match = PART_RE.match(path.name)
            if match is not None:
                base = match.group("base")
                part = int(match.group("part"))
                key = (problem_dir.name, base)
                part_groups.setdefault(key, {})[part] = path
                continue

            whole_files.append((problem_dir.name, path))

    datasets: list[DatasetStatus] = []

    for (problem, base), parts in sorted(part_groups.items()):
        spec = split_spec(base)
        chunks: list[ChunkStatus] = []
        for part in range(spec.num_parts):
            path = parts.get(part)
            if path is None:
                chunks.append(
                    ChunkStatus(part=part, path=None, count=0, expected=spec.chunk_size)
                )
                continue
            chunks.append(
                ChunkStatus(
                    part=part,
                    path=path,
                    count=count_lines(path),
                    expected=spec.chunk_size,
                )
            )
        datasets.append(
            DatasetStatus(
                problem=problem,
                name=base,
                spec=spec,
                chunks=tuple(chunks),
                is_parted=True,
            )
        )

    parted_names = {base for _, base in part_groups}
    for problem, path in whole_files:
        base = path.stem
        if base in parted_names:
            continue
        spec = split_spec(base)
        datasets.append(
            DatasetStatus(
                problem=problem,
                name=base,
                spec=spec,
                chunks=(
                    ChunkStatus(
                        part=0,
                        path=path,
                        count=count_lines(path),
                        expected=spec.total,
                    ),
                ),
                is_parted=False,
            )
        )

    datasets.sort(key=lambda item: (item.problem, item.spec.kind, item.name))
    return datasets


def render_report(datasets: list[DatasetStatus], *, verbose: bool) -> str:
    if not datasets:
        return "No JSONL datasets found."

    lines: list[str] = []
    current_problem: str | None = None

    for dataset in datasets:
        if dataset.problem != current_problem:
            current_problem = dataset.problem
            lines.append("")
            lines.append(f"== {dataset.problem} ==")

        lines.append(
            f"{dataset.name} ({dataset.spec.kind}, {dataset.expected:,} total) "
            f"{format_count(dataset.done, dataset.expected)} {format_percent(dataset.fraction)}"
        )

        if dataset.is_parted:
            for chunk in dataset.chunks:
                status = "done" if chunk.count >= chunk.expected else "...."
                if chunk.path is None:
                    status = "miss"
                elif chunk.count == 0 and chunk.path is not None:
                    status = "run?"

                size = ""
                if verbose and chunk.path is not None:
                    size = f"  {chunk.path.stat().st_size / 1024:.0f}K"

                lines.append(
                    f"  {chunk.label}  {format_count(chunk.count, chunk.expected)}  "
                    f"{format_bar(chunk.fraction)} {format_percent(chunk.fraction)}  "
                    f"[{status}]{size}"
                )
        elif verbose and dataset.chunks[0].path is not None:
            path = dataset.chunks[0].path
            lines.append(f"  file  {path.name}  ({path.stat().st_size / 1024:.0f}K)")

    grand_done = sum(dataset.done for dataset in datasets)
    grand_expected = sum(dataset.expected for dataset in datasets)
    lines.append("")
    lines.append(
        f"Overall: {format_count(grand_done, grand_expected)} "
        f"{format_percent(grand_done / grand_expected if grand_expected else 0.0)}"
    )
    return "\n".join(lines).lstrip("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect JSONL data-generation progress under the local data root."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_ROOT,
        help=f"Root data directory (default: {DATA_ROOT})",
    )
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SEC",
        nargs="?",
        const=5.0,
        help="Refresh every SEC seconds (default: 5)",
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include *_smoke_* datasets",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show file sizes",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir: Path = args.data_dir

    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    def run_once() -> str:
        datasets = discover_datasets(data_dir, include_smoke=args.include_smoke)
        return render_report(datasets, verbose=args.verbose)

    if args.watch is None:
        print(run_once())
        return

    try:
        while True:
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(run_once())
            print(f"\nRefreshing every {args.watch:g}s. Press Ctrl+C to stop.")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
