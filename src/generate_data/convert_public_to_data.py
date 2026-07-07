import argparse
import json
from pathlib import Path
from typing import Literal, TextIO

from tqdm import tqdm

from src.constants import PROBLEM_NAMES, ProblemName
from src.generate_data.common import instance_seed
from src.generate_data.ml4co_to_jsonl import (
    SPLIT_SEEDS,
    TARGET_COUNTS,
    build_assignment,
    convert_jsonl_file,
    count_jsonl_lines,
    normalize_record,
    resolve_problem_name,
)
from src.paths import DATA_ROOT, LOCAL_DB_ROOT, PUBLIC_DATA_100K_ROOT

Layout = Literal["public", "data"]

PROBLEM_NAME_TO_KEY: dict[ProblemName, str] = {
    "tsp": "tsp",
    "cvrp": "cvrp",
    "mis": "mis",
    "max_clique": "mcl",
    "vertex_cover": "mvc",
}


def public_output_path(output_dir: Path, split_name: str) -> Path:
    return output_dir / f"{split_name}.jsonl"


def discover_train_files(input_roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in input_roots:
        files.extend(
            sorted(path for path in root.glob("*/train.jsonl") if path.is_file())
        )
    return files


def problem_key_from_name(problem: ProblemName | None) -> str | None:
    if problem is None:
        return None
    return PROBLEM_NAME_TO_KEY[problem]


def convert_jsonl_file_with_layout(
    input_path: Path,
    *,
    layout: Layout,
    output_dir: Path | None,
    data_root: Path,
    problem: ProblemName | None,
    seed: int,
    target_counts: dict[str, int],
    normalize: bool,
) -> dict[str, int]:
    if layout == "data":
        if not normalize:
            raise ValueError(
                "layout=data requires normalized output; omit --no-normalize."
            )
        return convert_jsonl_file(
            input_path,
            output_root=data_root,
            problem_key=problem_key_from_name(problem),
            seed=seed,
            target_counts=target_counts,
        )

    if output_dir is None:
        raise ValueError("output_dir is required when layout=public")

    return convert_public_layout(
        input_path,
        output_dir=output_dir,
        problem=problem,
        seed=seed,
        target_counts=target_counts,
        normalize=normalize,
    )


def convert_public_layout(
    input_path: Path,
    *,
    output_dir: Path,
    problem: ProblemName | None,
    seed: int,
    target_counts: dict[str, int],
    normalize: bool,
) -> dict[str, int]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    total = count_jsonl_lines(input_path)
    if total == 0:
        raise ValueError(f"No records found in {input_path}")

    split_counts, assignment = build_assignment(
        total,
        seed=seed,
        target_counts=target_counts,
    )

    if total < sum(target_counts.values()):
        requested = ", ".join(
            f"{name}={count:,}" for name, count in target_counts.items()
        )
        actual = ", ".join(f"{name}={count:,}" for name, count in split_counts.items())
        print(
            f"Warning: only {total:,} instances available; "
            f"using proportional split ({actual}) instead of ({requested})."
        )

    handles: dict[str, TextIO] = {}
    output_paths: dict[str, Path] = {}

    with input_path.open("r", encoding="utf-8") as probe:
        first_line = next(line for line in probe if line.strip())
    resolved_problem = resolve_problem_name(
        json.loads(first_line),
        directory_name=input_path.parent.name,
        problem_key=problem_key_from_name(problem),
    )

    try:
        for split_name, count in split_counts.items():
            if count == 0:
                continue
            out_path = public_output_path(output_dir, split_name)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            output_paths[split_name] = out_path
            handles[split_name] = out_path.open("w", encoding="utf-8")

        with input_path.open("r", encoding="utf-8") as handle:
            record_index = 0
            for line in tqdm(handle, total=total, desc=resolved_problem):
                stripped = line.strip()
                if not stripped:
                    continue
                if record_index >= len(assignment):
                    break

                split_name, split_index = assignment[record_index]
                record_index += 1
                record = json.loads(stripped)
                if normalize:
                    record = normalize_record(
                        record,
                        problem=resolved_problem,
                        split_name=split_name,
                        split_index=split_index,
                    )
                else:
                    record = dict(record)
                    record["split"] = split_name
                    record["index"] = split_index
                    record["seed"] = instance_seed(SPLIT_SEEDS[split_name], split_index)

                handles[split_name].write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
    finally:
        for handle in handles.values():
            handle.close()

    for split_name, count in split_counts.items():
        if count:
            print(f"Wrote {count:,} rows to {output_paths[split_name]}")

    return split_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stream-convert monolithic ML4CO public JSONL files into train/val/test "
            "splits compatible with src/experiments/run.py.\n\n"
            "Designed for very large inputs (10-20GB): the file is scanned once to "
            "count lines, then streamed line-by-line without loading the whole file "
            "into memory.\n\n"
            "Example:\n"
            "  uv run python -m src.generate_data.convert_public_to_data \\\n"
            f"    --input-root {LOCAL_DB_ROOT / 'data_public_84k'} \\\n"
            f"    --input-root {PUBLIC_DATA_100K_ROOT} \\\n"
            f"    --data-root {DATA_ROOT}\n\n"
            "For ML4CO downloads, pass data.target_algorithm=ml4co when training."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, help="Path to a single JSONL file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for train.jsonl, val.jsonl, and test.jsonl when layout=public.",
    )
    parser.add_argument(
        "--input-root",
        action="append",
        type=Path,
        help="Convert every */train.jsonl found under this directory. Repeatable.",
    )
    parser.add_argument(
        "--layout",
        choices=("public", "data"),
        default="data",
        help=(
            "public writes train.jsonl/val.jsonl/test.jsonl next to the source; "
            "data writes <problem>/ files under the local data root, e.g. tsp/."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help=f"Data root used when layout=data (default: {DATA_ROOT}).",
    )
    parser.add_argument(
        "--problem",
        choices=list(PROBLEM_NAMES),
        help="Override problem inference from the parent directory name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Shuffle seed used before assigning train/val/test rows.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Keep the original ML4CO public record fields instead of the TSP-style schema.",
    )
    args = parser.parse_args()

    if args.input_root:
        train_files = discover_train_files(args.input_root)
        if not train_files:
            roots = ", ".join(str(root) for root in args.input_root)
            raise FileNotFoundError(f"No */train.jsonl files found under: {roots}")
        for train_file in train_files:
            print(f"\n=== {train_file} ===")
            convert_jsonl_file_with_layout(
                train_file,
                layout=args.layout,
                output_dir=train_file.parent if args.layout == "public" else None,
                data_root=args.data_root,
                problem=args.problem,
                seed=args.seed,
                target_counts=TARGET_COUNTS,
                normalize=not args.no_normalize,
            )
        return

    if args.input is None:
        parser.error("Provide --input or at least one --input-root.")

    convert_jsonl_file_with_layout(
        args.input,
        layout=args.layout,
        output_dir=args.output_dir,
        data_root=args.data_root,
        problem=args.problem,
        seed=args.seed,
        target_counts=TARGET_COUNTS,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
