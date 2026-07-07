import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi, RepoFile
from tqdm import tqdm

from src.generate_data.common import instance_seed
from src.generate_data.ml4co_to_jsonl import (
    PROBLEM_ALIASES,
    TARGET_COUNTS,
    build_parser_registry,
    record_from_parsed,
)
from src.paths import PUBLIC_DATA_84K_ROOT

REPO_ID = "ML4CO/ML4CO-Bench-101-SL"

# TSP is intentionally excluded here.
PROBLEMS = ["mlc"]


def problem_from_path(path: str, dataset_subdir: str) -> str | None:
    normalized = path.replace("\\", "/").lower()
    if not normalized.endswith(".txt"):
        return None

    parts = normalized.split("/")
    try:
        subdir_index = parts.index(dataset_subdir.lower())
    except ValueError:
        return None

    if subdir_index + 1 >= len(parts):
        return None

    candidate = parts[subdir_index + 1]
    if candidate in PROBLEMS:
        return candidate
    return None


def list_problem_files(dataset_subdir: str) -> dict[str, list[str]]:
    api = HfApi()
    items = list(
        api.list_repo_tree(
            repo_id=REPO_ID,
            repo_type="dataset",
            recursive=True,
            expand=True,
        )
    )

    result: dict[str, list[str]] = defaultdict(list)

    for item in items:
        if not isinstance(item, RepoFile):
            continue

        path = item.path
        problem = problem_from_path(path, dataset_subdir)
        if problem is None:
            continue

        result[problem].append(path)

    for problem in result:
        result[problem] = sorted(result[problem])

    return dict(result)


def make_hf_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{path}"


def iter_row_lines(row: dict) -> list[tuple[str, int, str]]:
    text = row.get("text")
    if not text or not str(text).strip():
        return []

    source_path = str(row.get("source") or row.get("file_name") or "")
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return []

    return [(source_path, line_no, line) for line_no, line in enumerate(lines)]


def allocate_split_counts(total: int, target_counts: dict[str, int]) -> dict[str, int]:
    target_total = sum(target_counts.values())
    if total >= target_total:
        return dict(target_counts)

    allocated: dict[str, int] = {}
    assigned = 0
    items = list(target_counts.items())
    for index, (split_name, target) in enumerate(items):
        if index == len(items) - 1:
            allocated[split_name] = total - assigned
        else:
            count = (total * target) // target_total
            allocated[split_name] = count
            assigned += count
    return allocated


def collect_all_instances(
    problem: str,
    files: list[str],
    *,
    solution_algorithm: str,
) -> list[tuple[str, int, dict[str, Any]]]:
    parser = build_parser_registry(solution_algorithm)[problem]
    urls = [make_hf_url(path) for path in files]
    dataset = load_dataset(
        "text",
        data_files={"data": urls},
        split="data",
        streaming=True,
    )

    instances: list[tuple[str, int, dict[str, Any]]] = []
    public_problem_name = PROBLEM_ALIASES[problem]

    for row in tqdm(dataset, desc=f"{public_problem_name}/collect"):
        for source_path, source_line, line in iter_row_lines(row):
            instances.append((source_path, source_line, parser(line)))

    return instances


def stream_problem_splits(
    problem: str,
    files: list[str],
    output_root: Path,
    *,
    seed: int,
    solution_algorithm: str,
    target_counts: dict[str, int],
) -> None:
    if not files:
        raise RuntimeError(f"No files found for problem={problem}")

    public_problem_name = PROBLEM_ALIASES[problem]
    instances = collect_all_instances(
        problem,
        files,
        solution_algorithm=solution_algorithm,
    )
    total = len(instances)
    if total == 0:
        raise RuntimeError(f"No instances found for problem={problem}")

    split_counts = allocate_split_counts(total, target_counts)
    if total < sum(target_counts.values()):
        requested = ", ".join(
            f"{split}={count:,}" for split, count in target_counts.items()
        )
        actual = ", ".join(
            f"{split}={count:,}" for split, count in split_counts.items()
        )
        print(
            f"Warning: {problem} has only {total:,} unique instances; "
            f"using proportional split ({actual}) instead of ({requested})."
        )

    order = list(range(total))
    random.Random(seed).shuffle(order)

    cursor = 0
    for split_name, count in target_counts.items():
        split_size = split_counts[split_name]
        output_path = output_root / public_problem_name / f"{split_name}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            for split_index, instance_index in enumerate(
                tqdm(
                    order[cursor : cursor + split_size],
                    desc=f"{public_problem_name}/{split_name}",
                )
            ):
                source_path, source_line, parsed = instances[instance_index]
                record = record_from_parsed(
                    problem_key=problem,
                    split=split_name,
                    index=split_index,
                    seed=instance_seed(seed, split_index),
                    parsed=parsed,
                    source_path=Path(source_path),
                    source_line=source_line,
                    algorithm=solution_algorithm,
                )
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )

        cursor += split_size
        print(f"Wrote {split_size:,} rows to {output_path}")


def stream_problem(
    problem: str,
    files: list[str],
    output_root: Path,
    count: int,
    split: str,
    *,
    seed: int,
    solution_algorithm: str,
) -> None:
    if not files:
        raise RuntimeError(f"No files found for problem={problem}")

    parser = build_parser_registry(solution_algorithm)[problem]
    public_problem_name = PROBLEM_ALIASES[problem]
    urls = [make_hf_url(path) for path in files]

    output_path = output_root / public_problem_name / f"{split}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    available = 0
    pass_index = 0

    with output_path.open("w", encoding="utf-8") as handle:
        pbar = tqdm(total=count, desc=f"{public_problem_name}/{split}")

        while written < count:
            dataset = load_dataset(
                "text",
                data_files={split: urls},
                split=split,
                streaming=True,
            )

            rows_this_pass = 0
            for row in dataset:
                for source_path, source_line, line in iter_row_lines(row):
                    record = record_from_parsed(
                        problem_key=problem,
                        split=split,
                        index=written,
                        seed=instance_seed(seed, written),
                        parsed=parser(line),
                        source_path=Path(source_path),
                        source_line=source_line,
                        algorithm=solution_algorithm,
                    )
                    handle.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )

                    written += 1
                    rows_this_pass += 1
                    pbar.update(1)

                    if written >= count:
                        break

                if written >= count:
                    break

            if rows_this_pass == 0:
                break

            if pass_index == 0:
                available = rows_this_pass
                if available < count:
                    print(
                        f"Warning: {problem} has only {available:,} unique instances "
                        f"in {split}; cycling source files to reach {count:,}."
                    )

            pass_index += 1

        pbar.close()

    if written < count:
        raise RuntimeError(
            f"Only wrote {written} rows for {problem}, but requested {count}."
        )

    print(f"Wrote {written:,} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream ML4CO-Bench-101-SL txt files into JSONL without a full local download."
    )
    parser.add_argument("--output-root", type=Path, default=PUBLIC_DATA_84K_ROOT)
    parser.add_argument("--count", type=int, default=84_000)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--multi-split",
        action="store_true",
        help=(
            "Stream all available instances once, shuffle, and write train/val/test "
            "using TARGET_COUNTS (64,000 : 10,000 : 10,000). When fewer instances "
            "exist than requested, split sizes are scaled proportionally."
        ),
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=list(PROBLEMS),
        choices=list(PROBLEMS),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--dataset-subdir",
        type=str,
        default="train_dataset",
        help="Only stream files under this repo folder, e.g. train_dataset.",
    )
    parser.add_argument(
        "--solution-algorithm",
        type=str,
        default="ml4co",
        help="Algorithm key stored under record['solutions'].",
    )
    args = parser.parse_args()

    files_by_problem = list_problem_files(args.dataset_subdir)

    print("Files found:")
    for problem in args.problems:
        files = files_by_problem.get(problem, [])
        print(f"{problem}: {len(files)} files")
        for path in files[:5]:
            print(f"  - {path}")
        if len(files) > 5:
            print("  ...")

    for problem in args.problems:
        files = files_by_problem.get(problem, [])
        if args.multi_split:
            stream_problem_splits(
                problem=problem,
                files=files,
                output_root=args.output_root,
                seed=args.seed,
                solution_algorithm=args.solution_algorithm,
                target_counts=TARGET_COUNTS,
            )
            continue

        stream_problem(
            problem=problem,
            files=files,
            output_root=args.output_root,
            count=args.count,
            split=args.split,
            seed=args.seed,
            solution_algorithm=args.solution_algorithm,
        )


if __name__ == "__main__":
    main()
