import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, RepoFile
from tqdm import tqdm

from src.generate_data.common import instance_seed
from src.generate_data.ml4co_to_jsonl import (
    PROBLEM_ALIASES,
    build_parser_registry,
    record_from_parsed,
)

REPO_ID = "ML4CO/ML4CO-Bench-101-SL"

# TSP is intentionally excluded here.
PROBLEMS = ["cvrp", "mis", "mcl", "mvc"]


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

    dataset = load_dataset(
        "text",
        data_files={split: urls},
        split=split,
        streaming=True,
    )

    output_path = output_root / public_problem_name / f"{split}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0

    with output_path.open("w", encoding="utf-8") as handle:
        pbar = tqdm(total=count, desc=f"{public_problem_name}/{split}")

        for row in dataset:
            text = row.get("text")
            if not text or not str(text).strip():
                continue

            source_path = row.get("source") or row.get("file_name") or ""
            record = record_from_parsed(
                problem_key=problem,
                split=split,
                index=written,
                seed=instance_seed(seed, written),
                parsed=parser(str(text).strip()),
                source_path=Path(str(source_path)),
                source_line=0,
                algorithm=solution_algorithm,
            )
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )

            written += 1
            pbar.update(1)

            if written >= count:
                break

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
    parser.add_argument("--output-root", type=Path, default=Path("data_public_100k"))
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--split", type=str, default="train")
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
    for problem in PROBLEMS:
        files = files_by_problem.get(problem, [])
        print(f"{problem}: {len(files)} files")
        for path in files[:5]:
            print(f"  - {path}")
        if len(files) > 5:
            print("  ...")

    for problem in PROBLEMS:
        stream_problem(
            problem=problem,
            files=files_by_problem.get(problem, []),
            output_root=args.output_root,
            count=args.count,
            split=args.split,
            seed=args.seed,
            solution_algorithm=args.solution_algorithm,
        )


if __name__ == "__main__":
    main()
