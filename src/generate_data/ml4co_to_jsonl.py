import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

from tqdm import tqdm

from src.constants import ProblemName
from src.generate_data.common import instance_seed
from src.paths import (
    LOCAL_DB_ROOT,
    PUBLIC_DATA_84K_ROOT,
    RAW_ML4CO_ROOT,
    problem_dataset_path,
)

TARGET_COUNTS = {
    "train": 64_000,
    "val": 10_000,
    "test": 10_000,
}

SPLIT_SEEDS = {
    "train": 1234,
    "val": 4321,
    "test": 9999,
}

PROBLEM_KEYS = ("tsp", "cvrp", "mis", "mcl", "mvc")

JSONL_DEFAULT_PROBLEMS = ("cvrp", "mis", "mvc", "mcl")

PROBLEM_ALIASES = {
    "tsp": "tsp",
    "cvrp": "cvrp",
    "mis": "mis",
    "mcl": "max_clique",
    "mvc": "vertex_cover",
}

PROBLEM_KEY_TO_CANONICAL: dict[str, ProblemName] = {
    "tsp": "tsp",
    "cvrp": "cvrp",
    "mis": "mis",
    "mcl": "max_clique",
    "mvc": "vertex_cover",
}

RECORD_TO_CANONICAL: dict[str, ProblemName] = {
    "tsp": "tsp",
    "cvrp": "cvrp",
    "mis": "mis",
    "maximum_clique": "max_clique",
    "minimum_vertex_cover": "vertex_cover",
}

DIR_TO_PROBLEM: dict[str, ProblemName] = {
    "tsp": "tsp",
    "cvrp": "cvrp",
    "mis": "mis",
    "max_clique": "max_clique",
    "vertex_cover": "vertex_cover",
}

PROBLEM_RECORD_NAMES = {
    "tsp": "tsp",
    "cvrp": "cvrp",
    "mis": "mis",
    "mcl": "maximum_clique",
    "mvc": "minimum_vertex_cover",
}

INSTANCE_FIELDS: dict[ProblemName, tuple[str, ...]] = {
    "tsp": ("num_nodes", "coordinates"),
    "cvrp": (
        "num_customers",
        "depot",
        "coordinates",
        "demands",
        "vehicle_capacity",
    ),
    "mis": ("num_nodes", "edges", "node_weights"),
    "max_clique": ("num_nodes", "edges", "node_weights"),
    "vertex_cover": ("num_nodes", "edges", "node_weights"),
}


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            count += 1
    return count


def count_jsonl_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


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


def build_assignment(
    total: int,
    *,
    seed: int,
    target_counts: dict[str, int],
) -> tuple[dict[str, int], list[tuple[str, int]]]:
    split_counts = allocate_split_counts(total, target_counts)
    used = sum(split_counts.values())

    if used < total:
        skipped = total - used
        print(
            f"Using {used:,} of {total:,} instances "
            f"({skipped:,} discarded after shuffle)."
        )

    order = list(range(used))
    random.Random(seed).shuffle(order)

    assignment: list[tuple[str, int] | None] = [None] * used
    cursor = 0
    for split_name, _target in target_counts.items():
        split_size = split_counts[split_name]
        for split_index, instance_index in enumerate(
            order[cursor : cursor + split_size]
        ):
            assignment[instance_index] = (split_name, split_index)
        cursor += split_size

    if any(item is None for item in assignment):
        raise RuntimeError("Failed to build split assignment.")

    return split_counts, [item for item in assignment if item is not None]


def resolve_problem_name(
    record: dict[str, Any],
    *,
    directory_name: str,
    problem_key: str | None = None,
) -> ProblemName:
    if problem_key is not None:
        return PROBLEM_KEY_TO_CANONICAL[problem_key]
    if directory_name in DIR_TO_PROBLEM:
        return DIR_TO_PROBLEM[directory_name]
    canonical = RECORD_TO_CANONICAL.get(str(record.get("problem")))
    if canonical is not None:
        return canonical
    raise ValueError(
        f"Could not infer problem for {directory_name!r}; pass --problem explicitly."
    )


def normalize_record(
    record: dict[str, Any],
    *,
    problem: ProblemName,
    split_name: str,
    split_index: int,
) -> dict[str, Any]:
    """Convert ML4CO JSONL rows into the minimal schema used by training datasets."""
    split_seed = SPLIT_SEEDS[split_name]
    normalized: dict[str, Any] = {
        "problem": problem,
        "index": split_index,
        "seed": instance_seed(split_seed, split_index),
    }

    for field in INSTANCE_FIELDS[problem]:
        if field in record:
            normalized[field] = record[field]

    solutions = record.get("solutions")
    if isinstance(solutions, dict):
        normalized["solutions"] = solutions

    return normalized


def data_output_path(data_root: Path, problem: ProblemName, split_name: str) -> Path:
    return problem_dataset_path(
        problem,
        split=split_name,
        instances=TARGET_COUNTS[split_name],
        data_root=data_root,
    )


def discover_train_files(input_roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in input_roots:
        files.extend(
            sorted(path for path in root.glob("*/train.jsonl") if path.is_file())
        )
    return files


def resolve_jsonl_input(
    problem_key: str,
    *,
    raw_root: Path,
    public_84k_root: Path,
) -> Path:
    public_name = PROBLEM_ALIASES[problem_key]
    if problem_key == "mcl":
        return public_84k_root / public_name / "train.jsonl"
    return raw_root / public_name / "train.jsonl"


def convert_jsonl_file(
    input_path: Path,
    *,
    output_root: Path,
    problem_key: str | None = None,
    seed: int,
    target_counts: dict[str, int] | None = None,
) -> dict[str, int]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    counts = target_counts or TARGET_COUNTS
    total = count_jsonl_lines(input_path)
    if total == 0:
        raise ValueError(f"No records found in {input_path}")

    split_counts, assignment = build_assignment(
        total,
        seed=seed,
        target_counts=counts,
    )

    if total < sum(counts.values()):
        requested = ", ".join(f"{name}={count:,}" for name, count in counts.items())
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
        problem_key=problem_key,
    )

    try:
        for split_name, count in split_counts.items():
            if count == 0:
                continue
            out_path = data_output_path(output_root, resolved_problem, split_name)
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
                record = normalize_record(
                    json.loads(stripped),
                    problem=resolved_problem,
                    split_name=split_name,
                    split_index=split_index,
                )

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


def iter_text_lines(paths: list[Path]) -> Iterable[tuple[Path, int, str]]:
    for path in paths:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_no, line in enumerate(handle):
                line = line.strip()
                if line:
                    yield path, line_no, line


def collect_txt_files(root: Path, problem_key: str) -> list[Path]:
    problem_root = root / problem_key
    if not problem_root.exists():
        raise FileNotFoundError(f"Missing folder: {problem_root}")

    files = [
        path
        for path in problem_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".txt"
    ]
    if not files:
        raise FileNotFoundError(f"No .txt data files found under: {problem_root}")

    return sorted(files)


def _parse_edge_list(tokens: list[str]) -> list[list[int]]:
    return [
        [int(tokens[index]), int(tokens[index + 1])]
        for index in range(0, len(tokens), 2)
    ]


def _selected_nodes(labels: list[int]) -> list[int]:
    return [index for index, value in enumerate(labels) if value == 1]


def _graph_solution_record(
    labels: list[int],
    algorithm: str,
) -> dict[str, Any]:
    nodes = _selected_nodes(labels)
    return {
        "algorithm": algorithm,
        "is_exact": True,
        "nodes": nodes,
        "size": len(nodes),
        "metadata": {"source": "ML4CO-Bench-101-SL"},
    }


def parse_graph_line(line: str, algorithm: str) -> dict[str, Any]:
    if " weights " in line:
        edge_text, remainder = line.split(" weights ", 1)
        weights_text, labels_text = remainder.split(" label ", 1)
        node_weights = [float(value) for value in weights_text.split()]
    else:
        edge_text, labels_text = line.split(" label ", 1)
        node_weights = None

    edges = _parse_edge_list(edge_text.split())
    labels = [int(value) for value in labels_text.split()]
    num_nodes = len(labels)

    instance: dict[str, Any] = {
        "num_nodes": num_nodes,
        "edges": edges,
    }
    if node_weights is not None:
        if len(node_weights) != num_nodes:
            raise ValueError(
                f"Expected {num_nodes} node weights, got {len(node_weights)}"
            )
        instance["node_weights"] = node_weights

    return {
        "instance": instance,
        "solution": _graph_solution_record(labels, algorithm),
    }


def parse_tsp_line(line: str, algorithm: str) -> dict[str, Any]:
    instance_text, tour_text = line.split(" output ", 1)
    values = instance_text.split()
    coordinates = [
        [float(values[index]), float(values[index + 1])]
        for index in range(0, len(values), 2)
    ]
    # ML4CO txt stores 1-indexed node ids in the tour.
    tour = [int(value) - 1 for value in tour_text.split()]
    if len(tour) > len(coordinates) and tour[0] == tour[-1]:
        tour = tour[:-1]

    return {
        "instance": {
            "num_nodes": len(coordinates),
            "coordinates": coordinates,
        },
        "solution": {
            "algorithm": algorithm,
            "is_exact": True,
            "tour": tour,
            "metadata": {"source": "ML4CO-Bench-101-SL"},
        },
    }


def _cvrp_tour_to_routes(tour: list[int]) -> list[list[int]]:
    routes: list[list[int]] = []
    current: list[int] = []
    for node in tour:
        if node == 0:
            if current:
                routes.append([customer - 1 for customer in current])
                current = []
            continue
        current.append(node)
    if current:
        routes.append([customer - 1 for customer in current])
    return routes


def parse_cvrp_line(line: str, algorithm: str) -> dict[str, Any]:
    if not line.startswith("depots "):
        raise ValueError("CVRP line must start with 'depots '")

    depot_and_rest = line.split("depots ", 1)[1]
    depot_text, points_and_rest = depot_and_rest.split(" points ", 1)
    points_text, demands_and_rest = points_and_rest.split(" demands ", 1)
    demands_text, capacity_and_tour = demands_and_rest.split(" capacity ", 1)
    capacity_text, tour_text = capacity_and_tour.split(" output ", 1)

    depot_values = [float(value) for value in depot_text.split()]
    depot = [depot_values[0], depot_values[1]]
    point_values = points_text.split()
    coordinates = [
        [float(point_values[index]), float(point_values[index + 1])]
        for index in range(0, len(point_values), 2)
    ]
    demands = [int(float(value)) for value in demands_text.split()]
    vehicle_capacity = int(float(capacity_text))
    tour = [int(value) for value in tour_text.split()]
    routes = _cvrp_tour_to_routes(tour)

    return {
        "instance": {
            "num_customers": len(demands),
            "depot": depot,
            "coordinates": coordinates,
            "demands": demands,
            "vehicle_capacity": vehicle_capacity,
        },
        "solution": {
            "algorithm": algorithm,
            "is_exact": True,
            "routes": routes,
            "metadata": {"source": "ML4CO-Bench-101-SL", "tour": tour},
        },
    }


def build_parser_registry(
    algorithm: str,
) -> dict[str, Callable[[str], dict[str, Any]]]:
    return {
        "tsp": lambda line: parse_tsp_line(line, algorithm),
        "cvrp": lambda line: parse_cvrp_line(line, algorithm),
        "mis": lambda line: parse_graph_line(line, algorithm),
        "mcl": lambda line: parse_graph_line(line, algorithm),
        "mvc": lambda line: parse_graph_line(line, algorithm),
    }


def record_from_parsed(
    *,
    problem_key: str,
    split: str,
    index: int,
    seed: int,
    parsed: dict[str, Any],
    source_path: Path,
    source_line: int,
    algorithm: str,
) -> dict[str, Any]:
    public_problem_name = PROBLEM_ALIASES[problem_key]
    instance = dict(parsed["instance"])
    solution = dict(parsed["solution"])

    record: dict[str, Any] = {
        "id": f"{public_problem_name}_{split}_{index:06d}",
        "problem": PROBLEM_RECORD_NAMES[problem_key],
        "source": "ML4CO-Bench-101-SL",
        "split": split,
        "index": index,
        "seed": seed,
        **instance,
        "solutions": {algorithm: solution},
        "metadata": {
            "original_file": str(source_path),
            "original_line": source_line,
        },
    }
    return record


def build_from_ml4co_txt(
    raw_root: Path,
    output_root: Path,
    problem_key: str,
    seed: int,
    *,
    solution_algorithm: str,
) -> None:
    public_problem_name = PROBLEM_ALIASES[problem_key]
    parser = build_parser_registry(solution_algorithm)[problem_key]

    files = collect_txt_files(raw_root, problem_key)
    rows = list(iter_text_lines(files))

    rng = random.Random(seed)
    rng.shuffle(rows)

    total_needed = sum(TARGET_COUNTS.values())
    if len(rows) < total_needed:
        raise ValueError(
            f"{problem_key} has only {len(rows):,} rows, "
            f"but {total_needed:,} are needed."
        )

    cursor = 0
    for split, count in TARGET_COUNTS.items():
        split_rows = rows[cursor : cursor + count]
        cursor += count

        def records() -> Iterable[dict[str, Any]]:
            for idx, (path, line_no, line) in enumerate(
                tqdm(split_rows, desc=f"{public_problem_name}/{split}")
            ):
                parsed = parser(line)
                yield record_from_parsed(
                    problem_key=problem_key,
                    split=split,
                    index=idx,
                    seed=instance_seed(seed, idx),
                    parsed=parsed,
                    source_path=path,
                    source_line=line_no,
                    algorithm=solution_algorithm,
                )

        out_path = output_root / public_problem_name / f"{split}.jsonl"
        written = write_jsonl(out_path, records())
        print(f"Wrote {written:,} records to {out_path}")


def build_from_jsonl(
    *,
    raw_root: Path,
    public_84k_root: Path,
    output_root: Path,
    problem_key: str,
    seed: int,
) -> None:
    input_path = resolve_jsonl_input(
        problem_key,
        raw_root=raw_root,
        public_84k_root=public_84k_root,
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    convert_jsonl_file(
        input_path,
        output_root=output_root,
        problem_key=problem_key,
        seed=seed,
        target_counts=TARGET_COUNTS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ML4CO datasets into train/val/test JSONL splits compatible "
            "with src/experiments/run.py.\n\n"
            "By default, reads monolithic JSONL files from local_db/raw and "
            "data_public_84k, then writes normalized files under the local data root "
            "(e.g. cvrp/cvrp50_train_64000.jsonl). Use --input-format txt for "
            "legacy ML4CO .txt sources.\n\n"
            "Example:\n"
            "  uv run python -m src.generate_data.ml4co_to_jsonl\n\n"
            "  uv run python -m src.generate_data.ml4co_to_jsonl \\\n"
            f"    --input-root {LOCAL_DB_ROOT / 'raw'} \\\n"
            f"    --input-root {PUBLIC_DATA_84K_ROOT}\n\n"
            "For ML4CO downloads, pass data.target_algorithm=ml4co when training."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ML4CO_ROOT)
    parser.add_argument(
        "--public-84k-root",
        type=Path,
        default=RAW_ML4CO_ROOT,
        help="Root containing max_clique/train.jsonl when --problems includes mcl.",
    )
    parser.add_argument("--output-root", type=Path, default=LOCAL_DB_ROOT)
    parser.add_argument(
        "--input-format",
        choices=("jsonl", "txt"),
        default="jsonl",
        help="jsonl streams local_db/raw/*.jsonl; txt parses legacy ML4CO .txt files.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Convert a single JSONL file instead of the default per-problem paths.",
    )
    parser.add_argument(
        "--input-root",
        action="append",
        type=Path,
        help="Convert every */train.jsonl found under this directory. Repeatable.",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=list(JSONL_DEFAULT_PROBLEMS),
        choices=list(PROBLEM_KEYS),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Shuffle seed used before assigning train/val/test rows.",
    )
    parser.add_argument(
        "--solution-algorithm",
        type=str,
        default="ml4co",
        help="Algorithm key stored under record['solutions'] (txt mode only).",
    )
    args = parser.parse_args()

    if args.input_format == "jsonl":
        if args.input_root:
            train_files = discover_train_files(args.input_root)
            if not train_files:
                roots = ", ".join(str(root) for root in args.input_root)
                raise FileNotFoundError(f"No */train.jsonl files found under: {roots}")
            for train_file in train_files:
                print(f"\n=== {train_file} ===")
                convert_jsonl_file(
                    train_file,
                    output_root=args.output_root,
                    seed=args.seed,
                    target_counts=TARGET_COUNTS,
                )
            return

        if args.input is not None:
            convert_jsonl_file(
                args.input,
                output_root=args.output_root,
                seed=args.seed,
                target_counts=TARGET_COUNTS,
            )
            return

        for problem_key in args.problems:
            print(f"\n=== {problem_key} ===")
            build_from_jsonl(
                raw_root=args.raw_root,
                public_84k_root=args.public_84k_root,
                output_root=args.output_root,
                problem_key=problem_key,
                seed=args.seed,
            )
        return

    for problem_key in args.problems:
        build_from_ml4co_txt(
            raw_root=args.raw_root,
            output_root=args.output_root,
            problem_key=problem_key,
            seed=args.seed,
            solution_algorithm=args.solution_algorithm,
        )


if __name__ == "__main__":
    main()
