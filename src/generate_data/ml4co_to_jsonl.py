from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm import tqdm

from src.generate_data.common import instance_seed

TARGET_COUNTS = {
    "train": 64_000,
    "val": 10_000,
    "test": 10_000,
}

PROBLEM_KEYS = ("tsp", "cvrp", "mis", "mcl", "mvc")

PROBLEM_ALIASES = {
    "tsp": "tsp",
    "cvrp": "cvrp",
    "mis": "mis",
    "mcl": "max_clique",
    "mvc": "vertex_cover",
}

PROBLEM_RECORD_NAMES = {
    "tsp": "tsp",
    "cvrp": "cvrp",
    "mis": "mis",
    "mcl": "maximum_clique",
    "mvc": "minimum_vertex_cover",
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


def iter_text_lines(paths: list[Path]) -> Iterable[tuple[Path, int, str]]:
    for path in paths:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_no, line in enumerate(handle):
                line = line.strip()
                if line:
                    yield path, line_no, line


def collect_files(root: Path, problem_key: str, dataset_subdir: str) -> list[Path]:
    problem_root = root / dataset_subdir / problem_key
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


def build_from_ml4co(
    raw_root: Path,
    output_root: Path,
    problem_key: str,
    seed: int,
    *,
    dataset_subdir: str,
    solution_algorithm: str,
) -> None:
    public_problem_name = PROBLEM_ALIASES[problem_key]
    parser = build_parser_registry(solution_algorithm)[problem_key]

    files = collect_files(raw_root, problem_key, dataset_subdir)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ML4CO-Bench-101-SL txt data into project JSONL splits."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("raw/ml4co"))
    parser.add_argument("--output-root", type=Path, default=Path("data_public"))
    parser.add_argument(
        "--dataset-subdir",
        type=str,
        default="train_dataset",
        help="Subdirectory under --raw-root that contains per-problem folders.",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=list(PROBLEM_KEYS),
        choices=list(PROBLEM_KEYS),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--solution-algorithm",
        type=str,
        default="ml4co",
        help="Algorithm key stored under record['solutions'].",
    )
    args = parser.parse_args()

    for problem_key in args.problems:
        build_from_ml4co(
            raw_root=args.raw_root,
            output_root=args.output_root,
            problem_key=problem_key,
            seed=args.seed,
            dataset_subdir=args.dataset_subdir,
            solution_algorithm=args.solution_algorithm,
        )


if __name__ == "__main__":
    main()
