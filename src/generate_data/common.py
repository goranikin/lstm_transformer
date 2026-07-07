import json
import os
from collections.abc import Iterable, Iterator
from typing import Any

from src.constants import ProblemName
from src.paths import problem_dataset_path

JsonRecord = dict[str, Any]


class ExternalSolverError(RuntimeError):
    """Raised when a requested external solver is unavailable or fails."""


def instance_seed(base_seed: int, index: int) -> int:
    """Derive a deterministic, reproducible seed for one generated instance."""
    if index < 0:
        raise ValueError("index must be non-negative")
    return base_seed + index


def iter_instance_indices(start_index: int, num_instances: int) -> Iterator[int]:
    """Yield global instance indices for a generation chunk."""
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if num_instances <= 0:
        raise ValueError("num_instances must be positive")
    for offset in range(num_instances):
        yield start_index + offset


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def write_jsonl(records: Iterable[JsonRecord], path: str) -> int:
    ensure_parent_dir(path)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: str) -> Iterator[JsonRecord]:
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object")
            yield record


def load_jsonl(path: str) -> list[JsonRecord]:
    return list(read_jsonl(path))


def resolve_output_path(
    problem: ProblemName,
    *,
    seed: int,
    output_path: str | None,
) -> str:
    if output_path is not None:
        return output_path
    return str(problem_dataset_path(problem, seed))
