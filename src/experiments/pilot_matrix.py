from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from src.constants import PROBLEM_NAMES, ProblemName
from src.experiments.matrix import run_from_config as run_matrix
from src.experiments.parameter_comparison import config_sequence, validate_values
from src.paths import problem_dataset_path, resolve_data_root

PILOT_PROBLEMS: tuple[ProblemName, ...] = (
    "tsp",
    "cvrp",
    "knapsack",
    "mis",
    "max_clique",
    "vertex_cover",
)


@hydra.main(version_base=None, config_path="../../configs", config_name="pilot_matrix")
def main(cfg: DictConfig) -> None:
    run_from_config(cfg)


def run_from_config(cfg: DictConfig) -> list[list[str]]:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    problems = resolve_pilot_problems(cfg.problems)
    data_root = resolve_data_root(cfg.data.root)
    source_counts = resolve_source_counts(cfg)
    pilot_counts = resolve_pilot_counts(cfg)

    if bool(cfg.extract):
        extract_pilot_datasets(
            problems=problems,
            data_root=data_root,
            source_counts=source_counts,
            pilot_counts=pilot_counts,
        )

    matrix_cfg = build_matrix_config(cfg, problems=problems)
    commands = run_matrix(matrix_cfg)
    print_summary(
        problems=problems,
        pilot_counts=pilot_counts,
        commands=commands,
        execute=bool(cfg.execute),
    )
    return commands


def resolve_pilot_problems(value: Any) -> tuple[ProblemName, ...]:
    selected = validate_values(
        config_sequence(value, PILOT_PROBLEMS),
        PROBLEM_NAMES,
        "problem",
    )
    return tuple(cast(ProblemName, problem) for problem in selected)


def resolve_source_counts(cfg: DictConfig) -> dict[str, int]:
    return {
        "train": int(cfg.source.train.instances),
        "validation": int(cfg.source.validation.instances),
        "test": int(cfg.source.test.instances),
    }


def resolve_pilot_counts(cfg: DictConfig) -> dict[str, int]:
    return {
        "train": int(cfg.data.train.instances),
        "validation": int(cfg.data.validation.instances),
        "test": int(cfg.data.test.instances),
    }


def extract_pilot_datasets(
    *,
    problems: tuple[ProblemName, ...],
    data_root: Path,
    source_counts: dict[str, int],
    pilot_counts: dict[str, int],
) -> None:
    print("Extracting pilot datasets:", flush=True)
    for problem in problems:
        for split in ("train", "validation", "test"):
            source_instances = source_counts[split]
            pilot_instances = pilot_counts[split]
            source_path = problem_dataset_path(
                problem,
                split=_split_name(split),
                instances=source_instances,
                data_root=data_root,
            )
            pilot_path = problem_dataset_path(
                problem,
                split=_split_name(split),
                instances=pilot_instances,
                data_root=data_root,
            )
            written = extract_jsonl_head(
                source_path=source_path,
                dest_path=pilot_path,
                count=pilot_instances,
            )
            print(
                f"  {problem} {split}: {written} lines "
                f"({source_path.name} -> {pilot_path.name})",
                flush=True,
            )


def extract_jsonl_head(
    *,
    source_path: Path,
    dest_path: Path,
    count: int,
) -> int:
    if count <= 0:
        raise ValueError("Pilot instance count must be positive")
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source dataset: {source_path}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with (
        source_path.open(encoding="utf-8") as source,
        dest_path.open("w", encoding="utf-8") as dest,
    ):
        for line in source:
            if written >= count:
                break
            dest.write(line)
            written += 1

    if written < count:
        raise ValueError(
            f"Source {source_path} has only {written} lines; "
            f"requested {count} pilot instances"
        )
    return written


def build_matrix_config(
    cfg: DictConfig,
    *,
    problems: tuple[ProblemName, ...],
) -> DictConfig:
    matrix_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    matrix_cfg.problems = list(problems)
    matrix_cfg.paths.output_root = str(
        Path(str(matrix_cfg.paths.output_root)).as_posix()
    )
    return cast(DictConfig, matrix_cfg)


def print_summary(
    *,
    problems: tuple[ProblemName, ...],
    pilot_counts: dict[str, int],
    commands: list[list[str]],
    execute: bool,
) -> None:
    action = "Executed" if execute else "Planned"
    print(
        f"\n{action} pilot matrix: {len(commands)} run(s).",
        flush=True,
    )
    print(
        "Pilot data sizes per problem: "
        f"train={pilot_counts['train']}, "
        f"val={pilot_counts['validation']}, "
        f"test={pilot_counts['test']}.",
        flush=True,
    )
    print(
        "Problems: "
        + ", ".join(problems)
        + " (orienteering omitted; not on current server datasets).",
        flush=True,
    )


def _split_name(split: str) -> str:
    if split == "validation":
        return "val"
    return split


if __name__ == "__main__":
    main()
