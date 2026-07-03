from __future__ import annotations

import shlex
import subprocess
from collections.abc import Sequence
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from src.constants import (
    DECODER_KINDS,
    DEFAULT_SEEDS,
    DEFAULT_TARGET_ALGORITHM,
    MATRIX_ENCODERS,
    PROBLEM_FILE_PREFIX,
    PROBLEM_NAMES,
    PROBLEM_PATH_DIR,
    ProblemName,
)
from src.experiments.parameter_comparison import config_sequence, validate_values

STAGES: dict[str, tuple[ProblemName, ...]] = {
    "all": PROBLEM_NAMES,
    "routing": ("tsp", "cvrp"),
    "subset": ("knapsack", "mis", "max_clique", "vertex_cover"),
    "hybrid": ("orienteering",),
}


@hydra.main(version_base=None, config_path="../../configs", config_name="matrix")
def main(cfg: DictConfig) -> None:
    run_from_config(cfg)


def run_from_config(cfg: DictConfig) -> list[list[str]]:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    stage = str(cfg.stage)
    if stage not in STAGES:
        raise ValueError(f"Unsupported stage: {stage}")
    problems = resolve_problems(cfg.problems, stage)
    encoders = validate_values(
        config_sequence(cfg.encoders, MATRIX_ENCODERS),
        MATRIX_ENCODERS,
        "encoder",
    )
    decoders = validate_values(
        config_sequence(cfg.decoders, DECODER_KINDS),
        DECODER_KINDS,
        "decoder",
    )
    modes = validate_values(
        config_sequence(cfg.modes, ("supervised", "rl")),
        ("supervised", "rl"),
        "mode",
    )
    seeds = tuple(
        int(seed) for seed in config_sequence(cfg.seeds, map(str, DEFAULT_SEEDS))
    )
    commands = build_commands(
        problems=problems,
        encoders=encoders,
        decoders=decoders,
        modes=modes,
        seeds=seeds,
        data_root=str(cfg.data.root),
        train_instances=int(cfg.data.train.instances),
        train_seed=int(cfg.data.train.seed),
        val_instances=int(cfg.data.validation.instances),
        val_seed=int(cfg.data.validation.seed),
        test_instances=int(cfg.data.test.instances),
        test_seed=int(cfg.data.test.seed),
        output_root=str(cfg.paths.output_root),
        parameter_budget=str(cfg.parameter_budget.path),
        use_parameter_budget=bool(cfg.parameter_budget.enabled),
        epochs=int(cfg.trainer.epochs),
        steps_per_epoch=none_or_int(cfg.trainer.steps_per_epoch),
        batch_size=int(cfg.data.batch_size),
        eval_batch_size=int(cfg.data.eval_batch_size or cfg.data.batch_size),
        learning_rate=float(cfg.trainer.learning_rate),
        d_model=int(cfg.model.d_model),
        d_ff=int(cfg.model.d_ff),
        num_layers=int(cfg.model.num_layers),
        num_heads=int(cfg.model.num_heads),
        device=str(cfg.device),
        num_workers=int(cfg.data.num_workers),
        skip_sigmoid_routing=bool(cfg.skip_sigmoid_routing),
    )
    action = "Running" if bool(cfg.execute) else "Dry run"
    print(f"{action} {len(commands)} command(s).")
    for index, command in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
        if bool(cfg.execute):
            subprocess.run(command, check=True)
    return commands


def resolve_problems(value: Any, stage: str) -> tuple[ProblemName, ...]:
    selected = validate_values(
        config_sequence(value, STAGES[stage]),
        PROBLEM_NAMES,
        "problem",
    )
    return tuple(cast(ProblemName, problem) for problem in selected)


def build_commands(
    *,
    problems: Sequence[ProblemName],
    encoders: Sequence[str],
    decoders: Sequence[str],
    modes: Sequence[str],
    seeds: Sequence[int],
    data_root: str,
    train_instances: int,
    train_seed: int,
    val_instances: int,
    val_seed: int,
    test_instances: int,
    test_seed: int,
    output_root: str,
    parameter_budget: str,
    use_parameter_budget: bool,
    epochs: int,
    steps_per_epoch: int | None,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    device: str,
    num_workers: int,
    skip_sigmoid_routing: bool,
) -> list[list[str]]:
    commands: list[list[str]] = []
    for seed in seeds:
        for mode in modes:
            for problem in problems:
                for encoder in encoders:
                    for decoder in decoders:
                        if (
                            skip_sigmoid_routing
                            and decoder == "sigmoid_subset"
                            and problem in {"tsp", "cvrp"}
                        ):
                            continue
                        paths = problem_paths(
                            data_root,
                            problem,
                            train_instances=train_instances,
                            train_seed=train_seed,
                            val_instances=val_instances,
                            val_seed=val_seed,
                            test_instances=test_instances,
                            test_seed=test_seed,
                        )
                        output_dir = (
                            f"{output_root}/seed_{seed}/{mode}/"
                            f"{problem}/{encoder}/{decoder}"
                        )
                        command = [
                            "uv",
                            "run",
                            "python",
                            "-m",
                            "src.experiments.run",
                            f"problem={problem}",
                            f"encoder={encoder}",
                            f"decoder={decoder}",
                            f"mode={mode}",
                            f"data.train_path={paths['train']}",
                            f"data.val_path={paths['val']}",
                            f"data.test_path={paths['test']}",
                            f"data.target_algorithm={DEFAULT_TARGET_ALGORITHM[problem]}",
                            f"seed={seed}",
                            f"trainer.epochs={epochs}",
                            f"data.batch_size={batch_size}",
                            f"data.eval_batch_size={eval_batch_size}",
                            f"trainer.learning_rate={learning_rate}",
                            f"model.num_layers={num_layers}",
                            f"model.num_heads={num_heads}",
                            f"device={device}",
                            f"data.num_workers={num_workers}",
                            f"paths.output_dir={output_dir}",
                            f"parameter_budget.enabled={str(use_parameter_budget).lower()}",
                            f"parameter_budget.path={parameter_budget}",
                        ]
                        if steps_per_epoch is not None:
                            command.append(f"trainer.steps_per_epoch={steps_per_epoch}")
                        if not use_parameter_budget:
                            command.extend(
                                [
                                    f"model.d_model={d_model}",
                                    f"model.d_ff={d_ff}",
                                ]
                            )
                        commands.append(command)
    return commands


def problem_paths(
    data_root: str,
    problem: ProblemName,
    *,
    train_instances: int,
    train_seed: int,
    val_instances: int,
    val_seed: int,
    test_instances: int,
    test_seed: int,
) -> dict[str, str]:
    directory = PROBLEM_PATH_DIR[problem]
    prefix = PROBLEM_FILE_PREFIX[problem]
    return {
        "train": (
            f"{data_root}/{directory}/"
            f"{prefix}_train_{train_instances}_seed{train_seed}.jsonl"
        ),
        "val": (
            f"{data_root}/{directory}/{prefix}_val_{val_instances}_seed{val_seed}.jsonl"
        ),
        "test": (
            f"{data_root}/{directory}/"
            f"{prefix}_test_{test_instances}_seed{test_seed}.jsonl"
        ),
    }


def none_or_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return int(value)


if __name__ == "__main__":
    main()
