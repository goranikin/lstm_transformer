import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from src.constants import (
    DECODER_KINDS,
    DEFAULT_SEEDS,
    DEFAULT_TARGET_ALGORITHM,
    GRAPH_PROBLEMS,
    MATRIX_ENCODERS,
    PROBLEM_NAMES,
    ProblemName,
)
from src.experiments.parameter_comparison import config_sequence, validate_values
from src.paths import problem_split_paths, resolve_data_root, resolve_user_path

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
        data_root=resolve_data_root(cfg.data.root),
        train_instances=int(cfg.data.train.instances),
        val_instances=int(cfg.data.validation.instances),
        test_instances=int(cfg.data.test.instances),
        output_root=str(resolve_user_path(cfg.paths.output_root)),
        parameter_budget=str(resolve_user_path(cfg.parameter_budget.path)),
        use_parameter_budget=bool(cfg.parameter_budget.enabled),
        epochs=int(cfg.trainer.epochs),
        steps_per_epoch=none_or_int(cfg.trainer.steps_per_epoch),
        batch_size=int(cfg.data.batch_size),
        eval_batch_size=int(cfg.data.eval_batch_size or cfg.data.batch_size),
        graph_batch_size=none_or_int(cfg.data.get("graph_batch_size")),
        graph_eval_batch_size=none_or_int(cfg.data.get("graph_eval_batch_size")),
        learning_rate=float(cfg.trainer.learning_rate),
        d_model=int(cfg.model.d_model),
        d_ff=int(cfg.model.d_ff),
        num_layers=int(cfg.model.num_layers),
        num_heads=int(cfg.model.num_heads),
        device=str(cfg.device),
        num_workers=int(cfg.data.num_workers),
        skip_sigmoid_routing=bool(cfg.skip_sigmoid_routing),
        stage=stage,
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
    data_root: Path,
    train_instances: int,
    val_instances: int,
    test_instances: int,
    output_root: str,
    parameter_budget: str,
    use_parameter_budget: bool,
    epochs: int,
    steps_per_epoch: int | None,
    batch_size: int,
    eval_batch_size: int,
    graph_batch_size: int | None,
    graph_eval_batch_size: int | None,
    learning_rate: float,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    device: str,
    num_workers: int,
    skip_sigmoid_routing: bool,
    stage: str,
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
                        paths = problem_split_paths(
                            problem,
                            train_instances=train_instances,
                            val_instances=val_instances,
                            test_instances=test_instances,
                            data_root=data_root,
                        )
                        output_dir = (
                            f"{output_root}/seed_{seed}/{mode}/"
                            f"{problem}/{encoder}/{decoder}"
                        )
                        train_batch_size, eval_batch = resolve_batch_sizes(
                            problem,
                            batch_size=batch_size,
                            eval_batch_size=eval_batch_size,
                            graph_batch_size=graph_batch_size,
                            graph_eval_batch_size=graph_eval_batch_size,
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
                            f"data.root={data_root}",
                            f"data.train_path={paths['train']}",
                            f"data.val_path={paths['val']}",
                            f"data.test_path={paths['test']}",
                            f"data.target_algorithm={DEFAULT_TARGET_ALGORITHM[problem]}",
                            f"seed={seed}",
                            f"trainer.epochs={epochs}",
                            f"data.batch_size={train_batch_size}",
                            f"data.eval_batch_size={eval_batch}",
                            f"trainer.learning_rate={learning_rate}",
                            f"model.num_layers={num_layers}",
                            f"model.num_heads={num_heads}",
                            f"device={device}",
                            f"data.num_workers={num_workers}",
                            f"paths.output_dir={output_dir}",
                            f"parameter_budget.enabled={str(use_parameter_budget).lower()}",
                            f"parameter_budget.path={parameter_budget}",
                            f"wandb.group=matrix/{stage}/seed_{seed}",
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


def resolve_batch_sizes(
    problem: ProblemName,
    *,
    batch_size: int,
    eval_batch_size: int,
    graph_batch_size: int | None,
    graph_eval_batch_size: int | None,
) -> tuple[int, int]:
    if problem not in GRAPH_PROBLEMS:
        return batch_size, eval_batch_size
    train_batch = graph_batch_size if graph_batch_size is not None else batch_size
    if graph_eval_batch_size is not None:
        eval_batch = graph_eval_batch_size
    elif graph_batch_size is not None:
        eval_batch = graph_batch_size
    else:
        eval_batch = eval_batch_size
    return train_batch, eval_batch


def none_or_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return int(value)


if __name__ == "__main__":
    main()
