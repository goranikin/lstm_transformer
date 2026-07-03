import argparse
import shlex
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

from configs.validation import SCALES, load_base_trainer, load_scale
from src.constants import (
    DEFAULT_MATRIX_MODELS,
    MATRIX_NAMES,
    MODEL_NAMES,
    MODULAR_ARCHITECTURES,
    MODULE_TEST_PROBLEM,
    PROBLEM_NAMES,
    MatrixName,
)
from src.main.experiments.generate_smoke_data import (
    SmokeDatasetRequest,
    ensure_smoke_datasets,
    missing_smoke_paths,
)

PILOT_SCALE = load_scale("pilot")
BASE_TRAINER = load_base_trainer()


@dataclass(frozen=True)
class ProblemRun:
    problem: str
    train_path: str
    val_path: str | None = None
    target_algorithm: str | None = None


@dataclass(frozen=True)
class MatrixDefaults:
    epochs: int
    steps_per_epoch: str
    batch_size: int
    output_root: str
    progress_bar: bool
    checkpoint_every: int | None = None
    log_every: int | None = None
    save_best: bool | None = None
    save_last: bool | None = None
    keep_last_k: int | None = None
    label_smoothing: float | None = None
    baseline: str | None = None


@dataclass(frozen=True)
class ModuleTestPair:
    model: str
    run: ProblemRun


@dataclass(frozen=True)
class RunnerArgs:
    matrix: MatrixName
    scale: str
    models: str | None
    problems: str | None
    execute: bool
    prepare_data: bool
    skip_prepare_data: bool
    epochs: int | None
    steps_per_epoch: str | None
    batch_size: int | None
    output_root: str | None
    device: str | None
    progress_bar: bool | None
    checkpoint_every: int | None
    log_every: int | None
    num_workers: int | None
    learning_rate: float | None
    val_path: str | None


@dataclass(frozen=True)
class ResolvedSettings:
    scale: str
    epochs: int
    steps_per_epoch: str
    batch_size: int
    output_root: str
    progress_bar: bool
    checkpoint_every: int | None
    log_every: int | None
    save_best: bool | None
    save_last: bool | None
    keep_last_k: int | None
    label_smoothing: float | None
    baseline: str | None
    device: str | None
    num_workers: int | None
    learning_rate: float | None
    val_path: str | None


SUPERVISED_SMOKE_RUNS: tuple[ProblemRun, ...] = (
    ProblemRun(
        problem="tsp",
        train_path="data/tsp/tsp20_smoke_seed1234.jsonl",
        target_algorithm="concorde",
    ),
    ProblemRun(
        problem="cvrp",
        train_path="data/cvrp/cvrp20_smoke_seed1234.jsonl",
        target_algorithm="gurobi",
    ),
    ProblemRun(
        problem="mis",
        train_path="data/mis/mis30_p015_smoke_seed1234.jsonl",
        target_algorithm="gurobi",
    ),
    ProblemRun(
        problem="maximum_clique",
        train_path="data/max_clique/max_clique30_p050_smoke_seed1234.jsonl",
        target_algorithm="gurobi",
    ),
    ProblemRun(
        problem="minimum_vertex_cover",
        train_path="data/vertex_cover/vertex_cover30_p015_smoke_seed1234.jsonl",
        target_algorithm="gurobi",
    ),
    ProblemRun(
        problem="knapsack",
        train_path="data/knapsack/knapsack50_smoke_seed1234.jsonl",
        target_algorithm="dynamic_programming",
    ),
    ProblemRun(
        problem="orienteering",
        train_path="data/orienteering/orienteering20_smoke_seed1234.jsonl",
        target_algorithm="gurobi",
    ),
)

SUPERVISED_RUNS: tuple[ProblemRun, ...] = (
    ProblemRun(
        problem="tsp",
        train_path="data/tsp/tsp50_seed1234.jsonl",
        val_path="data/tsp/tsp50_val_seed4321.jsonl",
        target_algorithm="concorde",
    ),
    ProblemRun(
        problem="cvrp",
        train_path="data/cvrp/cvrp50_seed1234.jsonl",
        val_path="data/cvrp/cvrp50_val_seed4321.jsonl",
        target_algorithm="gurobi",
    ),
    ProblemRun(
        problem="mis",
        train_path="data/mis/mis100_p015_seed1234.jsonl",
        val_path="data/mis/mis100_p015_val_seed4321.jsonl",
        target_algorithm="gurobi",
    ),
    ProblemRun(
        problem="maximum_clique",
        train_path="data/max_clique/max_clique100_p050_seed1234.jsonl",
        val_path="data/max_clique/max_clique100_p050_val_seed4321.jsonl",
        target_algorithm="gurobi",
    ),
    ProblemRun(
        problem="minimum_vertex_cover",
        train_path="data/vertex_cover/vertex_cover100_p015_seed1234.jsonl",
        val_path="data/vertex_cover/vertex_cover100_p015_val_seed4321.jsonl",
        target_algorithm="gurobi",
    ),
    ProblemRun(
        problem="knapsack",
        train_path="data/knapsack/knapsack100_seed1234.jsonl",
        val_path="data/knapsack/knapsack100_val_seed4321.jsonl",
        target_algorithm="dynamic_programming",
    ),
    ProblemRun(
        problem="orienteering",
        train_path="data/orienteering/orienteering50_seed1234.jsonl",
        val_path="data/orienteering/orienteering50_val_seed4321.jsonl",
        target_algorithm="gurobi",
    ),
)

RL_RUNS: tuple[ProblemRun, ...] = (
    ProblemRun(
        problem="tsp",
        train_path="data/tsp/tsp50_seed1234.jsonl",
        val_path="data/tsp/tsp50_val_seed4321.jsonl",
    ),
    ProblemRun(
        problem="cvrp",
        train_path="data/cvrp/cvrp50_seed1234.jsonl",
        val_path="data/cvrp/cvrp50_val_seed4321.jsonl",
    ),
    ProblemRun(
        problem="mis",
        train_path="data/mis/mis100_p015_seed1234.jsonl",
        val_path="data/mis/mis100_p015_val_seed4321.jsonl",
    ),
    ProblemRun(
        problem="maximum_clique",
        train_path="data/max_clique/max_clique100_p050_seed1234.jsonl",
        val_path="data/max_clique/max_clique100_p050_val_seed4321.jsonl",
    ),
    ProblemRun(
        problem="minimum_vertex_cover",
        train_path="data/vertex_cover/vertex_cover100_p015_seed1234.jsonl",
        val_path="data/vertex_cover/vertex_cover100_p015_val_seed4321.jsonl",
    ),
    ProblemRun(
        problem="knapsack",
        train_path="data/knapsack/knapsack100_seed1234.jsonl",
        val_path="data/knapsack/knapsack100_val_seed4321.jsonl",
    ),
    ProblemRun(
        problem="orienteering",
        train_path="data/orienteering/orienteering50_seed1234.jsonl",
        val_path="data/orienteering/orienteering50_val_seed4321.jsonl",
    ),
)

MATRIX_RUNS: dict[MatrixName, tuple[ProblemRun, ...]] = {
    "supervised-smoke": SUPERVISED_SMOKE_RUNS,
    "supervised": SUPERVISED_RUNS,
    "rl": RL_RUNS,
    "module-test": SUPERVISED_SMOKE_RUNS,
}

MATRIX_DEFAULTS: dict[MatrixName, MatrixDefaults] = {
    "supervised-smoke": MatrixDefaults(
        epochs=1,
        steps_per_epoch="1",
        batch_size=8,
        output_root="outputs/smoke",
        progress_bar=False,
        checkpoint_every=99,
        log_every=1,
        save_best=False,
        save_last=False,
        keep_last_k=1,
        label_smoothing=0.0,
    ),
    "supervised": MatrixDefaults(
        epochs=PILOT_SCALE.epochs,
        steps_per_epoch=str(PILOT_SCALE.steps_per_epoch),
        batch_size=PILOT_SCALE.batch_size,
        output_root="outputs/matrix",
        progress_bar=BASE_TRAINER.progress_bar,
        checkpoint_every=BASE_TRAINER.checkpoint_every,
        log_every=BASE_TRAINER.log_every,
        save_best=BASE_TRAINER.save_best,
        save_last=BASE_TRAINER.save_last,
        keep_last_k=BASE_TRAINER.keep_last_k,
        label_smoothing=BASE_TRAINER.label_smoothing,
    ),
    "rl": MatrixDefaults(
        epochs=PILOT_SCALE.epochs,
        steps_per_epoch=str(PILOT_SCALE.steps_per_epoch),
        batch_size=PILOT_SCALE.batch_size,
        output_root="outputs/matrix",
        progress_bar=BASE_TRAINER.progress_bar,
        checkpoint_every=BASE_TRAINER.checkpoint_every,
        log_every=BASE_TRAINER.log_every,
        save_best=BASE_TRAINER.save_best,
        save_last=BASE_TRAINER.save_last,
        keep_last_k=BASE_TRAINER.keep_last_k,
        baseline=BASE_TRAINER.baseline,
    ),
    "module-test": MatrixDefaults(
        epochs=1,
        steps_per_epoch="1",
        batch_size=8,
        output_root="outputs/module_test",
        progress_bar=False,
        checkpoint_every=99,
        log_every=1,
        save_best=False,
        save_last=False,
        keep_last_k=1,
        label_smoothing=0.0,
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Build or execute the model/problem training matrix through "
            "src.main.train Hydra overrides."
        )
    )
    parser.add_argument(
        "--matrix",
        choices=MATRIX_NAMES,
        default="supervised-smoke",
        help=(
            "Training matrix preset. module-test runs one supervised smoke job "
            "per modular encoder/decoder architecture."
        ),
    )
    parser.add_argument(
        "--scale",
        choices=tuple(SCALES),
        default=PILOT_SCALE.name,
        help="Training scale profile. Defaults to pilot.",
    )
    parser.add_argument(
        "--models",
        help=(
            "Comma-separated model names. Defaults to the six modular comparison "
            "models."
        ),
    )
    parser.add_argument(
        "--problems",
        help="Comma-separated problem names. Defaults to every problem in the matrix.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the generated commands. Without this flag, commands are printed only.",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help=(
            "Generate missing smoke JSONL files for the selected matrix runs, then exit "
            "unless --execute is also set."
        ),
    )
    parser.add_argument(
        "--skip-prepare-data",
        action="store_true",
        help=(
            "Do not auto-generate smoke data before --execute. Use when datasets are "
            "already present."
        ),
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--steps-per-epoch",
        help="Hydra value for trainer.steps_per_epoch, for example 1, 125, or null.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--device")
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable tqdm progress bars in generated runs.",
    )
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--val-path",
        help="Validation JSONL path to use for every run. Defaults to Hydra null.",
    )
    namespace = parser.parse_args(argv)
    return RunnerArgs(
        matrix=cast(MatrixName, namespace.matrix),
        scale=namespace.scale,
        models=namespace.models,
        problems=namespace.problems,
        execute=bool(namespace.execute),
        prepare_data=bool(namespace.prepare_data),
        skip_prepare_data=bool(namespace.skip_prepare_data),
        epochs=namespace.epochs,
        steps_per_epoch=namespace.steps_per_epoch,
        batch_size=namespace.batch_size,
        output_root=namespace.output_root,
        device=namespace.device,
        progress_bar=namespace.progress_bar,
        checkpoint_every=namespace.checkpoint_every,
        log_every=namespace.log_every,
        num_workers=namespace.num_workers,
        learning_rate=namespace.learning_rate,
        val_path=namespace.val_path,
    )


def split_csv(raw: str | None, default: Iterable[str]) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return tuple(default)
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise SystemExit("Expected at least one comma-separated value.")
    return values


def validate_selected(
    selected: Sequence[str],
    allowed: Sequence[str],
    label: str,
) -> None:
    allowed_set = set(allowed)
    invalid = [item for item in selected if item not in allowed_set]
    if invalid:
        raise SystemExit(
            f"Unsupported {label}: {', '.join(invalid)}. "
            f"Allowed values: {', '.join(allowed)}"
        )


def runs_by_problem(runs: Sequence[ProblemRun]) -> dict[str, ProblemRun]:
    return {run.problem: run for run in runs}


def select_models(raw_models: str | None, matrix: MatrixName) -> tuple[str, ...]:
    models = split_csv(raw_models, DEFAULT_MATRIX_MODELS)
    if matrix == "module-test":
        validate_selected(models, DEFAULT_MATRIX_MODELS, "modular model")
    else:
        validate_selected(models, MODEL_NAMES, "model")
    return models


def select_problem_runs(
    matrix: MatrixName, raw_problems: str | None
) -> tuple[ProblemRun, ...]:
    run_lookup = runs_by_problem(MATRIX_RUNS[matrix])
    problems = split_csv(raw_problems, run_lookup.keys())
    validate_selected(problems, PROBLEM_NAMES, "problem")
    missing = [problem for problem in problems if problem not in run_lookup]
    if missing:
        raise SystemExit(
            f"Problems are not configured for {matrix}: {', '.join(missing)}. "
            f"Available values: {', '.join(run_lookup)}"
        )
    return tuple(run_lookup[problem] for problem in problems)


def smoke_dataset_requests(
    runs: Sequence[ProblemRun],
) -> tuple[SmokeDatasetRequest, ...]:
    return tuple(
        SmokeDatasetRequest(problem=run.problem, output_path=run.train_path)
        for run in runs
    )


def uses_smoke_data(matrix: MatrixName) -> bool:
    return matrix in ("supervised-smoke", "module-test")


def prepare_smoke_data_for_runs(runs: Sequence[ProblemRun]) -> None:
    requests = smoke_dataset_requests(runs)
    missing = missing_smoke_paths(requests)
    if not missing:
        print("Smoke datasets already present.")
        return
    print(f"Generating {len(missing)} missing smoke dataset(s)...")
    ensure_smoke_datasets(requests)


def validate_smoke_data_for_runs(runs: Sequence[ProblemRun]) -> None:
    missing = missing_smoke_paths(smoke_dataset_requests(runs))
    if not missing:
        return
    joined = "\n  ".join(missing)
    raise SystemExit(
        "Missing smoke dataset file(s):\n"
        f"  {joined}\n"
        "Generate them with:\n"
        "  uv run python -m src.main.experiments.generate_smoke_data\n"
        "or rerun with --prepare-data / omit --skip-prepare-data on --execute."
    )


def runs_for_data_prep(
    *,
    matrix: MatrixName,
    models: Sequence[str],
    runs: Sequence[ProblemRun],
) -> tuple[ProblemRun, ...]:
    if matrix == "module-test":
        return tuple(pair.run for pair in select_module_test_pairs(models, runs))
    return tuple(runs)


def select_module_test_pairs(
    models: Sequence[str],
    smoke_runs: Sequence[ProblemRun],
) -> tuple[ModuleTestPair, ...]:
    run_lookup = runs_by_problem(smoke_runs)
    pairs: list[ModuleTestPair] = []
    for model in models:
        if model not in MODULE_TEST_PROBLEM:
            raise SystemExit(
                f"Model {model} is not configured for module-test. "
                f"Allowed values: {', '.join(MODULE_TEST_PROBLEM)}"
            )
        problem = MODULE_TEST_PROBLEM[model]
        if problem not in run_lookup:
            raise SystemExit(
                f"Missing smoke dataset for module-test problem {problem}. "
                f"Available values: {', '.join(run_lookup)}"
            )
        pairs.append(ModuleTestPair(model=model, run=run_lookup[problem]))
    return tuple(pairs)


def resolve_settings(args: RunnerArgs) -> ResolvedSettings:
    defaults = MATRIX_DEFAULTS[args.matrix]
    scale = load_scale(args.scale)
    use_scale = args.matrix not in ("supervised-smoke", "module-test")
    return ResolvedSettings(
        scale=args.scale,
        epochs=(
            args.epochs
            if args.epochs is not None
            else scale.epochs
            if use_scale
            else defaults.epochs
        ),
        steps_per_epoch=(
            args.steps_per_epoch
            if args.steps_per_epoch is not None
            else str(scale.steps_per_epoch)
            if use_scale
            else defaults.steps_per_epoch
        ),
        batch_size=(
            args.batch_size
            if args.batch_size is not None
            else scale.batch_size
            if use_scale
            else defaults.batch_size
        ),
        output_root=(
            args.output_root if args.output_root is not None else defaults.output_root
        ),
        progress_bar=(
            args.progress_bar
            if args.progress_bar is not None
            else defaults.progress_bar
        ),
        checkpoint_every=(
            args.checkpoint_every
            if args.checkpoint_every is not None
            else defaults.checkpoint_every
        ),
        log_every=args.log_every if args.log_every is not None else defaults.log_every,
        save_best=defaults.save_best,
        save_last=defaults.save_last,
        keep_last_k=defaults.keep_last_k,
        label_smoothing=defaults.label_smoothing,
        baseline=defaults.baseline,
        device=args.device,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        val_path=args.val_path,
    )


def build_train_command(
    *,
    matrix: MatrixName,
    model: str,
    run: ProblemRun,
    settings: ResolvedSettings,
) -> list[str]:
    mode = "rl" if matrix == "rl" else "supervised"
    output_dir = f"{settings.output_root}/{model}/{run.problem}/{mode}"
    val_path = settings.val_path if settings.val_path is not None else run.val_path
    overrides = [
        f"scale={settings.scale}",
        f"problem={run.problem}",
        f"mode={mode}",
        f"model.name={model}",
        f"paths.train={run.train_path}",
        f"paths.val={val_path or 'null'}",
        f"paths.output_dir={output_dir}",
        f"data.batch_size={settings.batch_size}",
        f"trainer.epochs={settings.epochs}",
        f"trainer.steps_per_epoch={settings.steps_per_epoch}",
        f"trainer.progress_bar={hydra_bool(settings.progress_bar)}",
    ]
    if mode == "supervised":
        if run.target_algorithm is None:
            raise ValueError(f"Missing supervised target algorithm for {run.problem}.")
        overrides.append(f"data.target_algorithm={run.target_algorithm}")
    if settings.baseline is not None:
        overrides.append(f"trainer.baseline={settings.baseline}")
    if settings.checkpoint_every is not None:
        overrides.append(f"trainer.checkpoint_every={settings.checkpoint_every}")
    if settings.log_every is not None:
        overrides.append(f"trainer.log_every={settings.log_every}")
    if settings.save_best is not None:
        overrides.append(f"trainer.save_best={hydra_bool(settings.save_best)}")
    if settings.save_last is not None:
        overrides.append(f"trainer.save_last={hydra_bool(settings.save_last)}")
    if settings.keep_last_k is not None:
        overrides.append(f"trainer.keep_last_k={settings.keep_last_k}")
    if mode == "supervised" and settings.label_smoothing is not None:
        overrides.append(f"trainer.label_smoothing={settings.label_smoothing}")
    if settings.device is not None:
        overrides.append(f"device={settings.device}")
    if settings.num_workers is not None:
        overrides.append(f"data.num_workers={settings.num_workers}")
    if settings.learning_rate is not None:
        overrides.append(f"trainer.learning_rate={settings.learning_rate}")
    return ["uv", "run", "python", "-m", "src.main.train", *overrides]


def hydra_bool(value: bool) -> str:
    return "true" if value else "false"


def build_commands(
    *,
    args: RunnerArgs,
    models: Sequence[str],
    runs: Sequence[ProblemRun],
    settings: ResolvedSettings,
) -> list[list[str]]:
    if args.matrix == "module-test":
        pairs = select_module_test_pairs(models, runs)
        return [
            build_train_command(
                matrix=args.matrix,
                model=pair.model,
                run=pair.run,
                settings=settings,
            )
            for pair in pairs
        ]
    return [
        build_train_command(
            matrix=args.matrix,
            model=model,
            run=run,
            settings=settings,
        )
        for model in models
        for run in runs
    ]


def run_commands(commands: Sequence[Sequence[str]], *, execute: bool) -> None:
    action = "Running" if execute else "Dry run"
    print(f"{action} {len(commands)} command(s).")
    if not execute:
        print("Pass --execute to run them.")
    for index, command in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
        if execute:
            subprocess.run(command, check=True)


def print_module_test_plan(models: Sequence[str]) -> None:
    print("Module-test encoder/decoder plan:")
    for architecture in MODULAR_ARCHITECTURES:
        if architecture.name not in models:
            continue
        problem = MODULE_TEST_PROBLEM[architecture.name]
        print(
            f"  {architecture.name}: {architecture.encoder} + {architecture.decoder} "
            f"-> {problem} ({architecture.description})"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    models = select_models(args.models, args.matrix)
    runs = select_problem_runs(args.matrix, args.problems)
    settings = resolve_settings(args)
    if args.matrix == "module-test":
        print_module_test_plan(models)
    data_runs = runs_for_data_prep(matrix=args.matrix, models=models, runs=runs)
    should_prepare = args.prepare_data or (
        args.execute and uses_smoke_data(args.matrix) and not args.skip_prepare_data
    )
    if should_prepare:
        prepare_smoke_data_for_runs(data_runs)
    elif args.execute and uses_smoke_data(args.matrix):
        validate_smoke_data_for_runs(data_runs)
    if args.prepare_data and not args.execute:
        return 0
    commands = build_commands(
        args=args,
        models=models,
        runs=runs,
        settings=settings,
    )
    run_commands(commands, execute=args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
