import argparse
import shlex
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from configs.validation import SCALES, load_base_trainer, load_scale
from src.training.utils import MODEL_NAMES, PROBLEM_NAMES

MatrixName = Literal["supervised-smoke", "supervised", "rl"]

MATRIX_NAMES: tuple[MatrixName, ...] = ("supervised-smoke", "supervised", "rl")
DEFAULT_MODELS = (
    "modular_pn",
    "modular_am",
    "am_lstm_pointer",
    "am_gru_pointer",
    "am_lstm_subset",
    "am_sigmoid_subset",
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
class RunnerArgs:
    matrix: MatrixName
    scale: str
    models: str | None
    problems: str | None
    execute: bool
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
        target_algorithm="kamis",
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
        target_algorithm="kamis",
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
}


def parse_args(argv: Sequence[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Build or execute the model/problem training matrix through "
            "src.main.train Hydra overrides."
        )
    )
    parser.add_argument("--matrix", choices=MATRIX_NAMES, default="supervised-smoke")
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


def select_models(raw_models: str | None) -> tuple[str, ...]:
    models = split_csv(raw_models, DEFAULT_MODELS)
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


def resolve_settings(args: RunnerArgs) -> ResolvedSettings:
    defaults = MATRIX_DEFAULTS[args.matrix]
    scale = load_scale(args.scale)
    use_scale = args.matrix != "supervised-smoke"
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    models = select_models(args.models)
    runs = select_problem_runs(args.matrix, args.problems)
    settings = resolve_settings(args)
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
