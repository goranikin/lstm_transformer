from __future__ import annotations

import argparse
import shlex
import subprocess
from collections.abc import Iterable, Sequence

from src_new.constants import (
    DECODER_KINDS,
    DEFAULT_SEEDS,
    DEFAULT_TARGET_ALGORITHM,
    MATRIX_ENCODERS,
    PROBLEM_FILE_PREFIX,
    PROBLEM_NAMES,
    PROBLEM_PATH_DIR,
)


STAGES: dict[str, tuple[str, ...]] = {
    "all": PROBLEM_NAMES,
    "routing": ("tsp", "cvrp"),
    "subset": ("knapsack", "mis", "max_clique", "vertex_cover"),
    "hybrid": ("orienteering",),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand the src_new architecture/problem/mode/seed matrix."
    )
    parser.add_argument("--stage", choices=tuple(STAGES), default="all")
    parser.add_argument("--problems", help="Comma-separated override.")
    parser.add_argument("--encoders", default=",".join(MATRIX_ENCODERS))
    parser.add_argument("--decoders", default=",".join(DECODER_KINDS))
    parser.add_argument("--modes", default="supervised,rl")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs/src_new/matrix")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--skip-sigmoid-routing",
        action="store_true",
        help="Skip weaker sigmoid baselines for TSP/CVRP.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    problems = tuple(split_csv(args.problems, STAGES[args.stage]))
    encoders = tuple(split_csv(args.encoders, MATRIX_ENCODERS))
    decoders = tuple(split_csv(args.decoders, DECODER_KINDS))
    modes = tuple(split_csv(args.modes, ("supervised", "rl")))
    seeds = tuple(int(seed) for seed in split_csv(args.seeds, tuple(map(str, DEFAULT_SEEDS))))
    commands = build_commands(
        problems=problems,
        encoders=encoders,
        decoders=decoders,
        modes=modes,
        seeds=seeds,
        data_root=args.data_root,
        output_root=args.output_root,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        num_workers=args.num_workers,
        skip_sigmoid_routing=args.skip_sigmoid_routing,
    )
    action = "Running" if args.execute else "Dry run"
    print(f"{action} {len(commands)} command(s).")
    for index, command in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
        if args.execute:
            subprocess.run(command, check=True)
    return 0


def build_commands(
    *,
    problems: Sequence[str],
    encoders: Sequence[str],
    decoders: Sequence[str],
    modes: Sequence[str],
    seeds: Sequence[int],
    data_root: str,
    output_root: str,
    epochs: int,
    steps_per_epoch: int | None,
    batch_size: int,
    learning_rate: float,
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
                        paths = problem_paths(data_root, problem)
                        output_dir = (
                            f"{output_root}/seed_{seed}/{mode}/"
                            f"{problem}/{encoder}/{decoder}"
                        )
                        command = [
                            "uv",
                            "run",
                            "python",
                            "-m",
                            "src_new.experiments.run",
                            "--problem",
                            problem,
                            "--encoder",
                            encoder,
                            "--decoder",
                            decoder,
                            "--mode",
                            mode,
                            "--train-path",
                            paths["train"],
                            "--val-path",
                            paths["val"],
                            "--test-path",
                            paths["test"],
                            "--target-algorithm",
                            DEFAULT_TARGET_ALGORITHM[problem],
                            "--seed",
                            str(seed),
                            "--epochs",
                            str(epochs),
                            "--batch-size",
                            str(batch_size),
                            "--learning-rate",
                            str(learning_rate),
                            "--device",
                            device,
                            "--num-workers",
                            str(num_workers),
                            "--output-dir",
                            output_dir,
                        ]
                        if steps_per_epoch is not None:
                            command.extend(["--steps-per-epoch", str(steps_per_epoch)])
                        commands.append(command)
    return commands


def problem_paths(data_root: str, problem: str) -> dict[str, str]:
    directory = PROBLEM_PATH_DIR[problem]
    prefix = PROBLEM_FILE_PREFIX[problem]
    return {
        "train": f"{data_root}/{directory}/{prefix}_train_64000_seed1234.jsonl",
        "val": f"{data_root}/{directory}/{prefix}_val_10000_seed4321.jsonl",
        "test": f"{data_root}/{directory}/{prefix}_test_10000_seed9999.jsonl",
    }


def split_csv(raw: str | None, default: Iterable[str]) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return tuple(default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


if __name__ == "__main__":
    raise SystemExit(main())
