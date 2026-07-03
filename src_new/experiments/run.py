from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from src_new.constants import (
    DECODER_KINDS,
    DEFAULT_TARGET_ALGORITHM,
    ENCODER_KINDS,
    PROBLEM_NAMES,
)
from src_new.data import build_dataloader
from src_new.model import NCOModel
from src_new.problems import make_problem
from src_new.training import Trainer, TrainingConfig
from src_new.utils import move_to_device, resolve_device, set_seed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one src_new experiment.")
    parser.add_argument("--problem", choices=PROBLEM_NAMES, required=True)
    parser.add_argument("--encoder", choices=ENCODER_KINDS, required=True)
    parser.add_argument("--decoder", choices=DECODER_KINDS, required=True)
    parser.add_argument("--mode", choices=("supervised", "rl"), required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path")
    parser.add_argument("--test-path")
    parser.add_argument("--target-algorithm")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--baseline", choices=("rollout", "exponential"), default="rollout")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/src_new/run")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    target_algorithm = args.target_algorithm or DEFAULT_TARGET_ALGORITHM[args.problem]
    eval_batch_size = args.eval_batch_size or args.batch_size
    train_loader = build_dataloader(
        args.train_path,
        args.problem,
        batch_size=args.batch_size,
        target_algorithm=target_algorithm,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = (
        build_dataloader(
            args.val_path,
            args.problem,
            batch_size=eval_batch_size,
            target_algorithm=target_algorithm,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if args.val_path
        else None
    )
    test_loader = (
        build_dataloader(
            args.test_path,
            args.problem,
            batch_size=eval_batch_size,
            target_algorithm=target_algorithm,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if args.test_path
        else None
    )

    input_dim = infer_input_dim(train_loader, args.problem, device)
    model = NCOModel(
        problem=args.problem,
        encoder_kind=args.encoder,
        decoder_kind=args.decoder,
        input_dim=input_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    train_config = TrainingConfig(
        mode=args.mode,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        baseline=args.baseline,
        progress_bar=not args.no_progress,
        output_dir=args.output_dir,
        save_checkpoints=not args.no_checkpoints,
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
    )
    result = trainer.fit()
    payload = {
        "args": vars(args),
        "training_config": asdict(train_config),
        "training_time_sec": result.training_time_sec,
        "history": result.history,
    }
    if test_loader is not None:
        payload["test"] = trainer.evaluate(test_loader).to_dict("test")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "result.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def infer_input_dim(loader, problem_name: str, device: torch.device) -> int:
    batch = move_to_device(next(iter(loader)), device)
    problem = make_problem(problem_name)
    features, _, _ = problem.build_features(batch)
    return int(features.size(-1))


if __name__ == "__main__":
    raise SystemExit(main())
