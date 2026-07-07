import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.constants import TrainingMode
from src.model import NCOModel
from src.training.baselines import ExponentialRewardBaseline, RolloutRewardBaseline
from src.training.metrics import (
    EvaluationMetrics,
    aggregate_metrics,
    batch_metrics,
    wandb_metrics,
    wandb_rl_step_metrics,
    wandb_supervised_step_metrics,
)
from src.training.wandb_support import log as wandb_log
from src.utils import move_to_device, timer


@dataclass
class TrainingConfig:
    mode: TrainingMode
    epochs: int = 1
    steps_per_epoch: int | None = None
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    baseline: str = "rollout"
    baseline_alpha: float = 0.05
    baseline_warmup_epochs: int = 1
    exp_baseline_beta: float = 0.8
    log_every: int = 25
    progress_bar: bool = True
    output_dir: str = "outputs/src"
    save_checkpoints: bool = True
    wandb_log: bool = False
    wandb_train_eval_batches: int = 10


@dataclass
class TrainingResult:
    history: list[dict[str, Any]] = field(default_factory=list)
    training_time_sec: float = 0.0
    best_validation_objective: float | None = None


class Trainer:
    def __init__(
        self,
        *,
        model: NCOModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.exp_baseline = ExponentialRewardBaseline(config.exp_baseline_beta)
        self.rollout_baseline = RolloutRewardBaseline(
            device=device,
            alpha=config.baseline_alpha,
        )
        self.global_step = 0

    def fit(self) -> TrainingResult:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        result = TrainingResult()
        with timer() as elapsed:
            for epoch in range(self.config.epochs):
                if self.config.mode == "supervised":
                    train_loss = self._train_supervised_epoch(epoch)
                    row: dict[str, Any] = {"epoch": epoch + 1, "train_loss": train_loss}
                elif self.config.mode == "rl":
                    train_loss = self._train_rl_epoch(epoch)
                    row = {"epoch": epoch + 1, "train_policy_loss": train_loss}
                else:
                    raise ValueError(f"Unsupported training mode: {self.config.mode}")
                if self.val_loader is not None:
                    metrics = self.evaluate(self.val_loader)
                    row.update(metrics.to_dict("val"))
                    if (
                        result.best_validation_objective is None
                        or metrics.objective < result.best_validation_objective
                    ):
                        result.best_validation_objective = metrics.objective
                        if self.config.save_checkpoints:
                            self.save_checkpoint("best.pt", epoch + 1)
                    if self.config.mode == "rl" and self.config.baseline == "rollout":
                        updated = self.rollout_baseline.maybe_update(
                            self.model,
                            self.val_loader,
                            warmup_done=epoch + 1 >= self.config.baseline_warmup_epochs,
                        )
                        row["rollout_updated"] = updated
                result.history.append(row)
                self._write_history(result)
                if self.config.wandb_log:
                    self._log_epoch_metrics(row)
                    self._log_train_eval_metrics(epoch + 1)
                if self.config.save_checkpoints:
                    self.save_checkpoint("last.pt", epoch + 1)
        result.training_time_sec = elapsed["elapsed"]
        self._write_history(result)
        if self.config.wandb_log:
            wandb_log(
                {
                    "train/training_time_sec": result.training_time_sec,
                    "train/best_validation_objective": result.best_validation_objective,
                },
                step=self.global_step,
            )
        return result

    def _train_supervised_epoch(self, epoch: int) -> float:
        self.model.train()
        losses: list[float] = []
        batches = self._epoch_batches(self.train_loader)
        iterator = tqdm(
            batches,
            total=self.config.steps_per_epoch,
            disable=not self.config.progress_bar,
            desc=f"supervised {epoch + 1}/{self.config.epochs}",
        )
        for step, batch in enumerate(iterator, start=1):
            batch = move_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model.supervised_loss(batch)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.global_step += 1
            losses.append(float(loss.detach().item()))
            if step % self.config.log_every == 0:
                avg_loss = sum(losses) / len(losses)
                iterator.set_postfix(loss=avg_loss)
                if self.config.wandb_log:
                    with torch.no_grad():
                        output = self.model(batch, decode_type="greedy")
                        step_metrics = batch_metrics(
                            self.model.problem, batch, output, 0.0
                        )
                    wandb_log(
                        wandb_supervised_step_metrics(
                            loss=avg_loss,
                            metrics=step_metrics,
                            epoch=epoch + 1,
                        ),
                        step=self.global_step,
                    )
        return sum(losses) / max(len(losses), 1)

    def _train_rl_epoch(self, epoch: int) -> float:
        self.model.train()
        losses: list[float] = []
        if (
            self.config.baseline == "rollout"
            and self.rollout_baseline.baseline_model is None
        ):
            self.rollout_baseline.init_from(self.model)
        batches = self._epoch_batches(self.train_loader)
        iterator = tqdm(
            batches,
            total=self.config.steps_per_epoch,
            disable=not self.config.progress_bar,
            desc=f"rl {epoch + 1}/{self.config.epochs}",
        )
        for step, batch in enumerate(iterator, start=1):
            batch = move_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(batch, decode_type="sampling")
            if output.reward is None or output.log_probs is None:
                raise RuntimeError("RL requires reward and log_probs")
            baseline = self._baseline_value(output.reward, batch, epoch)
            advantage = output.reward - baseline
            loss = -(advantage.detach() * output.log_probs).mean()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.global_step += 1
            losses.append(float(loss.detach().item()))
            if step % self.config.log_every == 0:
                avg_loss = sum(losses) / len(losses)
                iterator.set_postfix(loss=avg_loss)
                if self.config.wandb_log:
                    step_metrics = batch_metrics(self.model.problem, batch, output, 0.0)
                    wandb_log(
                        wandb_rl_step_metrics(
                            policy_loss=avg_loss,
                            metrics=step_metrics,
                            reward=output.reward,
                            advantage=advantage,
                            baseline=baseline,
                            epoch=epoch + 1,
                        ),
                        step=self.global_step,
                    )
        return sum(losses) / max(len(losses), 1)

    def _baseline_value(
        self,
        reward: torch.Tensor,
        batch: dict[str, Any],
        epoch: int,
    ) -> torch.Tensor:
        if self.config.baseline == "exponential":
            return self.exp_baseline.evaluate(reward)
        if self.config.baseline != "rollout":
            raise ValueError(f"Unsupported baseline: {self.config.baseline}")
        if epoch < self.config.baseline_warmup_epochs:
            return self.exp_baseline.evaluate(reward)
        return self.rollout_baseline.evaluate_batch(batch)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EvaluationMetrics:
        self.model.eval()
        items = []
        for batch in loader:
            batch = move_to_device(batch, self.device)
            with timer() as elapsed:
                output = self.model(batch, decode_type="greedy")
            items.append(
                batch_metrics(self.model.problem, batch, output, elapsed["elapsed"])
            )
        return aggregate_metrics(items)

    @torch.no_grad()
    def _evaluate_subset(
        self, loader: DataLoader, *, max_batches: int
    ) -> EvaluationMetrics:
        self.model.eval()
        items = []
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = move_to_device(batch, self.device)
            with timer() as elapsed:
                output = self.model(batch, decode_type="greedy")
            items.append(
                batch_metrics(self.model.problem, batch, output, elapsed["elapsed"])
            )
        return aggregate_metrics(items)

    def _epoch_batches(self, loader: DataLoader):
        if self.config.steps_per_epoch is None:
            yield from loader
            return
        iterator = iter(loader)
        for _ in range(self.config.steps_per_epoch):
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(loader)
                yield next(iterator)

    def save_checkpoint(self, filename: str, epoch: int) -> str:
        path = os.path.join(self.config.output_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_config": asdict(self.config),
                "problem": self.model.problem_name,
                "encoder": self.model.encoder_kind,
                "decoder": self.model.decoder_kind,
            },
            path,
        )
        return path

    def _log_epoch_metrics(self, row: dict[str, Any]) -> None:
        metrics: dict[str, Any] = {"epoch": row["epoch"]}
        mode_prefix = "sl" if self.config.mode == "supervised" else "rl"
        val_key_map = {
            "val/gap": "val/optimal_gap",
            "val/gap_pct": "val/optimal_gap_pct",
        }
        for key, value in row.items():
            if key == "epoch":
                continue
            if key == "train_loss":
                metrics[f"train/{mode_prefix}/loss_epoch"] = value
            elif key == "train_policy_loss":
                metrics["train/rl/policy_loss_epoch"] = value
            elif key == "rollout_updated":
                metrics["train/rl/rollout_updated"] = float(value)
            elif key in val_key_map:
                metrics[val_key_map[key]] = value
            elif key.startswith("val/"):
                metrics[key] = value
            else:
                metrics[key] = value
        wandb_log(metrics, step=self.global_step)

    def _log_train_eval_metrics(self, epoch: int) -> None:
        was_training = self.model.training
        metrics = self._evaluate_subset(
            self.train_loader,
            max_batches=self.config.wandb_train_eval_batches,
        )
        if was_training:
            self.model.train()
        mode_prefix = "sl" if self.config.mode == "supervised" else "rl"
        wandb_log(
            {
                "epoch": epoch,
                **wandb_metrics(metrics, f"train/{mode_prefix}/eval"),
            },
            step=self.global_step,
        )

    def _write_history(self, result: TrainingResult) -> None:
        payload = {
            "training_time_sec": result.training_time_sec,
            "best_validation_objective": result.best_validation_objective,
            "history": result.history,
        }
        with open(
            os.path.join(self.config.output_dir, "history.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
