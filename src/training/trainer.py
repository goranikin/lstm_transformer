import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.baselines.baseline import (
    ExponentialBaseline,
    RolloutBaseline,
)
from src.models.utils import set_decode_type_if_supported
from src.training.utils import move_batch_to_device
from src.training.wandb_logger import WandbConfig, WandbLogger

ProblemName = Literal[
    "tsp",
    "cvrp",
    "mis",
    "maximum_clique",
    "minimum_vertex_cover",
    "knapsack",
    "orienteering",
]
OptimizerName = Literal["adam", "sgd"]
BaselineName = Literal["rollout", "exponential"]


@dataclass(frozen=True)
class SolutionStats:
    solution_value: float
    exact_value: float
    gap: float
    gap_pct: float


@dataclass(frozen=True)
class SupervisedEpochStats:
    loss: float
    solution: SolutionStats


class SolutionStatsAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.solution_value = 0.0
        self.exact_value = 0.0
        self.gap = 0.0
        self.gap_pct = 0.0

    def update(self, stats: SolutionStats, count: int) -> None:
        self.count += count
        self.solution_value += stats.solution_value * count
        self.exact_value += stats.exact_value * count
        self.gap += stats.gap * count
        self.gap_pct += stats.gap_pct * count

    def average(self) -> SolutionStats:
        count = max(self.count, 1)
        return SolutionStats(
            solution_value=self.solution_value / count,
            exact_value=self.exact_value / count,
            gap=self.gap / count,
            gap_pct=self.gap_pct / count,
        )


class TrainerConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    problem: ProblemName
    output_dir: str = "outputs"
    n_epochs: int = Field(default=100, gt=0)
    steps_per_epoch: int | None = Field(default=None, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    learning_rate_decay: float = Field(default=1.0, gt=0, le=1.0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    log_every: int = Field(default=25, gt=0)
    checkpoint_every: int = Field(default=5, gt=0)
    optimizer: OptimizerName = "adam"
    baseline: BaselineName = "rollout"
    baseline_alpha: float = Field(default=0.05, ge=0, le=1)
    baseline_warmup_epochs: int = Field(default=1, ge=0)
    exp_baseline_beta: float = Field(default=0.8, ge=0, le=1)
    save_best: bool = True
    save_last: bool = True
    keep_last_k: int = Field(default=3, ge=0)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    progress_bar: bool = True


class SupervisedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        config: TrainerConfig,
        device: torch.device,
        val_loader: DataLoader | None = None,
        wandb_logger: WandbLogger | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.optimizer = self._build_optimizer()
        self.wandb_logger = wandb_logger or WandbLogger.disabled()
        self.global_step = 0
        self.best_score: float | None = None

    def fit(self) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        with self._progress(
            range(self.config.n_epochs),
            desc="epochs",
            total=self.config.n_epochs,
            unit="epoch",
            leave=True,
        ) as epochs:
            for epoch in epochs:
                if self.config.learning_rate_decay != 1.0:
                    self._set_learning_rate(
                        self.config.learning_rate
                        * (self.config.learning_rate_decay**epoch)
                    )
                train_stats = self.train_epoch_stats(epoch)
                message = (
                    f"epoch={epoch + 1} "
                    f"train_loss={train_stats.loss:.6f} "
                    f"train_solution={train_stats.solution.solution_value:.6f} "
                    f"train_exact={train_stats.solution.exact_value:.6f} "
                    f"train_gap={train_stats.solution.gap:.6f} "
                    f"train_gap_pct={train_stats.solution.gap_pct:.4f}"
                )
                postfix: dict[str, str] = {
                    "train_loss": f"{train_stats.loss:.6f}",
                    "train_sol": f"{train_stats.solution.solution_value:.6f}",
                    "train_exact": f"{train_stats.solution.exact_value:.6f}",
                    "train_gap%": f"{train_stats.solution.gap_pct:.2f}",
                    "lr": f"{self._current_learning_rate():.2e}",
                }
                metrics: dict[str, float | int] = {
                    "epoch": epoch + 1,
                    "train/loss_epoch": train_stats.loss,
                    "train/solution_value_epoch": train_stats.solution.solution_value,
                    "train/exact_value_epoch": train_stats.solution.exact_value,
                    "train/gap_epoch": train_stats.solution.gap,
                    "train/gap_pct_epoch": train_stats.solution.gap_pct,
                    "optimizer/learning_rate": self._current_learning_rate(),
                }
                if self.val_loader is not None:
                    val_stats = self.evaluate()
                    message += (
                        f" val_loss={val_stats.loss:.6f}"
                        f" val_solution={val_stats.solution.solution_value:.6f}"
                        f" val_exact={val_stats.solution.exact_value:.6f}"
                        f" val_gap={val_stats.solution.gap:.6f}"
                        f" val_gap_pct={val_stats.solution.gap_pct:.4f}"
                    )
                    postfix["val_loss"] = f"{val_stats.loss:.6f}"
                    postfix["val_sol"] = f"{val_stats.solution.solution_value:.6f}"
                    postfix["val_gap%"] = f"{val_stats.solution.gap_pct:.2f}"
                    metrics["val/loss"] = val_stats.loss
                    metrics["val/solution_value"] = val_stats.solution.solution_value
                    metrics["val/exact_value"] = val_stats.solution.exact_value
                    metrics["val/gap"] = val_stats.solution.gap
                    metrics["val/gap_pct"] = val_stats.solution.gap_pct
                epochs.set_postfix(postfix, refresh=False)
                self._log(message)
                self.wandb_logger.log(metrics, step=self.global_step)
                checkpoint_score = (
                    val_stats.loss if self.val_loader is not None else train_stats.loss
                )
                self._save_last_and_best(epoch + 1, checkpoint_score)
                if (epoch + 1) % self.config.checkpoint_every == 0:
                    self.save_checkpoint(epoch + 1)
                    self._prune_epoch_checkpoints()

    def train_epoch(self, epoch: int) -> float:
        return self.train_epoch_stats(epoch).loss

    def train_epoch_stats(self, epoch: int) -> SupervisedEpochStats:
        self.model.train()
        total_loss = 0.0
        solution_totals = SolutionStatsAccumulator()
        steps = 0
        with self._progress(
            self._epoch_batches(self.train_loader),
            desc=f"train {epoch + 1}/{self.config.n_epochs}",
            total=self._train_total(),
            unit="batch",
            leave=False,
        ) as batches:
            for step, batch in enumerate(batches, start=1):
                batch = move_batch_to_device(batch, self.device)
                self.optimizer.zero_grad()
                loss = self._supervised_loss(batch)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                loss_value = float(loss.detach().item())
                solution_stats = self._supervised_solution_stats(batch)
                batch_size = self._batch_size(batch)
                solution_totals.update(solution_stats, batch_size)
                total_loss += loss_value
                steps += 1
                self.global_step += 1
                loss_avg = total_loss / steps
                solution_avg = solution_totals.average()
                batches.set_postfix(
                    loss=f"{loss_value:.6f}",
                    loss_avg=f"{loss_avg:.6f}",
                    sol=f"{solution_avg.solution_value:.6f}",
                    exact=f"{solution_avg.exact_value:.6f}",
                    gap=f"{solution_avg.gap:.6f}",
                    gap_pct=f"{solution_avg.gap_pct:.2f}",
                    lr=f"{self._current_learning_rate():.2e}",
                    refresh=False,
                )
                if step % self.config.log_every == 0:
                    self._log(
                        f"epoch={epoch + 1} step={step} "
                        f"loss={loss_value:.6f} loss_avg={loss_avg:.6f} "
                        f"solution={solution_avg.solution_value:.6f} "
                        f"exact={solution_avg.exact_value:.6f} "
                        f"gap={solution_avg.gap:.6f} "
                        f"gap_pct={solution_avg.gap_pct:.4f}"
                    )
                    self.wandb_logger.log(
                        {
                            "epoch": epoch + 1,
                            "train/step": step,
                            "train/loss": loss_value,
                            "train/loss_avg": loss_avg,
                            "train/solution_value": solution_stats.solution_value,
                            "train/exact_value": solution_stats.exact_value,
                            "train/gap": solution_stats.gap,
                            "train/gap_pct": solution_stats.gap_pct,
                            "train/solution_value_avg": solution_avg.solution_value,
                            "train/exact_value_avg": solution_avg.exact_value,
                            "train/gap_avg": solution_avg.gap,
                            "train/gap_pct_avg": solution_avg.gap_pct,
                            "optimizer/learning_rate": self._current_learning_rate(),
                        },
                        step=self.global_step,
                    )
        return SupervisedEpochStats(
            loss=total_loss / max(steps, 1),
            solution=solution_totals.average(),
        )

    @torch.no_grad()
    def evaluate(self) -> SupervisedEpochStats:
        if self.val_loader is None:
            raise ValueError("val_loader is not configured")
        self.model.eval()
        total = 0.0
        count = 0
        solution_totals = SolutionStatsAccumulator()
        for batch in self._validation_batches(desc="val supervised"):
            loss = self._supervised_loss(batch)
            batch_size = self._batch_size(batch)
            solution_totals.update(
                self._supervised_solution_stats(batch),
                batch_size,
            )
            total += float(loss.item()) * batch_size
            count += batch_size
        return SupervisedEpochStats(
            loss=total / max(count, 1),
            solution=solution_totals.average(),
        )

    @torch.no_grad()
    def evaluate_loss(self) -> float:
        return self.evaluate().loss

    @torch.no_grad()
    def _supervised_solution_stats(
        self,
        batch: dict[str, torch.Tensor],
    ) -> SolutionStats:
        model_values = self._greedy_solution_values(batch)
        exact_values = self._exact_solution_values(batch).to(
            device=model_values.device,
            dtype=model_values.dtype,
        )

        objective_sense = self._objective_sense()
        if objective_sense == "min":
            gap = model_values - exact_values
        elif objective_sense == "max":
            gap = exact_values - model_values
        else:
            raise ValueError(f"Unsupported objective sense: {objective_sense}")

        gap_pct = 100.0 * gap / exact_values.abs().clamp_min(1e-12)
        return SolutionStats(
            solution_value=float(model_values.mean().item()),
            exact_value=float(exact_values.mean().item()),
            gap=float(gap.mean().item()),
            gap_pct=float(gap_pct.mean().item()),
        )

    @torch.no_grad()
    def _greedy_solution_values(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        was_training = self.model.training
        previous_decode_type = getattr(self.model, "decode_type", None)
        self.model.eval()
        set_decode_type_if_supported(self.model, "greedy")

        try:
            cost, _, solution = self.model(
                batch,
                problem=self.config.problem,
                return_pi=True,
            )
            solution_value = getattr(self.model, "solution_value", None)
            if callable(solution_value):
                return cast(Callable[..., torch.Tensor], solution_value)(
                    batch,
                    solution,
                    problem=self.config.problem,
                ).detach()
            if self.config.problem == "tsp":
                return cost.detach()
            if self.config.problem == "mis":
                if isinstance(solution, torch.Tensor):
                    return solution.detach().to(dtype=cost.dtype).sum(dim=1)
                return -cost.detach()
            raise ValueError(f"Unsupported problem: {self.config.problem}")
        finally:
            if isinstance(previous_decode_type, str):
                set_decode_type_if_supported(self.model, previous_decode_type)
            self.model.train(was_training)

    def _exact_solution_values(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        target_value = getattr(self.model, "target_value", None)
        if callable(target_value):
            return cast(Callable[..., torch.Tensor], target_value)(
                batch,
                problem=self.config.problem,
            ).detach()
        if self.config.problem == "tsp":
            target_cost = batch.get("target_cost")
            if isinstance(target_cost, torch.Tensor):
                return target_cost.detach()
            loc = self._require_tensor(batch, "loc")
            target_tour = self._require_tensor(batch, "target_tour").long()
            return self._tsp_cost(loc, target_tour).detach()

        if self.config.problem == "mis":
            target_size = batch.get("target_size")
            if isinstance(target_size, torch.Tensor):
                return target_size.detach().float()
            target_set = self._require_tensor(batch, "target_set")
            return target_set.detach().float().sum(dim=1)

        raise ValueError(f"Unsupported problem: {self.config.problem}")

    def _objective_sense(self) -> str:
        objective_sense = getattr(self.model, "objective_sense", None)
        if callable(objective_sense):
            return str(cast(Callable[..., str], objective_sense)(self.config.problem))
        if self.config.problem == "tsp":
            return "min"
        if self.config.problem == "mis":
            return "max"
        raise ValueError(f"Unsupported problem: {self.config.problem}")

    @staticmethod
    def _tsp_cost(loc: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
        ordered = loc.gather(1, tour.unsqueeze(-1).expand(-1, -1, loc.size(-1)))
        return (ordered[:, 1:] - ordered[:, :-1]).norm(p=2, dim=-1).sum(dim=1) + (
            ordered[:, 0] - ordered[:, -1]
        ).norm(p=2, dim=-1)

    def save_checkpoint(self, epoch: int, filename: str | None = None) -> str:
        path = os.path.join(
            self.config.output_dir,
            filename or f"epoch_{epoch:03d}.pt",
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.model_dump(),
            },
            path,
        )
        if self.config.wandb.log_checkpoints:
            self.wandb_logger.save_file(path)
        return path

    def _save_last_and_best(self, epoch: int, score: float) -> None:
        if self.config.save_last:
            self.save_checkpoint(epoch, "last.pt")
        if self.config.save_best and self._is_best_score(score):
            self.best_score = score
            self.save_checkpoint(epoch, "best.pt")

    def _is_best_score(self, score: float) -> bool:
        return self.best_score is None or score < self.best_score

    def _prune_epoch_checkpoints(self) -> None:
        keep = self.config.keep_last_k
        entries = []
        for filename in os.listdir(self.config.output_dir):
            if not (filename.startswith("epoch_") and filename.endswith(".pt")):
                continue
            raw_epoch = filename.removeprefix("epoch_").removesuffix(".pt")
            try:
                epoch = int(raw_epoch)
            except ValueError:
                continue
            entries.append((epoch, filename))
        entries.sort()
        stale_entries = entries[:-keep] if keep > 0 else entries
        for _, filename in stale_entries:
            os.remove(os.path.join(self.config.output_dir, filename))

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=self.config.learning_rate
            )
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=self.config.learning_rate
            )
        raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def _set_learning_rate(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def _current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _supervised_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        supervised_loss = getattr(self.model, "supervised_loss", None)
        if not callable(supervised_loss):
            raise TypeError("model must define callable supervised_loss")
        return cast(Callable[..., torch.Tensor], supervised_loss)(
            batch=batch,
            problem=self.config.problem,
            label_smoothing=self.config.label_smoothing,
        )

    def _epoch_batches(
        self,
        loader: DataLoader,
    ) -> Iterable[dict[str, torch.Tensor]]:
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

    def _validation_batches(
        self,
        desc: str,
    ) -> Iterable[dict[str, torch.Tensor]]:
        if self.val_loader is None:
            return
        with self._progress(
            self.val_loader,
            desc=desc,
            total=self._loader_total(self.val_loader),
            unit="batch",
            leave=False,
        ) as batches:
            for batch in batches:
                yield move_batch_to_device(batch, self.device)

    def _progress(
        self,
        iterable: Iterable,
        *,
        desc: str,
        total: int | None,
        unit: str,
        leave: bool,
    ) -> tqdm:
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            unit=unit,
            leave=leave,
            dynamic_ncols=True,
            disable=not self.config.progress_bar,
        )

    def _train_total(self) -> int | None:
        if self.config.steps_per_epoch is not None:
            return self.config.steps_per_epoch
        return self._loader_total(self.train_loader)

    @staticmethod
    def _loader_total(loader: DataLoader) -> int | None:
        try:
            return len(loader)
        except TypeError:
            return None

    def _log(self, message: str) -> None:
        if self.config.progress_bar:
            tqdm.write(message)
            return
        print(message)

    @staticmethod
    def _batch_size(batch: dict[str, torch.Tensor]) -> int:
        for key in ("loc", "adjacency", "item_features", "coordinates", "weights"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                return int(value.size(0))
        raise ValueError("Cannot infer batch size")

    @staticmethod
    def _require_tensor(
        batch: dict[str, torch.Tensor],
        key: str,
    ) -> torch.Tensor:
        value = batch.get(key)
        if value is None:
            raise ValueError(f"Missing batch['{key}']")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch['{key}'] must be a torch.Tensor")
        return value


class RLTrainer(SupervisedTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        config: TrainerConfig,
        device: torch.device,
        val_loader: DataLoader | None = None,
        wandb_logger: WandbLogger | None = None,
    ) -> None:
        super().__init__(
            model,
            train_loader,
            config,
            device,
            val_loader,
            wandb_logger=wandb_logger,
        )
        self.exp_baseline = ExponentialBaseline(beta=config.exp_baseline_beta)
        self.rollout_baseline = RolloutBaseline(
            problem=config.problem,
            device=device,
            alpha=config.baseline_alpha,
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        set_decode_type_if_supported(self.model, "sampling")
        total_cost = 0.0
        steps = 0
        with self._progress(
            self._epoch_batches(self.train_loader),
            desc=f"train {epoch + 1}/{self.config.n_epochs}",
            total=self._train_total(),
            unit="batch",
            leave=False,
        ) as batches:
            for step, batch in enumerate(batches, start=1):
                batch = move_batch_to_device(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                cost, log_likelihood = self.model(batch, problem=self.config.problem)
                baseline = self._baseline_value(cost, batch, epoch)
                loss = ((cost - baseline).detach() * log_likelihood).mean()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                cost_value = float(cost.detach().mean().item())
                loss_value = float(loss.detach().item())
                baseline_value = float(baseline.detach().mean().item())
                total_cost += cost_value
                steps += 1
                self.global_step += 1
                cost_avg = total_cost / steps
                batches.set_postfix(
                    cost=f"{cost_value:.6f}",
                    avg=f"{cost_avg:.6f}",
                    loss=f"{loss_value:.6f}",
                    baseline=f"{baseline_value:.6f}",
                    lr=f"{self._current_learning_rate():.2e}",
                    refresh=False,
                )
                if step % self.config.log_every == 0:
                    self._log(
                        f"epoch={epoch + 1} step={step} "
                        f"cost={cost_avg:.6f} loss={loss_value:.6f}"
                    )
                    self.wandb_logger.log(
                        {
                            "epoch": epoch + 1,
                            "train/step": step,
                            "train/cost": cost_value,
                            "train/cost_avg": cost_avg,
                            "train/loss": loss_value,
                            "train/baseline": baseline_value,
                            "optimizer/learning_rate": self._current_learning_rate(),
                        },
                        step=self.global_step,
                    )
        return total_cost / max(steps, 1)

    def fit(self) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        with self._progress(
            range(self.config.n_epochs),
            desc="epochs",
            total=self.config.n_epochs,
            unit="epoch",
            leave=True,
        ) as epochs:
            for epoch in epochs:
                if self.config.learning_rate_decay != 1.0:
                    self._set_learning_rate(
                        self.config.learning_rate
                        * (self.config.learning_rate_decay**epoch)
                    )
                train_cost = self.train_epoch(epoch)
                message = f"epoch={epoch + 1} train_cost={train_cost:.6f}"
                postfix: dict[str, str] = {
                    "train_cost": f"{train_cost:.6f}",
                    "lr": f"{self._current_learning_rate():.2e}",
                }
                metrics: dict[str, float | int | bool] = {
                    "epoch": epoch + 1,
                    "train/cost_epoch": train_cost,
                    "optimizer/learning_rate": self._current_learning_rate(),
                }
                if self.val_loader is not None:
                    val_cost = self.evaluate_cost()
                    message += f" val_greedy_cost={val_cost:.6f}"
                    postfix["val_greedy_cost"] = f"{val_cost:.6f}"
                    metrics["val/greedy_cost"] = val_cost
                    if self.config.baseline == "rollout":
                        updated = self.rollout_baseline.maybe_update(
                            self.model,
                            self._validation_batches(desc="rollout baseline"),
                            epoch,
                            self.config.baseline_warmup_epochs,
                        )
                        message += f" rollout_updated={updated}"
                        postfix["rollout_updated"] = str(updated)
                        metrics["rollout/updated"] = updated
                epochs.set_postfix(postfix, refresh=False)
                self._log(message)
                self.wandb_logger.log(metrics, step=self.global_step)
                checkpoint_score = (
                    val_cost if self.val_loader is not None else train_cost
                )
                self._save_last_and_best(epoch + 1, checkpoint_score)
                if (epoch + 1) % self.config.checkpoint_every == 0:
                    self.save_checkpoint(epoch + 1)
                    self._prune_epoch_checkpoints()

    @torch.no_grad()
    def evaluate_cost(self) -> float:
        if self.val_loader is None:
            raise ValueError("val_loader is not configured")
        self.model.eval()
        set_decode_type_if_supported(self.model, "greedy")
        total = 0.0
        count = 0
        for batch in self._validation_batches(desc="val greedy"):
            cost, _ = self.model(batch, problem=self.config.problem)
            batch_size = self._batch_size(batch)
            total += float(cost.sum().item())
            count += batch_size
        set_decode_type_if_supported(self.model, "sampling")
        return total / max(count, 1)

    def _baseline_value(
        self,
        cost: torch.Tensor,
        batch: dict[str, torch.Tensor],
        epoch: int,
    ) -> torch.Tensor:
        if self.config.baseline == "exponential":
            return self.exp_baseline.eval(cost)
        if self.config.baseline != "rollout":
            raise ValueError(f"Unsupported baseline: {self.config.baseline}")
        if epoch < self.config.baseline_warmup_epochs:
            return self.exp_baseline.eval(cost)
        if self.rollout_baseline.baseline_model is None:
            self.rollout_baseline.init_from(self.model)
        return self.rollout_baseline.eval_batch(self.model, batch)
