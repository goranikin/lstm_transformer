from collections.abc import Mapping
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field


class WandbConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    enabled: bool = False
    project: str = "lstm_vs_transformer"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: list[str] = Field(default_factory=list)
    mode: Literal["online", "offline", "disabled"] = "online"
    dir: str | None = None
    notes: str | None = None
    job_type: str | None = None
    watch_model: bool = False
    watch_log: Literal["gradients", "parameters", "all"] = "gradients"
    watch_log_freq: int = Field(default=1000, gt=0)
    log_checkpoints: bool = False


class WandbLogger:
    def __init__(self, wandb_module: Any | None = None) -> None:
        self._wandb = wandb_module

    @classmethod
    def disabled(cls) -> "WandbLogger":
        return cls()

    @classmethod
    def from_config(
        cls,
        config: WandbConfig,
        run_config: Mapping[str, Any],
        default_name: str,
        default_group: str,
        default_job_type: str,
    ) -> "WandbLogger":
        if not config.enabled:
            return cls.disabled()

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "WandB logging is enabled, but the 'wandb' package is not installed. "
                "Install project dependencies with `uv sync`."
            ) from exc

        init_kwargs: dict[str, Any] = {
            "project": config.project,
            "name": config.name or default_name,
            "group": config.group or default_group,
            "job_type": config.job_type or default_job_type,
            "tags": config.tags,
            "mode": config.mode,
            "config": dict(run_config),
        }
        if config.entity is not None:
            init_kwargs["entity"] = config.entity
        if config.dir is not None:
            init_kwargs["dir"] = config.dir
        if config.notes is not None:
            init_kwargs["notes"] = config.notes

        wandb.init(**init_kwargs)
        return cls(wandb)

    @property
    def enabled(self) -> bool:
        return self._wandb is not None

    def watch(
        self,
        model: torch.nn.Module,
        log: Literal["gradients", "parameters", "all"],
        log_freq: int,
    ) -> None:
        if self._wandb is None:
            return
        self._wandb.watch(model, log=log, log_freq=log_freq)

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if self._wandb is None:
            return
        self._wandb.log(dict(metrics), step=step)

    def save_file(self, path: str) -> None:
        if self._wandb is None:
            return
        self._wandb.save(path, policy="now")

    def finish(self) -> None:
        if self._wandb is None:
            return
        self._wandb.finish()
