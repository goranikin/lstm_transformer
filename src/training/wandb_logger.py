from collections.abc import Mapping
from typing import Any, Literal

import torch


class WandbLogger:
    def __init__(self, wandb_module: Any | None = None) -> None:
        self._wandb = wandb_module

    @classmethod
    def disabled(cls) -> "WandbLogger":
        return cls()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        run_config: Mapping[str, Any],
        default_name: str,
        default_group: str,
        default_job_type: str,
    ) -> "WandbLogger":
        if not config.get("enabled", False):
            return cls.disabled()

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "WandB logging is enabled, but the 'wandb' package is not installed. "
                "Install project dependencies with `uv sync`."
            ) from exc

        init_kwargs: dict[str, Any] = {
            "project": config.get("project", "lstm_vs_transformer"),
            "name": config.get("name") or default_name,
            "group": config.get("group") or default_group,
            "job_type": config.get("job_type") or default_job_type,
            "tags": list(config.get("tags") or []),
            "mode": config.get("mode", "online"),
            "config": dict(run_config),
        }
        if config.get("entity") is not None:
            init_kwargs["entity"] = config["entity"]
        if config.get("dir") is not None:
            init_kwargs["dir"] = config["dir"]
        if config.get("notes") is not None:
            init_kwargs["notes"] = config["notes"]

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
