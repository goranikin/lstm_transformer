from __future__ import annotations

from typing import Any

import wandb
from omegaconf import DictConfig, OmegaConf


def init_from_config(
    cfg: DictConfig,
    *,
    output_dir: str,
    run_name: str | None = None,
    default_tags: list[str] | None = None,
) -> bool:
    if not bool(cfg.wandb.enabled):
        return False
    tags = [str(tag) for tag in cfg.wandb.tags] if cfg.wandb.tags else (default_tags or [])
    wandb.init(
        project=str(cfg.wandb.project),
        entity=cfg.wandb.entity,
        name=run_name or cfg.wandb.name,
        group=cfg.wandb.group,
        tags=tags,
        mode=str(cfg.wandb.mode),
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=output_dir,
    )
    return True


def log(metrics: dict[str, Any], *, step: int | None = None) -> None:
    if wandb.run is None:
        return
    wandb.log(metrics, step=step)


def finish() -> None:
    if wandb.run is not None:
        wandb.finish()
