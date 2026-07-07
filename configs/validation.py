from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.constants import DECODER_KINDS, ENCODER_KINDS, PROBLEM_NAMES

CONFIG_DIR = Path(__file__).resolve().parent
SCALES = ("pilot", "medium")


def load_scale(name: str) -> DictConfig:
    if name not in SCALES:
        raise ValueError(f"Unsupported scale: {name}. Allowed: {', '.join(SCALES)}")
    return OmegaConf.load(CONFIG_DIR / "scale" / f"{name}.yaml")


def validate_config(cfg: DictConfig) -> None:
    if str(cfg.problem) not in PROBLEM_NAMES:
        raise ValueError(f"Unsupported problem: {cfg.problem}")
    if str(cfg.encoder) not in ENCODER_KINDS:
        raise ValueError(f"Unsupported encoder: {cfg.encoder}")
    if str(cfg.decoder) not in DECODER_KINDS:
        raise ValueError(f"Unsupported decoder: {cfg.decoder}")
    if str(cfg.mode) not in {"supervised", "rl"}:
        raise ValueError(f"Unsupported mode: {cfg.mode}")
    has_d_model = cfg.model.d_model is not None
    has_d_ff = cfg.model.d_ff is not None
    if has_d_model != has_d_ff:
        raise ValueError("model.d_model and model.d_ff must be provided together")
    if not bool(cfg.parameter_budget.enabled) and not (has_d_model and has_d_ff):
        raise ValueError(
            "model.d_model and model.d_ff are required when "
            "parameter_budget.enabled=false"
        )
    if has_d_model and int(cfg.model.d_model) % int(cfg.model.num_heads) != 0:
        raise ValueError("model.d_model must be divisible by model.num_heads")


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
