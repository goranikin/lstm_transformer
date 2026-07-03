from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.constants import is_modular_model

CONFIG_DIR = Path(__file__).resolve().parent
SCALES = ("pilot", "medium")


def load_scale(name: str) -> DictConfig:
    if name not in SCALES:
        raise ValueError(f"Unsupported scale: {name}. Allowed: {', '.join(SCALES)}")
    return OmegaConf.load(CONFIG_DIR / "scale" / f"{name}.yaml")


def load_base_trainer() -> DictConfig:
    return OmegaConf.load(CONFIG_DIR / "base.yaml").trainer


def default_am_config() -> DictConfig:
    return OmegaConf.create(
        {
            "d_h": 128,
            "n_layers": 3,
            "n_heads": 8,
            "d_ff": 512,
            "tanh_clip": 10.0,
            "normalization": "batch",
        }
    )


def validate_am_config(config: DictConfig) -> None:
    d_h = int(config.d_h)
    n_heads = int(config.n_heads)
    if d_h % n_heads != 0:
        raise ValueError("model.am.d_h must be divisible by model.am.n_heads")


def validate_config(cfg: DictConfig) -> None:
    if is_modular_model(cfg.model.name):
        validate_am_config(cfg.model.am)

    if cfg.trainer.optimizer not in {"adam", "sgd"}:
        raise ValueError(f"Unsupported optimizer: {cfg.trainer.optimizer}")

    if cfg.mode == "rl" and cfg.trainer.baseline not in {"rollout", "exponential"}:
        raise ValueError(f"Unsupported baseline: {cfg.trainer.baseline}")


def config_to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
