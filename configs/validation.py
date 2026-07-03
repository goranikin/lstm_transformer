from omegaconf import DictConfig, OmegaConf


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


def default_pn_config() -> DictConfig:
    return OmegaConf.create(
        {
            "hidden_size": 256,
            "num_layers": 1,
            "dropout": 0.0,
            "tanh_clip": 10.0,
            "n_glimpses": 1,
            "mask_inner": True,
            "mask_logits": True,
        }
    )


def validate_am_config(config: DictConfig) -> None:
    d_h = int(config.d_h)
    n_heads = int(config.n_heads)
    if d_h % n_heads != 0:
        raise ValueError("model.am.d_h must be divisible by model.am.n_heads")


def validate_config(cfg: DictConfig) -> None:
    if cfg.model.name in {
        "am",
        "modified_am",
        "modular_am",
        "modular_pn",
        "am_lstm_pointer",
        "am_gru_pointer",
        "am_lstm_subset",
        "am_sigmoid_subset",
    }:
        validate_am_config(cfg.model.am)

    if cfg.trainer.optimizer not in {"adam", "sgd"}:
        raise ValueError(f"Unsupported optimizer: {cfg.trainer.optimizer}")

    if cfg.mode == "rl" and cfg.trainer.baseline not in {"rollout", "exponential"}:
        raise ValueError(f"Unsupported baseline: {cfg.trainer.baseline}")


def config_to_dict(cfg: DictConfig):
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
