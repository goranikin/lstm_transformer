from typing import Any

import wandb
from omegaconf import DictConfig, OmegaConf


def default_run_name(
    *,
    problem: str,
    encoder: str,
    decoder: str,
    mode: str,
) -> str:
    return f"{problem}_{encoder}_{decoder}_{mode}"


def build_wandb_config(
    *,
    cfg: DictConfig,
    matched_params: dict[str, Any],
    model: Any,
    train_path: str,
    val_path: str | None,
    test_path: str | None,
    target_algorithm: str,
    output_dir: str,
) -> dict[str, Any]:
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    num_layers = int(matched_params.get("num_layers", cfg.model.num_layers))
    num_heads = int(matched_params.get("num_heads", cfg.model.num_heads))
    transformer_decoder_layers = int(
        matched_params.get(
            "transformer_decoder_layers", cfg.model.transformer_decoder_layers
        )
    )
    d_model = int(matched_params["d_model"])
    d_ff = int(matched_params["d_ff"])

    eval_batch_size = cfg.data.get("eval_batch_size")
    if eval_batch_size is None:
        eval_batch_size = cfg.data.batch_size

    config: dict[str, Any] = {
        "run": {
            "problem": str(cfg.problem),
            "encoder": str(cfg.encoder),
            "decoder": str(cfg.decoder),
            "mode": str(cfg.mode),
            "seed": int(cfg.seed),
            "device": str(cfg.device),
        },
        "data": {
            "root": str(cfg.data.root),
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path,
            "target_algorithm": target_algorithm,
            "batch_size": int(cfg.data.batch_size),
            "eval_batch_size": int(eval_batch_size),
            "num_workers": int(cfg.data.num_workers),
            "shuffle": bool(cfg.data.shuffle),
        },
        "model": {
            "input_dim": matched_params.get("input_dim"),
            "d_model": d_model,
            "d_ff": d_ff,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "transformer_decoder_layers": transformer_decoder_layers,
            "dropout": float(cfg.model.dropout),
            "tanh_clip": float(cfg.model.tanh_clip),
            "trainable_params": trainable_params,
            "total_params": total_params,
        },
        "trainer": OmegaConf.to_container(cfg.trainer, resolve=True),
        "paths": {
            "output_dir": output_dir,
            "output_root": str(cfg.paths.output_root),
        },
    }

    if bool(cfg.parameter_budget.enabled):
        config["parameter_budget"] = {
            "enabled": True,
            "path": str(cfg.parameter_budget.path),
            "strict": bool(cfg.parameter_budget.strict),
            "target_params_override": cfg.parameter_budget.target_params,
            "search": OmegaConf.to_container(cfg.parameter_budget.search, resolve=True),
            "base": {
                "d_model": matched_params.get("base_d_model"),
                "d_ff": matched_params.get("base_d_ff"),
                "params": matched_params.get("base_params"),
            },
            "matched": {
                "params": matched_params["matched_params"],
                "target_params": matched_params["target_params"],
                "delta": matched_params["delta"],
                "delta_pct": matched_params["delta_pct"],
            },
            "source": matched_params["source"],
            "command_args": matched_params.get("command_args"),
        }
    else:
        config["parameter_budget"] = {"enabled": False}

    return config


def init_from_config(
    cfg: DictConfig,
    *,
    output_dir: str,
    run_name: str | None = None,
    default_tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> bool:
    if not bool(cfg.wandb.enabled):
        return False
    tags = (
        [str(tag) for tag in cfg.wandb.tags] if cfg.wandb.tags else (default_tags or [])
    )
    wandb.init(
        project=str(cfg.wandb.project),
        entity=cfg.wandb.entity,
        name=run_name or cfg.wandb.name,
        group=cfg.wandb.group,
        tags=tags,
        mode=str(cfg.wandb.mode),
        config=config or {},
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
