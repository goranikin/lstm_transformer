import os

from omegaconf import DictConfig, OmegaConf

from configs.validation import config_to_dict, validate_config
from src.training.trainer import RLTrainer, SupervisedTrainer
from src.training.utils import (
    build_loader,
    build_model,
    count_trainable_parameters,
    default_target_algorithm,
    resolve_device,
    set_seed,
)
from src.training.wandb_logger import WandbLogger


def run_pipeline(cfg: DictConfig) -> None:
    """Run one training job from a Hydra config."""

    validate_config(cfg)

    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))

    target_algorithm = cfg.data.target_algorithm
    if cfg.mode == "supervised" and target_algorithm is None:
        target_algorithm = default_target_algorithm(cfg.problem)

    train_loader = build_loader(
        problem=cfg.problem,
        path=str(cfg.paths.train),
        batch_size=int(cfg.data.batch_size),
        target_algorithm=str(target_algorithm) if cfg.mode == "supervised" else None,
        shuffle=bool(cfg.data.shuffle),
        num_workers=int(cfg.data.num_workers),
    )

    val_loader = None
    if cfg.paths.val is not None:
        val_loader = build_loader(
            problem=cfg.problem,
            path=str(cfg.paths.val),
            batch_size=int(cfg.data.eval_batch_size),
            target_algorithm=str(target_algorithm) if cfg.mode == "supervised" else None,
            shuffle=False,
            num_workers=int(cfg.data.num_workers),
        )

    model = build_model(cfg.model.name, cfg.problem, cfg.model)
    parameter_count = count_trainable_parameters(model)
    print(
        f"model={cfg.model.name} problem={cfg.problem} mode={cfg.mode} "
        f"parameters={parameter_count} device={device}"
    )

    output_dir = cfg.paths.output_dir or os.path.join(
        "outputs",
        cfg.model.name,
        cfg.problem,
        cfg.mode,
    )
    OmegaConf.update(cfg, "paths.output_dir", output_dir, merge=True)

    run_name = cfg.wandb.name or f"{cfg.model.name}-{cfg.problem}-{cfg.mode}"
    wandb_logger = WandbLogger.from_config(
        config=cfg.wandb,
        run_config=config_to_dict(cfg),
        default_name=run_name,
        default_group=f"{cfg.model.name}/{cfg.problem}/{cfg.mode}",
        default_job_type=cfg.mode,
    )
    if wandb_logger.enabled and cfg.wandb.watch_model:
        wandb_logger.watch(
            model,
            log=cfg.wandb.watch_log,
            log_freq=cfg.wandb.watch_log_freq,
        )

    trainer_cls = SupervisedTrainer if cfg.mode == "supervised" else RLTrainer
    trainer = trainer_cls(
        model=model,
        train_loader=train_loader,
        cfg=cfg,
        device=device,
        val_loader=val_loader,
        wandb_logger=wandb_logger,
    )
    try:
        trainer.fit()
    finally:
        wandb_logger.finish()
