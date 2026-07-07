import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from src.constants import (
    DECODER_KINDS,
    DEFAULT_TARGET_ALGORITHM,
    ENCODER_KINDS,
    MATRIX_ENCODERS,
    PROBLEM_NAMES,
    DecoderKind,
    EncoderKind,
    ProblemName,
)
from src.data import build_dataloader
from src.experiments.parameter_comparison import (
    INPUT_DIM_BY_PROBLEM,
    ParameterComparisonSettings,
    base_parameter_count,
    find_closest_budget,
    resolve_target,
)
from src.model import NCOModel
from src.paths import problem_dataset_path, resolve_user_path
from src.training import Trainer, TrainingConfig
from src.training.metrics import wandb_metrics
from src.training.wandb_support import build_wandb_config, default_run_name
from src.training.wandb_support import finish as finish_wandb
from src.training.wandb_support import init_from_config as init_wandb
from src.training.wandb_support import log as wandb_log
from src.utils import resolve_device, set_seed


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    run_from_config(cfg)


def run_from_config(cfg: DictConfig) -> dict[str, Any]:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    validate_config(cfg)
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))

    problem = cast(ProblemName, str(cfg.problem))
    encoder = cast(EncoderKind, str(cfg.encoder))
    decoder = cast(DecoderKind, str(cfg.decoder))
    target_algorithm = str(
        cfg.data.target_algorithm or DEFAULT_TARGET_ALGORITHM[problem]
    )
    train_path = resolve_data_path(cfg, split="train")
    val_path = resolve_data_path(cfg, split="val")
    test_path = resolve_data_path(cfg, split="test")
    output_dir = resolve_output_dir(cfg)
    matched_params = resolve_model_parameters(
        cfg,
        problem=problem,
        encoder=encoder,
        decoder=decoder,
    )

    train_loader = build_dataloader(
        train_path,
        problem,
        batch_size=int(cfg.data.batch_size),
        target_algorithm=target_algorithm,
        shuffle=bool(cfg.data.shuffle),
        num_workers=int(cfg.data.num_workers),
    )
    val_loader = (
        build_dataloader(
            val_path,
            problem,
            batch_size=int(cfg.data.eval_batch_size),
            target_algorithm=target_algorithm,
            shuffle=False,
            num_workers=int(cfg.data.num_workers),
        )
        if val_path is not None
        else None
    )
    test_loader = (
        build_dataloader(
            test_path,
            problem,
            batch_size=int(cfg.data.eval_batch_size),
            target_algorithm=target_algorithm,
            shuffle=False,
            num_workers=int(cfg.data.num_workers),
        )
        if test_path is not None
        else None
    )

    model = NCOModel(
        problem=problem,
        encoder_kind=encoder,
        decoder_kind=decoder,
        input_dim=INPUT_DIM_BY_PROBLEM[problem],
        d_model=matched_params["d_model"],
        num_layers=int(cfg.model.num_layers),
        num_heads=int(cfg.model.num_heads),
        d_ff=matched_params["d_ff"],
        dropout=float(cfg.model.dropout),
        tanh_clip=float(cfg.model.tanh_clip),
    )
    train_config = TrainingConfig(
        mode=str(cfg.mode),
        epochs=int(cfg.trainer.epochs),
        steps_per_epoch=none_or_int(cfg.trainer.steps_per_epoch),
        learning_rate=float(cfg.trainer.learning_rate),
        max_grad_norm=float(cfg.trainer.max_grad_norm),
        baseline=str(cfg.trainer.baseline),
        baseline_alpha=float(cfg.trainer.baseline_alpha),
        baseline_warmup_epochs=int(cfg.trainer.baseline_warmup_epochs),
        exp_baseline_beta=float(cfg.trainer.exp_baseline_beta),
        log_every=int(cfg.trainer.log_every),
        progress_bar=bool(cfg.trainer.progress_bar),
        output_dir=output_dir,
        save_checkpoints=bool(cfg.trainer.save_checkpoints),
        wandb_log=init_wandb(
            cfg,
            output_dir=output_dir,
            run_name=cfg.wandb.name
            or default_run_name(
                problem=problem,
                encoder=encoder,
                decoder=decoder,
                mode=str(cfg.mode),
            ),
            default_tags=[problem, encoder, decoder, str(cfg.mode)],
            config=build_wandb_config(
                cfg=cfg,
                matched_params=matched_params,
                model=model,
                train_path=train_path,
                val_path=val_path,
                test_path=test_path,
                target_algorithm=target_algorithm,
                output_dir=output_dir,
            ),
        ),
        wandb_train_eval_batches=int(cfg.wandb.train_eval_batches),
    )
    print(
        "run="
        f"problem={problem} encoder={encoder} decoder={decoder} mode={cfg.mode} "
        f"d_model={matched_params['d_model']} d_ff={matched_params['d_ff']} "
        f"params={matched_params['matched_params']} "
        f"target_params={matched_params['target_params']} device={device}"
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
    )
    result = trainer.fit()
    payload: dict[str, Any] = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "resolved": {
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path,
            "target_algorithm": target_algorithm,
            "output_dir": output_dir,
            "model": matched_params,
        },
        "training_config": asdict(train_config),
        "training_time_sec": result.training_time_sec,
        "history": result.history,
    }
    if test_loader is not None:
        test_eval = trainer.evaluate(test_loader)
        test_metrics = test_eval.to_dict("test")
        payload["test"] = test_metrics
        if train_config.wandb_log:
            wandb_log(wandb_metrics(test_eval, "test"), step=trainer.global_step)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "result.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    finish_wandb()
    return payload


def validate_config(cfg: DictConfig) -> None:
    if str(cfg.problem) not in PROBLEM_NAMES:
        raise ValueError(f"Unsupported problem: {cfg.problem}")
    if str(cfg.encoder) not in ENCODER_KINDS:
        raise ValueError(f"Unsupported encoder: {cfg.encoder}")
    if str(cfg.decoder) not in DECODER_KINDS:
        raise ValueError(f"Unsupported decoder: {cfg.decoder}")
    if str(cfg.mode) not in ("supervised", "rl"):
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


def resolve_data_path(cfg: DictConfig, *, split: str) -> str | None:
    configured = cfg.data.get(f"{split}_path")
    if configured is not None:
        return str(resolve_user_path(configured))
    if not bool(cfg.data.use_default_paths):
        if split == "train":
            raise ValueError("data.train_path is required when use_default_paths=false")
        return None
    if split == "train":
        instances = int(cfg.scale.train.instances)
    elif split == "val":
        instances = int(cfg.scale.validation.instances)
    elif split == "test":
        instances = int(cfg.scale.test.instances)
    else:
        raise ValueError(f"Unsupported split: {split}")
    problem = cast(ProblemName, str(cfg.problem))
    return str(
        problem_dataset_path(
            problem,
            split=split,
            instances=instances,
            data_root=cfg.data.root,
        )
    )


def resolve_output_dir(cfg: DictConfig) -> str:
    if cfg.paths.output_dir:
        return str(resolve_user_path(cfg.paths.output_dir))
    return (
        f"{cfg.paths.output_root}/seed_{cfg.seed}/{cfg.mode}/"
        f"{cfg.problem}/{cfg.encoder}/{cfg.decoder}"
    )


def resolve_model_parameters(
    cfg: DictConfig,
    *,
    problem: ProblemName,
    encoder: EncoderKind,
    decoder: DecoderKind,
) -> dict[str, int | float | str | None]:
    num_layers = int(cfg.model.num_layers)
    num_heads = int(cfg.model.num_heads)
    input_dim = INPUT_DIM_BY_PROBLEM[problem]

    if not bool(cfg.parameter_budget.enabled):
        d_model = int(cfg.model.d_model)
        d_ff = int(cfg.model.d_ff)
        matched_params = count_current_params(
            cfg, problem, encoder, decoder, d_model, d_ff
        )
        return finalize_resolved_parameters(
            source="explicit",
            input_dim=input_dim,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
            base_d_model=d_model,
            base_d_ff=d_ff,
            base_params=matched_params,
            matched_params=matched_params,
            target_params=matched_params,
            delta=0,
            delta_pct=0.0,
        )
    if cfg.model.d_model is not None and cfg.model.d_ff is not None:
        d_model = int(cfg.model.d_model)
        d_ff = int(cfg.model.d_ff)
        matched_params = count_current_params(
            cfg, problem, encoder, decoder, d_model, d_ff
        )
        target_params = int(cfg.parameter_budget.target_params or matched_params)
        settings = parameter_budget_settings(cfg)
        base_params = base_parameter_count(
            problem=problem,
            encoder=encoder,
            decoder=decoder,
            args=settings,
        )
        return finalize_resolved_parameters(
            source="explicit_over_budget",
            input_dim=input_dim,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
            base_d_model=settings.d_model,
            base_d_ff=settings.d_ff or settings.d_model * settings.d_ff_multiplier,
            base_params=base_params,
            matched_params=matched_params,
            target_params=target_params,
            delta=matched_params - target_params,
            delta_pct=100.0 * (matched_params - target_params) / max(target_params, 1),
        )

    row = find_budget_row(
        path=resolve_user_path(str(cfg.parameter_budget.path)),
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        target_params=none_or_int(cfg.parameter_budget.target_params),
        num_layers=num_layers,
        num_heads=num_heads,
    )
    if row is None:
        if bool(cfg.parameter_budget.strict):
            raise FileNotFoundError(
                "No parameter budget row for "
                f"problem={problem}, encoder={encoder}, decoder={decoder} "
                f"in {cfg.parameter_budget.path}"
            )
        row = compute_budget_row(cfg, problem=problem, encoder=encoder, decoder=decoder)
    return finalize_resolved_parameters(
        source=str(row.get("source", cfg.parameter_budget.path)),
        input_dim=int(row.get("input_dim", input_dim)),
        d_model=int(row["matched_d_model"]),
        d_ff=int(row["matched_d_ff"]),
        num_layers=num_layers,
        num_heads=num_heads,
        base_d_model=row.get("base_d_model"),
        base_d_ff=row.get("base_d_ff"),
        base_params=row.get("base_params"),
        matched_params=int(row["matched_params"]),
        target_params=int(row["target_params"]),
        delta=int(row["delta"]),
        delta_pct=float(row["delta_pct"]),
        command_args=row.get("command_args"),
    )


def finalize_resolved_parameters(
    *,
    source: str,
    input_dim: int,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    base_d_model: int | None,
    base_d_ff: int | None,
    base_params: int | None,
    matched_params: int,
    target_params: int,
    delta: int,
    delta_pct: float,
    command_args: str | None = None,
) -> dict[str, int | float | str | None]:
    if command_args is None:
        command_args = build_model_command_args(
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
        )
    return {
        "source": source,
        "input_dim": input_dim,
        "d_model": d_model,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "base_d_model": base_d_model,
        "base_d_ff": base_d_ff,
        "base_params": base_params,
        "matched_params": matched_params,
        "target_params": target_params,
        "delta": delta,
        "delta_pct": delta_pct,
        "command_args": command_args,
    }


def build_model_command_args(
    *,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
) -> str:
    return (
        f"model.d_model={d_model} model.d_ff={d_ff} "
        f"model.num_layers={num_layers} model.num_heads={num_heads}"
    )


def find_budget_row(
    *,
    path: Path,
    problem: ProblemName,
    encoder: EncoderKind,
    decoder: DecoderKind,
    target_params: int | None,
    num_layers: int,
    num_heads: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    settings = payload.get("settings", {})
    if settings:
        if int(settings.get("num_layers", num_layers)) != num_layers:
            return None
        if int(settings.get("num_heads", num_heads)) != num_heads:
            return None
    rows = payload.get("rows", [])
    matches = [
        row
        for row in rows
        if row.get("problem") == problem
        and row.get("encoder") == encoder
        and row.get("decoder") == decoder
        and (target_params is None or int(row.get("target_params")) == target_params)
    ]
    if not matches:
        return None
    row = dict(matches[0])
    row["source"] = str(path)
    return row


def compute_budget_row(
    cfg: DictConfig,
    *,
    problem: ProblemName,
    encoder: EncoderKind,
    decoder: DecoderKind,
) -> dict[str, Any]:
    settings = parameter_budget_settings(cfg)
    if cfg.parameter_budget.target_params is None:
        base_counts = [
            base_parameter_count(
                problem=problem_name,
                encoder=encoder_name,
                decoder=decoder_name,
                args=settings,
            )
            for problem_name in PROBLEM_NAMES
            for encoder_name in MATRIX_ENCODERS
            for decoder_name in DECODER_KINDS
        ]
        target_params = resolve_target(settings, base_counts)
    else:
        target_params = int(cfg.parameter_budget.target_params)
    matched_d_model, matched_d_ff, matched_params = find_closest_budget(
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        target_params=target_params,
        args=settings,
    )
    delta = matched_params - target_params
    base_params = base_parameter_count(
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        args=settings,
    )
    base_d_ff = settings.d_ff or settings.d_model * settings.d_ff_multiplier
    return {
        "source": "computed",
        "problem": problem,
        "encoder": encoder,
        "decoder": decoder,
        "input_dim": INPUT_DIM_BY_PROBLEM[problem],
        "base_d_model": settings.d_model,
        "base_d_ff": base_d_ff,
        "base_params": base_params,
        "target_params": target_params,
        "matched_d_model": matched_d_model,
        "matched_d_ff": matched_d_ff,
        "matched_params": matched_params,
        "delta": delta,
        "delta_pct": 100.0 * delta / max(target_params, 1),
        "command_args": build_model_command_args(
            d_model=matched_d_model,
            d_ff=matched_d_ff,
            num_layers=settings.num_layers,
            num_heads=settings.num_heads,
        ),
    }


def parameter_budget_settings(cfg: DictConfig) -> ParameterComparisonSettings:
    return ParameterComparisonSettings(
        problems=PROBLEM_NAMES,
        encoders=MATRIX_ENCODERS,
        include_graph_attention=False,
        decoders=DECODER_KINDS,
        d_model=int(cfg.parameter_budget.search.base_d_model),
        d_ff=none_or_int(cfg.parameter_budget.search.d_ff),
        d_ff_multiplier=int(cfg.parameter_budget.search.d_ff_multiplier),
        num_layers=int(cfg.model.num_layers),
        num_heads=int(cfg.model.num_heads),
        dropout=float(cfg.model.dropout),
        tanh_clip=float(cfg.model.tanh_clip),
        target_params=none_or_int(cfg.parameter_budget.target_params),
        match_target="max",
        min_d_model=int(cfg.parameter_budget.search.min_d_model),
        max_d_model=int(cfg.parameter_budget.search.max_d_model),
        d_model_step=int(cfg.parameter_budget.search.d_model_step),
        format="json",
        output=None,
    )


def count_current_params(
    cfg: DictConfig,
    problem: ProblemName,
    encoder: EncoderKind,
    decoder: DecoderKind,
    d_model: int,
    d_ff: int,
) -> int:
    model = NCOModel(
        problem=problem,
        encoder_kind=encoder,
        decoder_kind=decoder,
        input_dim=INPUT_DIM_BY_PROBLEM[problem],
        d_model=d_model,
        num_layers=int(cfg.model.num_layers),
        num_heads=int(cfg.model.num_heads),
        d_ff=d_ff,
        dropout=float(cfg.model.dropout),
        tanh_clip=float(cfg.model.tanh_clip),
    )
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def none_or_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return int(value)


if __name__ == "__main__":
    main()
