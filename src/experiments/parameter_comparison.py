###
#  uv run python -m src.experiments.parameter_comparison \
#    format=json output=outputs/src/parameter_budget.json
###
import csv
import json
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from src.constants import (
    DECODER_KINDS,
    MATRIX_ENCODERS,
    PROBLEM_NAMES,
    DecoderKind,
    EncoderKind,
    ProblemName,
)
from src.model import NCOModel
from src.paths import resolve_user_path

INPUT_DIM_BY_PROBLEM: dict[str, int] = {
    "tsp": 2,
    "cvrp": 3,
    "orienteering": 3,
    "knapsack": 2,
    "mis": 1,
    "max_clique": 1,
    "vertex_cover": 1,
}


@dataclass(frozen=True)
class ParameterComparisonSettings:
    problems: tuple[str, ...]
    encoders: tuple[str, ...]
    include_graph_attention: bool
    decoders: tuple[str, ...]
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int | None
    d_ff_multiplier: int
    dropout: float
    tanh_clip: float
    target_params: int | None
    match_target: str
    min_d_model: int
    max_d_model: int
    d_model_step: int
    format: str
    output: str | None


@dataclass(frozen=True)
class ParameterRow:
    problem: str
    encoder: str
    decoder: str
    input_dim: int
    base_d_model: int
    base_d_ff: int
    base_params: int
    target_params: int
    matched_d_model: int
    matched_d_ff: int
    matched_params: int
    delta: int
    delta_pct: float
    command_args: str


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="parameter_comparison",
)
def main(cfg: DictConfig) -> None:
    run_from_config(cfg)


def run_from_config(cfg: DictConfig) -> str:
    settings = settings_from_config(cfg)
    text = build_parameter_comparison(settings)
    if settings.output:
        output_path = resolve_user_path(settings.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text)
    return text


def settings_from_config(cfg: DictConfig) -> ParameterComparisonSettings:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    problems = validate_values(
        config_sequence(cfg.problems, PROBLEM_NAMES),
        PROBLEM_NAMES,
        "problem",
    )
    encoders = validate_values(
        config_sequence(cfg.encoders, MATRIX_ENCODERS),
        MATRIX_ENCODERS,
        "encoder",
    )
    include_graph_attention = bool(cfg.include_graph_attention)
    if include_graph_attention and "graph_attention" not in encoders:
        encoders = (*encoders, "graph_attention")
    decoders = validate_values(
        config_sequence(cfg.decoders, DECODER_KINDS),
        DECODER_KINDS,
        "decoder",
    )
    match_target = str(cfg.match_target)
    if match_target not in {"none", "max", "min", "median"}:
        raise ValueError(f"Unsupported match_target: {match_target}")
    output_format = str(cfg.format)
    if output_format not in {"markdown", "csv", "json"}:
        raise ValueError(f"Unsupported format: {output_format}")
    min_d_model = int(cfg.search.min_d_model)
    max_d_model = int(cfg.search.max_d_model)
    d_model_step = int(cfg.search.d_model_step)
    if min_d_model <= 0 or max_d_model < min_d_model or d_model_step <= 0:
        raise ValueError("search d_model bounds must be positive and ordered")
    return ParameterComparisonSettings(
        problems=problems,
        encoders=encoders,
        include_graph_attention=include_graph_attention,
        decoders=decoders,
        d_model=int(cfg.model.d_model),
        num_layers=int(cfg.model.num_layers),
        num_heads=int(cfg.model.num_heads),
        d_ff=none_or_int(cfg.model.d_ff),
        d_ff_multiplier=int(cfg.model.d_ff_multiplier),
        dropout=float(cfg.model.dropout),
        tanh_clip=float(cfg.model.tanh_clip),
        target_params=none_or_int(cfg.target_params),
        match_target=match_target,
        min_d_model=min_d_model,
        max_d_model=max_d_model,
        d_model_step=d_model_step,
        format=output_format,
        output=none_or_str(cfg.output),
    )


def build_parameter_comparison(settings: ParameterComparisonSettings) -> str:
    base_counts = [
        base_parameter_count(
            problem=problem,
            encoder=encoder,
            decoder=decoder,
            args=settings,
        )
        for problem in settings.problems
        for encoder in settings.encoders
        for decoder in settings.decoders
    ]
    target_params = resolve_target(settings, base_counts)
    rows = [
        parameter_row(
            problem=problem,
            encoder=encoder,
            decoder=decoder,
            args=settings,
            target_params=target_params,
        )
        for problem in settings.problems
        for encoder in settings.encoders
        for decoder in settings.decoders
    ]
    return format_rows(rows, args=settings, target_params=target_params)


def base_parameter_count(
    *,
    problem: str,
    encoder: str,
    decoder: str,
    args: ParameterComparisonSettings,
) -> int:
    return count_parameters(
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        d_model=args.d_model,
        d_ff=resolve_d_ff(args, args.d_model),
        args=args,
    )


def parameter_row(
    *,
    problem: str,
    encoder: str,
    decoder: str,
    args: ParameterComparisonSettings,
    target_params: int,
) -> ParameterRow:
    base_d_ff = resolve_d_ff(args, args.d_model)
    base_params = count_parameters(
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        d_model=args.d_model,
        d_ff=base_d_ff,
        args=args,
    )
    matched_d_model, matched_d_ff, matched_params = find_closest_budget(
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        target_params=target_params,
        args=args,
    )
    delta = matched_params - target_params
    delta_pct = 100.0 * delta / max(target_params, 1)
    command_args = (
        f"model.d_model={matched_d_model} "
        f"model.d_ff={matched_d_ff} "
        f"model.num_layers={args.num_layers} "
        f"model.num_heads={args.num_heads}"
    )
    return ParameterRow(
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        input_dim=INPUT_DIM_BY_PROBLEM[problem],
        base_d_model=args.d_model,
        base_d_ff=base_d_ff,
        base_params=base_params,
        target_params=target_params,
        matched_d_model=matched_d_model,
        matched_d_ff=matched_d_ff,
        matched_params=matched_params,
        delta=delta,
        delta_pct=delta_pct,
        command_args=command_args,
    )


def find_closest_budget(
    *,
    problem: str,
    encoder: str,
    decoder: str,
    target_params: int,
    args: ParameterComparisonSettings,
) -> tuple[int, int, int]:
    if args.match_target == "none" and args.target_params is None:
        d_ff = resolve_d_ff(args, args.d_model)
        params = count_parameters(
            problem=problem,
            encoder=encoder,
            decoder=decoder,
            d_model=args.d_model,
            d_ff=d_ff,
            args=args,
        )
        return args.d_model, d_ff, params

    best: tuple[int, int, int] | None = None
    best_abs_delta: int | None = None
    for d_model in range(args.min_d_model, args.max_d_model + 1, args.d_model_step):
        if not valid_d_model(encoder, d_model, args.num_heads):
            continue
        d_ff = resolve_d_ff(args, d_model)
        params = count_parameters(
            problem=problem,
            encoder=encoder,
            decoder=decoder,
            d_model=d_model,
            d_ff=d_ff,
            args=args,
        )
        abs_delta = abs(params - target_params)
        if best is None or best_abs_delta is None:
            best = (d_model, d_ff, params)
            best_abs_delta = abs_delta
            continue
        if (abs_delta, d_model) < (best_abs_delta, best[0]):
            best = (d_model, d_ff, params)
            best_abs_delta = abs_delta
    if best is None:
        raise ValueError(
            f"No valid d_model candidates for encoder={encoder}; "
            "check search.min_d_model/search.max_d_model/"
            "search.d_model_step/model.num_heads."
        )
    return best


def count_parameters(
    *,
    problem: str,
    encoder: str,
    decoder: str,
    d_model: int,
    d_ff: int,
    args: ParameterComparisonSettings,
) -> int:
    if not valid_d_model(encoder, d_model, args.num_heads):
        return sys.maxsize
    model = NCOModel(
        problem=cast(ProblemName, problem),
        encoder_kind=cast(EncoderKind, encoder),
        decoder_kind=cast(DecoderKind, decoder),
        input_dim=INPUT_DIM_BY_PROBLEM[problem],
        d_model=d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=d_ff,
        dropout=args.dropout,
        tanh_clip=args.tanh_clip,
    )
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def resolve_target(args: ParameterComparisonSettings, base_counts: list[int]) -> int:
    if args.target_params is not None:
        return args.target_params
    if not base_counts:
        raise ValueError("Cannot resolve parameter target without any models")
    if args.match_target == "none":
        return base_counts[0]
    if args.match_target == "max":
        return max(base_counts)
    if args.match_target == "min":
        return min(base_counts)
    if args.match_target == "median":
        return int(statistics.median(base_counts))
    raise ValueError(f"Unsupported match target: {args.match_target}")


def resolve_d_ff(args: ParameterComparisonSettings, d_model: int) -> int:
    if args.d_ff is not None:
        return args.d_ff
    return d_model * args.d_ff_multiplier


def valid_d_model(encoder: str, d_model: int, num_heads: int) -> bool:
    if d_model <= 0:
        return False
    if encoder == "lstm" and d_model % 2 != 0:
        return False
    if encoder in ("attention", "graph_attention") and d_model % num_heads != 0:
        return False
    return True


def config_sequence(value: Any, default: Iterable[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        if not value.strip():
            return tuple(default)
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return (str(value),)


def validate_values(
    values: Sequence[str],
    allowed: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    allowed_set = set(allowed)
    invalid = [value for value in values if value not in allowed_set]
    if invalid:
        raise ValueError(
            f"Unsupported {label}: {', '.join(invalid)}. Allowed: {', '.join(allowed)}"
        )
    return tuple(values)


def format_rows(
    rows: list[ParameterRow],
    *,
    args: ParameterComparisonSettings,
    target_params: int,
) -> str:
    if args.format == "json":
        payload = {
            "settings": {
                "target_params": target_params,
                "base_d_model": args.d_model,
                "base_d_ff": resolve_d_ff(args, args.d_model),
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "d_ff_fixed": args.d_ff,
                "d_ff_multiplier": args.d_ff_multiplier,
                "min_d_model": args.min_d_model,
                "max_d_model": args.max_d_model,
                "d_model_step": args.d_model_step,
            },
            "summary": summary(rows),
            "rows": [asdict(row) for row in rows],
        }
        return json.dumps(payload, indent=2, sort_keys=True)
    if args.format == "csv":
        return format_csv(rows)
    return format_markdown(rows, target_params=target_params)


def format_markdown(rows: list[ParameterRow], *, target_params: int) -> str:
    headers = [
        "problem",
        "encoder",
        "decoder",
        "base params",
        "target",
        "d_model",
        "d_ff",
        "matched params",
        "delta %",
    ]
    lines = [
        f"Parameter target: `{target_params:,}` trainable parameters",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.problem,
                    row.encoder,
                    row.decoder,
                    f"{row.base_params:,}",
                    f"{row.target_params:,}",
                    str(row.matched_d_model),
                    str(row.matched_d_ff),
                    f"{row.matched_params:,}",
                    f"{row.delta_pct:+.2f}",
                ]
            )
            + " |"
        )
    stats = summary(rows)
    lines.extend(
        [
            "",
            "Summary:",
            f"- rows: `{stats['rows']}`",
            f"- max_abs_delta_pct: `{stats['max_abs_delta_pct']:.2f}`",
            f"- mean_abs_delta_pct: `{stats['mean_abs_delta_pct']:.2f}`",
            "",
            "Use the reported `d_model` and `d_ff` as run arguments, for example:",
            "",
            "```bash",
            "uv run python -m src.experiments.run ... "
            "model.d_model=<d_model> model.d_ff=<d_ff> "
            "parameter_budget.enabled=false",
            "```",
        ]
    )
    return "\n".join(lines)


def format_csv(rows: list[ParameterRow]) -> str:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    output = []
    writer = csv.DictWriter(_ListWriter(output), fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row))
    return "".join(output)


class _ListWriter:
    def __init__(self, output: list[str]) -> None:
        self.output = output

    def write(self, value: str) -> int:
        self.output.append(value)
        return len(value)


def summary(rows: list[ParameterRow]) -> dict[str, float | int]:
    if not rows:
        return {"rows": 0, "max_abs_delta_pct": 0.0, "mean_abs_delta_pct": 0.0}
    abs_delta = [abs(row.delta_pct) for row in rows]
    return {
        "rows": len(rows),
        "max_abs_delta_pct": max(abs_delta),
        "mean_abs_delta_pct": sum(abs_delta) / len(abs_delta),
    }


def none_or_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return int(value)


def none_or_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return str(value)


if __name__ == "__main__":
    main()
