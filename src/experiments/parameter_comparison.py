###
#  uv run python -m src.experiments.parameter_comparison --format json --output outputs/src/parameter_budget.json
###
import argparse
import csv
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, cast

from src.constants import (
    DECODER_KINDS,
    ENCODER_KINDS,
    MATRIX_ENCODERS,
    PROBLEM_NAMES,
    DecoderKind,
    EncoderKind,
    ProblemName,
)
from src.model import NCOModel


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare trainable parameter counts for src architecture "
            "combinations and find d_model settings that match one parameter budget."
        )
    )
    parser.add_argument("--problems", default=",".join(PROBLEM_NAMES))
    parser.add_argument("--encoders", default=",".join(MATRIX_ENCODERS))
    parser.add_argument(
        "--include-graph-attention",
        action="store_true",
        help="Also include graph_attention, which currently uses attention code.",
    )
    parser.add_argument("--decoders", default=",".join(DECODER_KINDS))
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument(
        "--d-ff",
        type=int,
        help=(
            "Fixed feed-forward width. By default d_ff is d_model multiplied by "
            "--d-ff-multiplier."
        ),
    )
    parser.add_argument("--d-ff-multiplier", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tanh-clip", type=float, default=10.0)
    parser.add_argument(
        "--target-params",
        type=int,
        help="Explicit shared parameter target. Overrides --match-target.",
    )
    parser.add_argument(
        "--match-target",
        choices=("none", "max", "min", "median"),
        default="max",
        help=(
            "Target derived from base model counts. Default max upsizes smaller "
            "architectures to the largest base count."
        ),
    )
    parser.add_argument("--min-d-model", type=int, default=16)
    parser.add_argument("--max-d-model", type=int, default=512)
    parser.add_argument("--d-model-step", type=int, default=8)
    parser.add_argument(
        "--format",
        choices=("markdown", "csv", "json"),
        default="markdown",
    )
    parser.add_argument("--output", help="Write output to a file instead of stdout.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    problems = validate_values(split_csv(args.problems), PROBLEM_NAMES, "problem")
    encoders = validate_values(split_csv(args.encoders), ENCODER_KINDS, "encoder")
    if args.include_graph_attention and "graph_attention" not in encoders:
        encoders = (*encoders, "graph_attention")
    decoders = validate_values(split_csv(args.decoders), DECODER_KINDS, "decoder")

    base_counts = [
        base_parameter_count(
            problem=problem,
            encoder=encoder,
            decoder=decoder,
            args=args,
        )
        for problem in problems
        for encoder in encoders
        for decoder in decoders
    ]
    target_params = resolve_target(args, base_counts)
    rows = [
        parameter_row(
            problem=problem,
            encoder=encoder,
            decoder=decoder,
            args=args,
            target_params=target_params,
        )
        for problem in problems
        for encoder in encoders
        for decoder in decoders
    ]
    text = format_rows(rows, args=args, target_params=target_params)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


def base_parameter_count(
    *,
    problem: str,
    encoder: str,
    decoder: str,
    args: argparse.Namespace,
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
    args: argparse.Namespace,
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
        f"--d-model {matched_d_model} "
        f"--d-ff {matched_d_ff} "
        f"--num-layers {args.num_layers} "
        f"--num-heads {args.num_heads}"
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
    args: argparse.Namespace,
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
            f"check --min-d-model/--max-d-model/--d-model-step/--num-heads."
        )
    return best


def count_parameters(
    *,
    problem: str,
    encoder: str,
    decoder: str,
    d_model: int,
    d_ff: int,
    args: argparse.Namespace,
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


def resolve_target(args: argparse.Namespace, base_counts: list[int]) -> int:
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


def resolve_d_ff(args: argparse.Namespace, d_model: int) -> int:
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


def split_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def validate_values(
    values: Sequence[str],
    allowed: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    allowed_set = set(allowed)
    invalid = [value for value in values if value not in allowed_set]
    if invalid:
        raise SystemExit(
            f"Unsupported {label}: {', '.join(invalid)}. Allowed: {', '.join(allowed)}"
        )
    return tuple(values)


def format_rows(
    rows: list[ParameterRow],
    *,
    args: argparse.Namespace,
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
            "uv run python -m src.experiments.run ... --d-model <d_model> --d-ff <d_ff>",
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


if __name__ == "__main__":
    raise SystemExit(main())
