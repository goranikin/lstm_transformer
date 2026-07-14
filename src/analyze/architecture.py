"""Build an auditable inventory of the architectures used by each run."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from src.analyze.metadata import DECODER_FAMILY, problem_metadata
from src.analyze.records import ArchitectureRecord
from src.constants import DecoderKind, EncoderKind, ProblemName
from src.model import NCOModel


def build_architecture_records(
    manifest: Sequence[Mapping[str, Any]],
    configs: Mapping[str, Mapping[str, Any]],
) -> list[ArchitectureRecord]:
    records = []
    for entry in manifest:
        run_id = str(entry["id"])
        config = configs[run_id]
        run = _mapping(config, "run", run_id)
        model_config = _mapping(config, "model", run_id)
        trainer = _mapping(config, "trainer", run_id)
        data = _mapping(config, "data", run_id)
        budget = config.get("parameter_budget")
        budget = budget if isinstance(budget, Mapping) else {}
        matched = budget.get("matched")
        matched = matched if isinstance(matched, Mapping) else {}

        problem = str(run["problem"])
        encoder = str(run["encoder"])
        decoder = str(run["decoder"])
        metadata = problem_metadata(problem)
        model = _instantiate_model(problem, encoder, decoder, model_config)
        encoder_parameters = _parameter_count(model.encoder)
        decoder_parameters = _parameter_count(model.decoder)
        computed_total = encoder_parameters + decoder_parameters
        logged_total = int(model_config.get("total_params", computed_total))
        if computed_total != logged_total:
            raise ValueError(
                f"Run {run_id} architecture no longer matches its logged parameter "
                f"count: current={computed_total}, logged={logged_total}"
            )

        records.append(
            ArchitectureRecord(
                run_id=run_id,
                problem=problem,
                problem_family=metadata.family,
                topology=metadata.topology,
                solver=metadata.solver,
                objective_sense=metadata.objective_sense,
                encoder=encoder,
                decoder=decoder,
                decoder_family=DECODER_FAMILY.get(decoder, decoder),
                input_dim=int(model_config["input_dim"]),
                context_dim=int(model.problem.context_dim),
                d_model=int(model_config["d_model"]),
                d_ff=int(model_config["d_ff"]),
                num_layers=int(model_config["num_layers"]),
                num_heads=int(model_config["num_heads"]),
                transformer_decoder_layers=int(
                    model_config.get("transformer_decoder_layers", 1)
                ),
                dropout=float(model_config["dropout"]),
                tanh_clip=float(model_config["tanh_clip"]),
                encoder_parameters=encoder_parameters,
                decoder_parameters=decoder_parameters,
                total_parameters=logged_total,
                trainable_parameters=int(
                    model_config.get("trainable_params", logged_total)
                ),
                target_parameters=_optional_int(matched.get("target_params")),
                parameter_delta=_optional_int(matched.get("delta")),
                parameter_delta_pct=_optional_float(matched.get("delta_pct")),
                epochs=int(trainer["epochs"]),
                steps_per_epoch=_optional_int(trainer.get("steps_per_epoch")),
                batch_size=int(data["batch_size"]),
            )
        )
    return sorted(records, key=lambda row: (row.problem, row.decoder))


def _instantiate_model(
    problem: str,
    encoder: str,
    decoder: str,
    config: Mapping[str, Any],
) -> NCOModel:
    return NCOModel(
        problem=cast(ProblemName, problem),
        encoder_kind=cast(EncoderKind, encoder),
        decoder_kind=cast(DecoderKind, decoder),
        input_dim=int(config["input_dim"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
        transformer_decoder_layers=int(
            config.get("transformer_decoder_layers", 1)
        ),
        dropout=float(config["dropout"]),
        tanh_clip=float(config["tanh_clip"]),
    )


def _parameter_count(module: Any) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _mapping(
    value: Mapping[str, Any], key: str, run_id: str
) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ValueError(f"Run {run_id} is missing mapping config[{key!r}]")
    return nested


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
