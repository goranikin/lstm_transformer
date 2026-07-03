import random
from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.am_config import AMModelConfig
from configs.pn_config import PNModelConfig
from src.data_generating.CVRP.dataset import CVRPDataset, collate_cvrp
from src.data_generating.KNAPSACK.dataset import KnapsackDataset, collate_knapsack
from src.data_generating.MAX_CLIQUE.dataset import (
    MaxCliqueDataset,
    collate_max_clique,
)
from src.data_generating.MIS.dataset import MISDataset, collate_mis
from src.data_generating.ORIENTEERING.dataset import (
    OrienteeringDataset,
    collate_orienteering,
)
from src.data_generating.TSP.dataset import TSPDataset, collate_tsp
from src.data_generating.VERTEX_COVER.dataset import (
    VertexCoverDataset,
    collate_vertex_cover,
)
from src.models.modified_attention_model.model import ModifiedAttentionModel
from src.models.modular.model import DecoderKind, EncoderKind, ModularNCOModel
from src.models.pointer_network.model import PointerNetwork
from src.models.transformer.model import AttentionModel

ModelName = Literal[
    "am",
    "modified_am",
    "pn",
    "modular_am",
    "modular_pn",
    "am_lstm_pointer",
    "am_gru_pointer",
    "am_lstm_subset",
    "am_sigmoid_subset",
]
ProblemName = Literal[
    "tsp",
    "cvrp",
    "mis",
    "maximum_clique",
    "minimum_vertex_cover",
    "knapsack",
    "orienteering",
]
LegacyProblemName = Literal["tsp", "mis"]
MODEL_NAMES = tuple(ModelName.__args__)
PROBLEM_NAMES = tuple(ProblemName.__args__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def build_dataset(
    problem: ProblemName,
    path: str,
    target_algorithm: str | None = None,
) -> (
    TSPDataset
    | CVRPDataset
    | MISDataset
    | MaxCliqueDataset
    | VertexCoverDataset
    | KnapsackDataset
    | OrienteeringDataset
):
    if problem == "tsp":
        return TSPDataset(path, target_algorithm=target_algorithm)
    if problem == "cvrp":
        return CVRPDataset(path, target_algorithm=target_algorithm)
    if problem == "mis":
        return MISDataset(path, target_algorithm=target_algorithm)
    if problem == "maximum_clique":
        return MaxCliqueDataset(path, target_algorithm=target_algorithm)
    if problem == "minimum_vertex_cover":
        return VertexCoverDataset(path, target_algorithm=target_algorithm)
    if problem == "knapsack":
        return KnapsackDataset(path, target_algorithm=target_algorithm)
    if problem == "orienteering":
        return OrienteeringDataset(path, target_algorithm=target_algorithm)
    raise ValueError(f"Unsupported problem: {problem}")


def collate_for(
    problem: ProblemName,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    if problem == "tsp":
        return collate_tsp
    if problem == "cvrp":
        return collate_cvrp
    if problem == "mis":
        return collate_mis
    if problem == "maximum_clique":
        return collate_max_clique
    if problem == "minimum_vertex_cover":
        return collate_vertex_cover
    if problem == "knapsack":
        return collate_knapsack
    if problem == "orienteering":
        return collate_orienteering
    raise ValueError(f"Unsupported problem: {problem}")


def build_loader(
    problem: ProblemName,
    path: str,
    batch_size: int,
    target_algorithm: str | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        build_dataset(problem, path, target_algorithm=target_algorithm),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_for(problem),
    )


def build_model(
    model_name: ModelName,
    problem: ProblemName,
    model_options: Mapping[str, Any] | None = None,
) -> torch.nn.Module:
    options = dict(model_options or {})
    interface = dict(options.get("interface") or {})
    if model_name == "modular_pn" or (
        model_name == "pn" and problem not in ("tsp", "mis")
    ):
        return _build_modular_model(
            problem=problem,
            options=options,
            encoder_kind="lstm",
            decoder_kind="lstm_pointer",
        )
    if model_name == "modular_am" or (
        model_name == "am" and problem not in ("tsp", "mis")
    ):
        return _build_modular_model(
            problem=problem,
            options=options,
            encoder_kind="attention",
            decoder_kind="attention_pointer",
        )
    if model_name == "am_lstm_pointer":
        return _build_modular_model(
            problem=problem,
            options=options,
            encoder_kind="attention",
            decoder_kind="lstm_pointer",
        )
    if model_name == "am_gru_pointer":
        return _build_modular_model(
            problem=problem,
            options=options,
            encoder_kind="attention",
            decoder_kind="gru_pointer",
        )
    if model_name == "am_lstm_subset":
        return _build_modular_model(
            problem=problem,
            options=options,
            encoder_kind="attention",
            decoder_kind="lstm_subset",
        )
    if model_name == "am_sigmoid_subset":
        return _build_modular_model(
            problem=problem,
            options=options,
            encoder_kind="attention",
            decoder_kind="sigmoid_subset",
        )
    if model_name in ("am", "modified_am"):
        legacy_problem = _as_legacy_problem(problem)
        model_cls = AttentionModel if model_name == "am" else ModifiedAttentionModel
        return model_cls(
            config=AMModelConfig(**dict(options.get("am") or {})),
            default_problem=legacy_problem,
            tsp_input_size=int(options.get("tsp_input_size", 2)),
            mis_input_size=int(options.get("mis_input_size", 1)),
            mis_context_size=int(options.get("mis_context_size", 1)),
            loc_key=str(interface.get("loc_key", "loc")),
            adjacency_key=str(interface.get("adjacency_key", "adjacency")),
            target_tour_key=str(interface.get("target_tour_key", "target_tour")),
            target_set_key=str(interface.get("target_set_key", "target_set")),
        )
    if model_name == "pn":
        legacy_problem = _as_legacy_problem(problem)
        pn_input_size = (
            int(options.get("mis_input_size", 1))
            if problem == "mis"
            else int(options.get("input_size", options.get("tsp_input_size", 2)))
        )
        return PointerNetwork(
            input_size=pn_input_size,
            config=PNModelConfig(**dict(options.get("pn") or {})),
            default_problem=legacy_problem,
            loc_key=str(interface.get("loc_key", "loc")),
            adjacency_key=str(interface.get("adjacency_key", "adjacency")),
            target_tour_key=str(interface.get("target_tour_key", "target_tour")),
            target_set_key=str(interface.get("target_set_key", "target_set")),
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _as_legacy_problem(problem: ProblemName) -> LegacyProblemName:
    if problem not in ("tsp", "mis"):
        raise ValueError(
            f"Legacy AM/PN classes only support tsp/mis, got {problem}. "
            "Use modular_am, modular_pn, am_lstm_pointer, am_gru_pointer, "
            "am_lstm_subset, or am_sigmoid_subset for the expanded problem set."
        )
    return problem


def _build_modular_model(
    *,
    problem: ProblemName,
    options: Mapping[str, Any],
    encoder_kind: EncoderKind,
    decoder_kind: DecoderKind,
) -> ModularNCOModel:
    am_options = dict(options.get("am") or {})
    if encoder_kind == "lstm":
        pn_options = dict(options.get("pn") or {})
        if "hidden_size" in pn_options and "d_h" not in am_options:
            am_options["d_h"] = int(pn_options["hidden_size"])
    return ModularNCOModel(
        config=AMModelConfig(**am_options),
        encoder_kind=encoder_kind,
        decoder_kind=decoder_kind,
        default_problem=problem,
    )


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def default_target_algorithm(problem: ProblemName) -> str:
    if problem == "tsp":
        return "concorde"
    if problem == "cvrp":
        return "gurobi"
    if problem == "mis":
        return "kamis"
    if problem == "maximum_clique":
        return "gurobi"
    if problem == "minimum_vertex_cover":
        return "gurobi"
    if problem == "knapsack":
        return "dynamic_programming"
    if problem == "orienteering":
        return "gurobi"
    raise ValueError(f"Unsupported problem: {problem}")
