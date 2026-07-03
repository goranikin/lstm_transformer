import random
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from configs.validation import default_am_config
from src.constants import (
    DEFAULT_MATRIX_MODELS,
    DecoderKind,
    EncoderKind,
    ModelName,
    ProblemName,
    is_modular_model,
    modular_encoder_decoder,
)
from src.generate_data.CVRP.dataset import CVRPDataset, collate_cvrp
from src.generate_data.KNAPSACK.dataset import KnapsackDataset, collate_knapsack
from src.generate_data.MAX_CLIQUE.dataset import (
    MaxCliqueDataset,
    collate_max_clique,
)
from src.generate_data.MIS.dataset import MISDataset, collate_mis
from src.generate_data.ORIENTEERING.dataset import (
    OrienteeringDataset,
    collate_orienteering,
)
from src.generate_data.TSP.dataset import TSPDataset, collate_tsp
from src.generate_data.VERTEX_COVER.dataset import (
    VertexCoverDataset,
    collate_vertex_cover,
)
from src.models.modular.model import ModularNCOModel


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
    model_options: Mapping[str, Any] | DictConfig | None = None,
) -> torch.nn.Module:
    if not is_modular_model(model_name):
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Expected one of: {', '.join(DEFAULT_MATRIX_MODELS)}"
        )
    options = model_options if model_options is not None else {}
    encoder_kind, decoder_kind = modular_encoder_decoder(model_name)
    return _build_modular_model(
        problem=problem,
        options=options,
        encoder_kind=encoder_kind,
        decoder_kind=decoder_kind,
    )


def _build_modular_model(
    *,
    problem: ProblemName,
    options: Mapping[str, Any],
    encoder_kind: EncoderKind,
    decoder_kind: DecoderKind,
) -> ModularNCOModel:
    am_cfg = options.get("am") or default_am_config()
    if encoder_kind == "lstm":
        pn_options = options.get("pn")
        if (
            pn_options is not None
            and "hidden_size" in pn_options
            and "d_h" not in am_cfg
        ):
            am_cfg = OmegaConf.merge(am_cfg, {"d_h": int(pn_options["hidden_size"])})
    return ModularNCOModel(
        config=cast(DictConfig, am_cfg),
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
        return "gurobi"
    if problem == "maximum_clique":
        return "gurobi"
    if problem == "minimum_vertex_cover":
        return "gurobi"
    if problem == "knapsack":
        return "dynamic_programming"
    if problem == "orienteering":
        return "gurobi"
    raise ValueError(f"Unsupported problem: {problem}")
