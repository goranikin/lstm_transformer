"""Generate small JSONL smoke datasets for matrix runs."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.generate_data.CVRP.generate import CVRPGenerationConfig, generate_cvrp_dataset
from src.generate_data.KNAPSACK.generate import (
    KnapsackGenerationConfig,
    generate_knapsack_dataset,
)
from src.generate_data.MAX_CLIQUE.generate import (
    MaxCliqueGenerationConfig,
    generate_max_clique_dataset,
)
from src.generate_data.MIS.generate import MISGenerationConfig, generate_mis_dataset
from src.generate_data.ORIENTEERING.generate import (
    OrienteeringGenerationConfig,
    generate_orienteering_dataset,
)
from src.generate_data.TSP.generate import TSPGenerationConfig, generate_tsp_dataset
from src.generate_data.VERTEX_COVER.generate import (
    VertexCoverGenerationConfig,
    generate_vertex_cover_dataset,
)

SMOKE_NUM_INSTANCES = 64
SMOKE_SEED = 1234


@dataclass(frozen=True)
class SmokeDatasetRequest:
    problem: str
    output_path: str


def unique_requests(
    requests: Sequence[SmokeDatasetRequest],
) -> tuple[SmokeDatasetRequest, ...]:
    seen: set[str] = set()
    unique: list[SmokeDatasetRequest] = []
    for request in requests:
        if request.output_path in seen:
            continue
        seen.add(request.output_path)
        unique.append(request)
    return tuple(unique)


def smoke_dataset_ready(path: str) -> bool:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        return False
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return True
    return False


def missing_smoke_paths(requests: Sequence[SmokeDatasetRequest]) -> tuple[str, ...]:
    return tuple(
        request.output_path
        for request in unique_requests(requests)
        if not smoke_dataset_ready(request.output_path)
    )


def generate_smoke_dataset(
    request: SmokeDatasetRequest,
    *,
    num_instances: int = SMOKE_NUM_INSTANCES,
    seed: int = SMOKE_SEED,
) -> int:
    path = request.output_path
    if request.problem == "tsp":
        return generate_tsp_dataset(
            TSPGenerationConfig(
                num_instances=num_instances,
                num_nodes=20,
                seed=seed,
                output_path=path,
            )
        )
    if request.problem == "cvrp":
        return generate_cvrp_dataset(
            CVRPGenerationConfig(
                num_instances=num_instances,
                num_customers=20,
                seed=seed,
                output_path=path,
            )
        )
    if request.problem == "mis":
        return generate_mis_dataset(
            MISGenerationConfig(
                num_instances=num_instances,
                num_nodes=30,
                edge_probability=0.15,
                seed=seed,
                output_path=path,
            )
        )
    if request.problem == "maximum_clique":
        return generate_max_clique_dataset(
            MaxCliqueGenerationConfig(
                num_instances=num_instances,
                num_nodes=30,
                edge_probability=0.5,
                seed=seed,
                output_path=path,
            )
        )
    if request.problem == "minimum_vertex_cover":
        return generate_vertex_cover_dataset(
            VertexCoverGenerationConfig(
                num_instances=num_instances,
                num_nodes=30,
                edge_probability=0.15,
                seed=seed,
                output_path=path,
            )
        )
    if request.problem == "knapsack":
        return generate_knapsack_dataset(
            KnapsackGenerationConfig(
                num_instances=num_instances,
                num_items=50,
                seed=seed,
                output_path=path,
            )
        )
    if request.problem == "orienteering":
        return generate_orienteering_dataset(
            OrienteeringGenerationConfig(
                num_instances=num_instances,
                num_nodes=20,
                seed=seed,
                output_path=path,
            )
        )
    raise ValueError(f"Unsupported smoke problem: {request.problem}")


def ensure_smoke_datasets(
    requests: Sequence[SmokeDatasetRequest],
    *,
    skip_existing: bool = True,
    num_instances: int = SMOKE_NUM_INSTANCES,
    seed: int = SMOKE_SEED,
) -> tuple[str, ...]:
    written: list[str] = []
    for request in unique_requests(requests):
        if skip_existing and smoke_dataset_ready(request.output_path):
            continue
        count = generate_smoke_dataset(
            request,
            num_instances=num_instances,
            seed=seed,
        )
        print(f"Wrote {count} {request.problem} instances to {request.output_path}")
        written.append(request.output_path)
    return tuple(written)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate smoke JSONL datasets used by run_matrix."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate datasets even when output files already exist.",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=SMOKE_NUM_INSTANCES,
        help=f"Instances per file (default: {SMOKE_NUM_INSTANCES}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from src.main.experiments.run_matrix import SUPERVISED_SMOKE_RUNS

    args = _build_parser().parse_args(argv)
    requests = tuple(
        SmokeDatasetRequest(problem=run.problem, output_path=run.train_path)
        for run in SUPERVISED_SMOKE_RUNS
    )
    ensure_smoke_datasets(
        requests,
        skip_existing=not args.force,
        num_instances=args.num_instances,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
