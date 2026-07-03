import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.constants import DEFAULT_TARGET_ALGORITHM, ProblemName


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object")
            records.append(record)
    return records


class ProblemDataset(Dataset):
    """File-backed dataset for the new experiment stack.

    The loader accepts the current repository JSONL schema and the proposed
    nested schema where each row contains `instance` and `label` objects.
    """

    def __init__(
        self,
        path: str | Path,
        problem: ProblemName,
        target_algorithm: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.path = str(path)
        self.problem = problem
        self.target_algorithm = target_algorithm or DEFAULT_TARGET_ALGORITHM[problem]
        self.dtype = dtype
        self.records = read_jsonl(path)
        for index, raw in enumerate(self.records):
            record = self._normalise_record(raw)
            if _canonical_problem(record.get("problem")) != problem:
                raise ValueError(
                    f"Record {index} in {path} has problem={record.get('problem')!r}; "
                    f"expected {problem!r}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._normalise_record(self.records[index])
        if self.problem == "tsp":
            return self._tsp(record)
        if self.problem == "cvrp":
            return self._cvrp(record)
        if self.problem == "orienteering":
            return self._orienteering(record)
        if self.problem == "knapsack":
            return self._knapsack(record)
        if self.problem in ("mis", "max_clique", "vertex_cover"):
            return self._graph_subset(record)
        raise ValueError(f"Unsupported problem: {self.problem}")

    @staticmethod
    def _normalise_record(raw: dict[str, Any]) -> dict[str, Any]:
        if "instance" not in raw:
            return dict(raw)
        instance = raw["instance"]
        if not isinstance(instance, dict):
            raise ValueError("Nested JSONL row field 'instance' must be an object")
        record = dict(instance)
        for key in ("problem", "index", "seed"):
            if key in raw and key not in record:
                record[key] = raw[key]
        if "solutions" not in record and "label" in raw:
            label = raw["label"]
            if not isinstance(label, dict):
                raise ValueError("Nested JSONL row field 'label' must be an object")
            solver = raw.get("solver") or label.get("algorithm") or "solver"
            record["solutions"] = {str(solver): label}
        return record

    def _base_item(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": torch.tensor(int(record.get("index", 0)), dtype=torch.long),
            "seed": torch.tensor(int(record.get("seed", 0)), dtype=torch.long),
        }

    def _solution(self, record: dict[str, Any]) -> dict[str, Any] | None:
        solutions = record.get("solutions")
        if not isinstance(solutions, dict):
            return None
        if self.target_algorithm in solutions:
            solution = solutions[self.target_algorithm]
            return solution if isinstance(solution, dict) else None
        if len(solutions) == 1:
            solution = next(iter(solutions.values()))
            return solution if isinstance(solution, dict) else None
        available = ", ".join(str(key) for key in solutions)
        raise ValueError(
            f"{self.path} record is missing target algorithm "
            f"{self.target_algorithm!r}; available: {available}"
        )

    def _tsp(self, record: dict[str, Any]) -> dict[str, Any]:
        loc = torch.tensor(record["coordinates"], dtype=self.dtype)
        item = self._base_item(record)
        item.update({"loc": loc})
        solution = self._solution(record)
        if solution is not None and "tour" in solution:
            target = torch.tensor(solution["tour"], dtype=torch.long)
            item["target_actions"] = target
            item["target_tour"] = target
            item["target_mask"] = torch.ones(loc.size(0), dtype=self.dtype)
            if "cost" in solution:
                item["target_value"] = torch.tensor(solution["cost"], dtype=self.dtype)
        return item

    def _cvrp(self, record: dict[str, Any]) -> dict[str, Any]:
        depot = torch.tensor(record["depot"], dtype=self.dtype)
        coordinates = torch.tensor(record["coordinates"], dtype=self.dtype)
        loc = torch.cat([depot.view(1, -1), coordinates], dim=0)
        demands = torch.tensor(record["demands"], dtype=self.dtype)
        capacity = torch.tensor(
            record.get("vehicle_capacity", record.get("capacity")),
            dtype=self.dtype,
        )
        node_demands = torch.cat([torch.zeros(1, dtype=self.dtype), demands])
        item = self._base_item(record)
        item.update(
            {
                "depot": depot,
                "coordinates": coordinates,
                "loc": loc,
                "demands": demands,
                "node_demands": node_demands,
                "capacity": capacity,
            }
        )
        solution = self._solution(record)
        if solution is not None:
            actions = _cvrp_solution_actions(solution)
            if actions:
                item["target_actions"] = torch.tensor(actions, dtype=torch.long)
            mask = torch.zeros(loc.size(0), dtype=self.dtype)
            mask[1:] = 1.0
            item["target_mask"] = mask
            if "cost" in solution:
                item["target_value"] = torch.tensor(solution["cost"], dtype=self.dtype)
        return item

    def _orienteering(self, record: dict[str, Any]) -> dict[str, Any]:
        depot = torch.tensor(record["depot"], dtype=self.dtype)
        coordinates = torch.tensor(record["coordinates"], dtype=self.dtype)
        loc = torch.cat([depot.view(1, -1), coordinates], dim=0)
        prizes = torch.tensor(record["prizes"], dtype=self.dtype)
        node_prizes = torch.cat([torch.zeros(1, dtype=self.dtype), prizes])
        item = self._base_item(record)
        item.update(
            {
                "depot": depot,
                "coordinates": coordinates,
                "loc": loc,
                "prizes": prizes,
                "node_prizes": node_prizes,
                "travel_budget": torch.tensor(
                    record["travel_budget"],
                    dtype=self.dtype,
                ),
            }
        )
        solution = self._solution(record)
        if solution is not None:
            tour = [int(node) + 1 for node in solution.get("tour", [])]
            item["target_actions"] = torch.tensor([*tour, 0], dtype=torch.long)
            target_mask = torch.zeros(loc.size(0), dtype=self.dtype)
            if tour:
                target_mask[torch.tensor(tour, dtype=torch.long)] = 1.0
            item["target_mask"] = target_mask
            if "value" in solution:
                item["target_value"] = torch.tensor(
                    solution["value"],
                    dtype=self.dtype,
                )
            if "length" in solution:
                item["target_length"] = torch.tensor(
                    solution["length"],
                    dtype=self.dtype,
                )
        return item

    def _knapsack(self, record: dict[str, Any]) -> dict[str, Any]:
        weights = torch.tensor(record["weights"], dtype=self.dtype)
        values = torch.tensor(record["values"], dtype=self.dtype)
        capacity = torch.tensor(record["capacity"], dtype=self.dtype)
        item = self._base_item(record)
        item.update({"weights": weights, "values": values, "capacity": capacity})
        solution = self._solution(record)
        if solution is not None:
            selected = [int(item_id) for item_id in solution.get("items", [])]
            target_mask = torch.zeros(weights.size(0), dtype=self.dtype)
            if selected:
                target_mask[torch.tensor(selected, dtype=torch.long)] = 1.0
            item["target_mask"] = target_mask
            item["target_actions"] = torch.tensor(
                [*selected, weights.size(0)],
                dtype=torch.long,
            )
            if "value" in solution:
                item["target_value"] = torch.tensor(
                    solution["value"],
                    dtype=self.dtype,
                )
            if "weight" in solution:
                item["target_weight"] = torch.tensor(
                    solution["weight"],
                    dtype=self.dtype,
                )
        return item

    def _graph_subset(self, record: dict[str, Any]) -> dict[str, Any]:
        num_nodes = int(record["num_nodes"])
        adjacency = _edges_to_adjacency(num_nodes, record.get("edges", []))
        item = self._base_item(record)
        item.update({"adjacency": torch.tensor(adjacency, dtype=self.dtype)})
        solution = self._solution(record)
        if solution is not None:
            selected = [int(node) for node in solution.get("nodes", [])]
            target_mask = torch.zeros(num_nodes, dtype=self.dtype)
            if selected:
                target_mask[torch.tensor(selected, dtype=torch.long)] = 1.0
            item["target_mask"] = target_mask
            item["target_actions"] = torch.tensor(
                [*selected, num_nodes],
                dtype=torch.long,
            )
            if "size" in solution:
                item["target_value"] = torch.tensor(
                    solution["size"],
                    dtype=self.dtype,
                )
        return item


def build_dataloader(
    path: str | Path,
    problem: ProblemName,
    *,
    batch_size: int,
    target_algorithm: str | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        ProblemDataset(path, problem, target_algorithm=target_algorithm),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_problem_batch,
    )


def collate_problem_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    keys = set().union(*(item.keys() for item in items))
    batch: dict[str, Any] = {}
    for key in sorted(keys):
        values = [item.get(key) for item in items]
        present = [value for value in values if value is not None]
        if not present:
            continue
        first = present[0]
        if not isinstance(first, torch.Tensor):
            batch[key] = values
            continue
        if all(isinstance(value, torch.Tensor) for value in values):
            tensors = [value for value in values if isinstance(value, torch.Tensor)]
            if _same_shape(tensors):
                batch[key] = torch.stack(tensors)
            elif tensors[0].dtype == torch.long and tensors[0].ndim == 1:
                batch[key] = _pad_1d_long(tensors, pad_value=-1)
            else:
                raise ValueError(f"Cannot collate variable-shape tensor key {key!r}")
        else:
            batch[key] = values
    return batch


def _same_shape(tensors: Iterable[torch.Tensor]) -> bool:
    tensors = list(tensors)
    shape = tensors[0].shape
    return all(tensor.shape == shape for tensor in tensors)


def _pad_1d_long(tensors: list[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(int(tensor.numel()) for tensor in tensors)
    padded = torch.full((len(tensors), max_len), pad_value, dtype=torch.long)
    for row, tensor in enumerate(tensors):
        padded[row, : tensor.numel()] = tensor.long()
    return padded


def _canonical_problem(raw: Any) -> str:
    if raw == "maximum_clique":
        return "max_clique"
    if raw == "minimum_vertex_cover":
        return "vertex_cover"
    return str(raw)


def _edges_to_adjacency(num_nodes: int, edges: list[list[int]]) -> list[list[float]]:
    adjacency = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for edge in edges:
        u, v = int(edge[0]), int(edge[1])
        adjacency[u][v] = 1.0
        adjacency[v][u] = 1.0
    return adjacency


def _cvrp_solution_actions(solution: dict[str, Any]) -> list[int]:
    routes = solution.get("routes")
    if not routes:
        tour = solution.get("tour") or []
        return [int(customer) + 1 for customer in tour]
    actions: list[int] = []
    for route_index, route in enumerate(routes):
        if route_index > 0:
            actions.append(0)
        actions.extend(int(customer) + 1 for customer in route)
    return actions
