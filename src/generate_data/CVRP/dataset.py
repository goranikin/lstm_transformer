from typing import Any

import torch
from torch.utils.data import Dataset

from src.generate_data.common import load_jsonl


class CVRPDataset(Dataset):
    def __init__(
        self,
        path: str,
        target_algorithm: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.path = path
        self.target_algorithm = target_algorithm
        self.dtype = dtype
        self.records = load_jsonl(path)

        for index, record in enumerate(self.records):
            if record.get("problem") != "cvrp":
                raise ValueError(f"Record {index} is not a CVRP instance")
            if (
                target_algorithm is not None
                and target_algorithm not in record["solutions"]
            ):
                raise ValueError(
                    f"Record {index} does not contain target algorithm "
                    f"'{target_algorithm}'"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        depot = torch.tensor(record["depot"], dtype=self.dtype)
        coordinates = torch.tensor(record["coordinates"], dtype=self.dtype)
        loc = torch.cat([depot.view(1, -1), coordinates], dim=0)
        demands = torch.tensor(record["demands"], dtype=self.dtype)
        vehicle_capacity = torch.tensor(record["vehicle_capacity"], dtype=self.dtype)
        node_demands = torch.cat([torch.zeros(1, dtype=self.dtype), demands], dim=0)

        item: dict[str, Any] = {
            "index": torch.tensor(record["index"], dtype=torch.long),
            "seed": torch.tensor(record["seed"], dtype=torch.long),
            "depot": depot,
            "coordinates": coordinates,
            "loc": loc,
            "demands": demands,
            "node_demands": node_demands,
            "vehicle_capacity": vehicle_capacity,
            "capacity": vehicle_capacity,
        }

        if self.target_algorithm is not None:
            solution = record["solutions"][self.target_algorithm]
            item["target_routes"] = solution["routes"]
            item["target_tour"] = torch.tensor(
                [customer for route in solution["routes"] for customer in route],
                dtype=torch.long,
            )
            item["target_cost"] = torch.tensor(solution["cost"], dtype=self.dtype)

        return item


def collate_cvrp(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty CVRP batch")
    keys = items[0].keys()
    batch: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in items]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    return batch
