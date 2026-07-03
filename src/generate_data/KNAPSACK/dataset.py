from typing import Any

import torch
from torch.utils.data import Dataset

from src.generate_data.common import load_jsonl


class KnapsackDataset(Dataset):
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
            if record.get("problem") != "knapsack":
                raise ValueError(f"Record {index} is not a Knapsack instance")
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
        weights = torch.tensor(record["weights"], dtype=self.dtype)
        values = torch.tensor(record["values"], dtype=self.dtype)
        capacity = torch.tensor(record["capacity"], dtype=self.dtype)
        item_features = torch.stack(
            [
                weights / capacity.clamp_min(1.0),
                values / values.max().clamp_min(1.0),
            ],
            dim=1,
        )

        item: dict[str, Any] = {
            "index": torch.tensor(record["index"], dtype=torch.long),
            "seed": torch.tensor(record["seed"], dtype=torch.long),
            "weights": weights,
            "values": values,
            "capacity": capacity,
            "item_features": item_features,
        }

        if self.target_algorithm is not None:
            solution = record["solutions"][self.target_algorithm]
            target = torch.zeros(len(weights), dtype=self.dtype)
            target[torch.tensor(solution["items"], dtype=torch.long)] = 1.0
            item["target_set"] = target
            item["target_value"] = torch.tensor(solution["value"], dtype=self.dtype)
            item["target_weight"] = torch.tensor(solution["weight"], dtype=self.dtype)

        return item


def collate_knapsack(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty Knapsack batch")
    keys = items[0].keys()
    batch: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in items]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    return batch
