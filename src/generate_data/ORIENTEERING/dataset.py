from typing import Any

import torch
from torch.utils.data import Dataset

from src.generate_data.common import load_jsonl


class OrienteeringDataset(Dataset):
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
            if record.get("problem") != "orienteering":
                raise ValueError(f"Record {index} is not an Orienteering instance")
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
        prizes = torch.tensor(record["prizes"], dtype=self.dtype)
        node_prizes = torch.cat([torch.zeros(1, dtype=self.dtype), prizes], dim=0)

        item: dict[str, Any] = {
            "index": torch.tensor(record["index"], dtype=torch.long),
            "seed": torch.tensor(record["seed"], dtype=torch.long),
            "depot": depot,
            "coordinates": coordinates,
            "loc": loc,
            "prizes": prizes,
            "node_prizes": node_prizes,
            "travel_budget": torch.tensor(record["travel_budget"], dtype=self.dtype),
        }

        if self.target_algorithm is not None:
            solution = record["solutions"][self.target_algorithm]
            target = torch.zeros(len(prizes), dtype=self.dtype)
            target[torch.tensor(solution["tour"], dtype=torch.long)] = 1.0
            stop_index = len(prizes)
            target_sequence = torch.full(
                (len(prizes) + 1,),
                stop_index,
                dtype=torch.long,
            )
            target_tour = torch.tensor(solution["tour"], dtype=torch.long)
            target_sequence[: len(target_tour)] = target_tour
            item["target_tour"] = solution["tour"]
            item["target_sequence"] = target_sequence
            item["target_set"] = target
            item["target_value"] = torch.tensor(solution["value"], dtype=self.dtype)
            item["target_length"] = torch.tensor(solution["length"], dtype=self.dtype)

        return item


def collate_orienteering(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty Orienteering batch")
    keys = items[0].keys()
    batch: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in items]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    return batch
