"""Read and validate a local W&B export."""

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
import json


def load_manifest(export_dir: Path) -> list[dict[str, Any]]:
    manifest_path = export_dir / "manifest.json"
    payload = _read_json(manifest_path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {manifest_path}")

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_entry in payload:
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Invalid manifest entry in {manifest_path}")
        entry = dict(raw_entry)
        run_id = str(entry.get("id", ""))
        if not run_id:
            raise ValueError("Every manifest entry must have a non-empty run id")
        if run_id in seen:
            raise ValueError(f"Duplicate run id in manifest: {run_id}")
        seen.add(run_id)
        entries.append(entry)
    return entries


def load_run_configs(export_dir: Path) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    runs_dir = export_dir / "runs"
    for path in sorted(runs_dir.glob("*/config.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        run = payload.get("run")
        if not isinstance(run, dict):
            raise ValueError(f"Missing nested run config in {path}")
        run_id = path.parent.name.rsplit("__", maxsplit=1)[-1]
        if run_id in configs:
            raise ValueError(f"Duplicate config for run id {run_id}")
        configs[run_id] = dict(payload)
    return configs


def iter_history(export_dir: Path) -> Iterator[dict[str, Any]]:
    history_path = export_dir / "all_history.jsonl"
    with history_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {history_path}:{line_number}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object at {history_path}:{line_number}"
                )
            yield row


def validate_export(
    manifest: Sequence[Mapping[str, Any]],
    configs: Mapping[str, Mapping[str, Any]],
) -> None:
    manifest_ids = {str(entry["id"]) for entry in manifest}
    config_ids = set(configs)
    missing = manifest_ids - config_ids
    unexpected = config_ids - manifest_ids
    if missing or unexpected:
        raise ValueError(
            "Manifest/config mismatch: "
            f"missing configs={sorted(missing)}, unexpected configs={sorted(unexpected)}"
        )


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required export file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
