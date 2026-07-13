from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO
import json
import re

from src.wandb.config import ExportSettings


@dataclass(frozen=True)
class ManifestEntry:
    id: str
    name: str | None
    url: str
    state: str
    created_at: str | None
    tags: list[str]
    problem: str
    encoder: str
    decoder: str
    mode: str
    seed: int
    history_rows: int
    log_files: list[str]


ProgressCallback = Callable[[int, int, ManifestEntry], None]


def export_runs(
    runs: Iterable[Any],
    settings: ExportSettings,
    *,
    progress: ProgressCallback | None = None,
) -> Path:
    """Export configs, summaries, unsampled histories, and synchronized logs."""
    selected_runs = list(runs)
    output_dir = settings.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[ManifestEntry] = []

    combined_path = output_dir / "all_history.jsonl"
    with combined_path.open("w", encoding="utf-8") as combined_history:
        for index, run in enumerate(selected_runs, start=1):
            entry = export_run(
                run,
                output_dir=output_dir,
                combined_history=combined_history,
                history_page_size=settings.history_page_size,
                download_log_files=settings.download_log_files,
            )
            manifest.append(entry)
            if progress is not None:
                progress(index, len(selected_runs), entry)

    _write_json(output_dir / "manifest.json", [asdict(entry) for entry in manifest])
    return output_dir


def export_run(
    run: Any,
    *,
    output_dir: Path,
    combined_history: TextIO,
    history_page_size: int,
    download_log_files: bool,
) -> ManifestEntry:
    """Export a single W&B run and return its manifest metadata."""
    run_config = _nested_run_config(run)
    problem = str(run_config["problem"])
    encoder = str(run_config["encoder"])
    decoder = str(run_config["decoder"])
    mode = str(run_config["mode"])
    seed = int(run_config["seed"])

    directory_name = "__".join(
        _safe_component(value) for value in (problem, decoder, str(run.id))
    )
    run_dir = output_dir / "runs" / directory_name
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "config.json", run.config)
    _write_json(run_dir / "summary.json", dict(run.summary))

    metadata = {
        "_run_id": str(run.id),
        "_run_name": run.name,
        "_problem": problem,
        "_encoder": encoder,
        "_decoder": decoder,
        "_mode": mode,
        "_seed": seed,
    }
    history_rows = _export_history(
        run,
        destination=run_dir / "history.jsonl",
        combined_history=combined_history,
        metadata=metadata,
        page_size=history_page_size,
    )
    log_files = _download_log_files(run, run_dir) if download_log_files else []

    return ManifestEntry(
        id=str(run.id),
        name=run.name,
        url=str(run.url),
        state=str(run.state),
        created_at=getattr(run, "created_at", None),
        tags=[str(tag) for tag in run.tags],
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        mode=mode,
        seed=seed,
        history_rows=history_rows,
        log_files=log_files,
    )


def _export_history(
    run: Any,
    *,
    destination: Path,
    combined_history: TextIO,
    metadata: Mapping[str, Any],
    page_size: int,
) -> int:
    row_count = 0
    with destination.open("w", encoding="utf-8") as per_run_history:
        for row in run.scan_history(page_size=page_size):
            enriched = dict(row)
            enriched.update(metadata)
            line = json.dumps(enriched, default=str)
            per_run_history.write(line + "\n")
            combined_history.write(line + "\n")
            row_count += 1
    return row_count


def _download_log_files(run: Any, run_dir: Path) -> list[str]:
    files_dir = run_dir / "files"
    downloaded: list[str] = []
    for file in run.files(per_page=100):
        name = str(file.name)
        if not _is_log_file(name):
            continue
        relative_path = Path(name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe W&B file path: {name!r}")
        (files_dir / relative_path.parent).mkdir(parents=True, exist_ok=True)
        file.download(root=str(files_dir), replace=True)
        downloaded.append(name)
    return sorted(downloaded)


def _nested_run_config(run: Any) -> dict[str, Any]:
    run_config = run.config.get("run")
    if not isinstance(run_config, dict):
        raise ValueError(f"Run {run.id!r} has no nested 'run' config")
    return run_config


def _is_log_file(name: str) -> bool:
    return name == "output.log" or name.lower().endswith(".log")


def _safe_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not sanitized:
        raise ValueError(f"Cannot construct a directory name from {value!r}")
    return sanitized


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
