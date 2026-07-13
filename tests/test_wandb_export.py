import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.wandb.config import ExportSettings
from src.wandb.exporter import export_runs
from src.wandb.selection import build_run_filters, fetch_runs, validate_run_matrix


class FakeFile:
    def __init__(self, name: str) -> None:
        self.name = name

    def download(self, *, root: str, replace: bool) -> None:
        del replace
        destination = Path(root) / self.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"contents of {self.name}\n", encoding="utf-8")


class FakeRun:
    def __init__(
        self,
        *,
        problem: str,
        decoder: str,
        run_id: str,
        encoder: str = "attention",
        mode: str = "supervised",
        seed: int = 1234,
        history: list[dict[str, Any]] | None = None,
        files: list[FakeFile] | None = None,
    ) -> None:
        self.id = run_id
        self.name = f"{problem}_{decoder}_{mode}"
        self.url = f"https://wandb.example/runs/{run_id}"
        self.state = "finished"
        self.created_at = "2026-07-11T00:00:00Z"
        self.tags = [problem, encoder, decoder, mode]
        self.config = {
            "run": {
                "problem": problem,
                "encoder": encoder,
                "decoder": decoder,
                "mode": mode,
                "seed": seed,
            }
        }
        self.summary = {"train/sl/loss": 0.25}
        self._history = history or []
        self._files = files or []
        self.history_page_size: int | None = None

    def scan_history(self, *, page_size: int) -> list[dict[str, Any]]:
        self.history_page_size = page_size
        return self._history

    def files(self, *, per_page: int) -> list[FakeFile]:
        self.files_page_size = per_page
        return self._files


class FakeApi:
    def __init__(self, runs: list[FakeRun]) -> None:
        self.selected_runs = runs
        self.call: dict[str, Any] | None = None

    def runs(self, project: str, **kwargs: Any) -> list[FakeRun]:
        self.call = {"project": project, **kwargs}
        return self.selected_runs


class WandbSelectionTests(unittest.TestCase):
    def test_fetch_runs_uses_workspace_query(self) -> None:
        settings = ExportSettings()
        api = FakeApi([])

        self.assertEqual(fetch_runs(api, settings), [])
        self.assertIsNotNone(api.call)
        assert api.call is not None
        self.assertEqual(api.call["project"], settings.project)
        self.assertEqual(api.call["filters"], build_run_filters(settings))
        self.assertEqual(
            api.call["filters"],
            {
                "$and": [
                    {"updatedAt": {"$lte": "2026-07-12T01:03:40Z"}},
                    {"tags": "supervised"},
                ]
            },
        )

    def test_validate_complete_problem_decoder_matrix(self) -> None:
        settings = ExportSettings()
        runs = [
            FakeRun(problem=problem, decoder=decoder, run_id=f"run-{index}")
            for index, (problem, _, decoder, _, _) in enumerate(
                settings.expected_combinations
            )
        ]

        result = validate_run_matrix(runs, settings)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.run_count, 28)

    def test_validate_reports_missing_and_duplicate_combinations(self) -> None:
        settings = ExportSettings(
            problems=("tsp", "cvrp"),
            decoders=("attention_pointer",),
        )
        duplicate = FakeRun(
            problem="tsp",
            decoder="attention_pointer",
            run_id="duplicate",
        )

        result = validate_run_matrix([duplicate, duplicate], settings)

        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.missing), 1)
        self.assertEqual(next(iter(result.duplicates.values())), 2)
        self.assertIn("Missing combinations", result.error_message())
        self.assertIn("Duplicate combinations", result.error_message())


class WandbExporterTests(unittest.TestCase):
    def test_export_writes_history_manifest_and_log_files(self) -> None:
        run = FakeRun(
            problem="tsp",
            decoder="lstm_pointer",
            run_id="abc123",
            history=[
                {"_step": 1, "train/sl/loss": 0.5},
                {"_step": 2, "train/sl/loss": 0.25},
            ],
            files=[FakeFile("output.log"), FakeFile("model.pt")],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = ExportSettings(
                problems=("tsp",),
                decoders=("lstm_pointer",),
                output_dir=Path(temp_dir) / "export",
                history_page_size=123,
            )
            output_dir = export_runs([run], settings)

            combined_rows = _read_jsonl(output_dir / "all_history.jsonl")
            per_run_dir = output_dir / "runs" / "tsp__lstm_pointer__abc123"
            per_run_rows = _read_jsonl(per_run_dir / "history.jsonl")
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(combined_rows, per_run_rows)
            self.assertEqual(len(combined_rows), 2)
            self.assertEqual(combined_rows[0]["_run_id"], "abc123")
            self.assertEqual(combined_rows[0]["_problem"], "tsp")
            self.assertEqual(run.history_page_size, 123)
            self.assertEqual(manifest[0]["history_rows"], 2)
            self.assertEqual(manifest[0]["log_files"], ["output.log"])
            self.assertTrue((per_run_dir / "files" / "output.log").is_file())
            self.assertFalse((per_run_dir / "files" / "model.pt").exists())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
