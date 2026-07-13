from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from src.wandb.config import ExportSettings, RunCombination


def build_run_filters(settings: ExportSettings) -> dict[str, Any]:
    """Build the Public API filter matching the W&B workspace query."""
    return {
        "$and": [
            {"updatedAt": {"$lte": settings.cutoff}},
            {"tags": settings.tag},
        ]
    }


def fetch_runs(api: Any, settings: ExportSettings) -> list[Any]:
    """Fetch every run selected by the configured project query."""
    return list(
        api.runs(
            settings.project,
            filters=build_run_filters(settings),
            order="+created_at",
            per_page=settings.query_page_size,
        )
    )


def run_combination(run: Any) -> RunCombination:
    """Return the experiment dimensions stored in a run's nested config."""
    run_config = run.config.get("run")
    if not isinstance(run_config, dict):
        raise ValueError(f"Run {run.id!r} has no nested 'run' config")

    required = ("problem", "encoder", "decoder", "mode", "seed")
    missing = [key for key in required if run_config.get(key) is None]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Run {run.id!r} is missing config fields: {missing_text}")

    return (
        str(run_config["problem"]),
        str(run_config["encoder"]),
        str(run_config["decoder"]),
        str(run_config["mode"]),
        int(run_config["seed"]),
    )


@dataclass(frozen=True)
class MatrixValidation:
    run_count: int
    expected_count: int
    missing: frozenset[RunCombination]
    unexpected: frozenset[RunCombination]
    duplicates: dict[RunCombination, int]

    @property
    def is_valid(self) -> bool:
        return (
            self.run_count == self.expected_count
            and not self.missing
            and not self.unexpected
            and not self.duplicates
        )

    def error_message(self) -> str:
        lines = [
            "Selected runs do not form the expected experiment matrix.",
            f"Matched runs: {self.run_count}; expected: {self.expected_count}",
        ]
        if self.missing:
            lines.append(f"Missing combinations: {_format_combinations(self.missing)}")
        if self.unexpected:
            lines.append(
                f"Unexpected combinations: {_format_combinations(self.unexpected)}"
            )
        if self.duplicates:
            duplicate_text = ", ".join(
                f"{combination!r} x{count}"
                for combination, count in sorted(
                    self.duplicates.items(), key=lambda item: repr(item[0])
                )
            )
            lines.append(f"Duplicate combinations: {duplicate_text}")
        return "\n".join(lines)


def validate_run_matrix(
    runs: Iterable[Any],
    settings: ExportSettings,
) -> MatrixValidation:
    """Compare selected runs with the expected problem-by-decoder matrix."""
    combinations = [run_combination(run) for run in runs]
    counts = Counter(combinations)
    actual = frozenset(combinations)
    expected = settings.expected_combinations
    return MatrixValidation(
        run_count=len(combinations),
        expected_count=settings.expected_run_count,
        missing=expected - actual,
        unexpected=actual - expected,
        duplicates={
            combination: count for combination, count in counts.items() if count > 1
        },
    )


def _format_combinations(combinations: Iterable[RunCombination]) -> str:
    return ", ".join(repr(item) for item in sorted(combinations, key=repr))
