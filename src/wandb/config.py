from dataclasses import dataclass
from itertools import product
from pathlib import Path

from src.constants import MATRIX_ENCODERS, PROBLEM_NAMES

RunCombination = tuple[str, str, str, str, int]

DEFAULT_PROJECT = "goranikin-my-project/lstm_transformer"
DEFAULT_CUTOFF = "2026-07-12T01:03:40Z"
DEFAULT_TAG = "supervised"
DEFAULT_MODE = "supervised"
DEFAULT_SEED = 1234
DEFAULT_OUTPUT_DIR = Path("outputs/wandb_export/supervised_seed1234")
LEGACY_SUPERVISED_DECODERS = (
    "attention_pointer",
    "lstm_pointer",
    "gru_pointer",
    "sigmoid_subset",
)


@dataclass(frozen=True)
class ExportSettings:
    project: str = DEFAULT_PROJECT
    cutoff: str = DEFAULT_CUTOFF
    tag: str = DEFAULT_TAG
    mode: str = DEFAULT_MODE
    seed: int = DEFAULT_SEED
    encoder: str = MATRIX_ENCODERS[0]
    problems: tuple[str, ...] = PROBLEM_NAMES
    # This exporter targets the completed pre-Transformer 28-run snapshot.
    decoders: tuple[str, ...] = LEGACY_SUPERVISED_DECODERS
    output_dir: Path = DEFAULT_OUTPUT_DIR
    query_page_size: int = 100
    history_page_size: int = 1_000
    download_log_files: bool = True

    @property
    def expected_combinations(self) -> frozenset[RunCombination]:
        return frozenset(
            (problem, self.encoder, decoder, self.mode, self.seed)
            for problem, decoder in product(self.problems, self.decoders)
        )

    @property
    def expected_run_count(self) -> int:
        return len(self.expected_combinations)
