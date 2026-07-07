from pathlib import Path

from src.constants import (
    DEFAULT_SEEDS,
    PROBLEM_FILE_PREFIX,
    PROBLEM_PATH_DIR,
    ProblemName,
)

PROJECT_NAME = "lstm_transformer"
LOCAL_DB_ROOT = Path.home() / "local_db" / PROJECT_NAME
DATA_ROOT = LOCAL_DB_ROOT
PUBLIC_DATA_ROOT = LOCAL_DB_ROOT / "data_public"
PUBLIC_DATA_100K_ROOT = LOCAL_DB_ROOT / "data_public_100k"
PUBLIC_DATA_84K_ROOT = LOCAL_DB_ROOT / "data_public_84k"
RAW_ML4CO_ROOT = LOCAL_DB_ROOT / "raw"

_SPLIT_BY_SEED = {seed: split for split, seed in zip(("train", "val", "test"), DEFAULT_SEEDS)}


def resolve_data_root(value: str | Path | None = None) -> Path:
    if value is None:
        return DATA_ROOT
    return Path(value).expanduser()


def split_for_seed(seed: int) -> str:
    return _SPLIT_BY_SEED.get(seed, "train")


def problem_dataset_path(problem: ProblemName, seed: int) -> Path:
    directory = PROBLEM_PATH_DIR[problem]
    prefix = PROBLEM_FILE_PREFIX[problem]
    split = split_for_seed(seed)
    if split == "train":
        filename = f"{prefix}_seed{seed}.jsonl"
    else:
        filename = f"{prefix}_{split}_seed{seed}.jsonl"
    return DATA_ROOT / directory / filename
