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


def resolve_user_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def resolve_data_root(value: str | Path | None = None) -> Path:
    if value is None:
        return DATA_ROOT
    return resolve_user_path(value)


def split_for_seed(seed: int) -> str:
    return _SPLIT_BY_SEED.get(seed, "train")


def problem_dataset_path(
    problem: ProblemName,
    *,
    split: str,
    instances: int,
    data_root: str | Path | None = None,
) -> Path:
    root = resolve_data_root(data_root)
    directory = PROBLEM_PATH_DIR[problem]
    prefix = PROBLEM_FILE_PREFIX[problem]
    filename = f"{prefix}_{split}_{instances}.jsonl"
    return root / directory / filename


def problem_split_paths(
    problem: ProblemName,
    *,
    train_instances: int,
    val_instances: int,
    test_instances: int,
    data_root: str | Path | None = None,
) -> dict[str, str]:
    return {
        split: str(
            problem_dataset_path(
                problem,
                split=split,
                instances=instances,
                data_root=data_root,
            )
        )
        for split, instances in (
            ("train", train_instances),
            ("val", val_instances),
            ("test", test_instances),
        )
    }
