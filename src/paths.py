from pathlib import Path

PROJECT_NAME = "lstm_transformer"
LOCAL_DB_ROOT = Path.home() / "local_db" / PROJECT_NAME
DATA_ROOT = LOCAL_DB_ROOT
PUBLIC_DATA_ROOT = LOCAL_DB_ROOT / "data_public"
PUBLIC_DATA_100K_ROOT = LOCAL_DB_ROOT / "data_public_100k"
PUBLIC_DATA_84K_ROOT = LOCAL_DB_ROOT / "data_public_84k"
RAW_ML4CO_ROOT = LOCAL_DB_ROOT / "raw" / "ml4co"


def resolve_data_root(value: str | Path | None = None) -> Path:
    if value is None:
        return DATA_ROOT
    return Path(value).expanduser()
