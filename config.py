"""Load configuration from .env with sensible defaults."""
import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_env() -> None:
    """Read KEY=VALUE pairs from .env into os.environ (no external deps)."""
    if not ENV_PATH.exists():
        return
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def get_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, ""))
    except ValueError:
        return default


def get_list(key: str, default: str = "") -> list[str]:
    """Parse a comma-separated env var into a de-duplicated list of values."""
    raw = os.environ.get(key, default)
    seen: list[str] = []
    for item in raw.split(","):
        item = item.strip()
        if item and item not in seen:
            seen.append(item)
    return seen
