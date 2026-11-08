import os
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class MissingSettingError(RuntimeError):
    def __init__(self, name: str) -> None:
        super().__init__(f"required setting {name} is not set (.env or environment)")
        self.name = name


def load_env(env_file: Path = ENV_FILE) -> dict[str, str]:
    file_values = {}
    if env_file.exists():
        file_values = {key: value for key, value in dotenv_values(env_file).items() if value is not None}
    return {**file_values, **os.environ}


def require(values: dict[str, str], name: str) -> str:
    value = values.get(name, "").strip()
    if not value:
        raise MissingSettingError(name)
    return value


def optional(values: dict[str, str], name: str) -> str | None:
    value = values.get(name, "").strip()
    return value or None
