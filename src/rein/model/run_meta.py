"""A run's metadata file: what it is called and how to read it. Light on purpose, so serving clients can read a run
without the training stack."""

import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

CONFIG_FILE = "rein_config.json"
SOFT_FILE = "soft.safetensors"
FORMAT_VERSION = 1
MERGED_FORMAT_VERSION = 2
SUPPORTED_FORMATS = (FORMAT_VERSION, MERGED_FORMAT_VERSION)


def resolve(source: Path | str, token: str | None = None) -> Path:
    """A local run or export directory, or a Hugging Face repo id that is downloaded once."""
    path = Path(source)
    if path.exists():
        return path
    return Path(snapshot_download(str(source), token=token))


def read_meta(directory: Path) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((directory / CONFIG_FILE).read_text(encoding="utf-8"))
    if meta.get("format_version") not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported checkpoint format {meta.get('format_version')!r}")
    return meta
