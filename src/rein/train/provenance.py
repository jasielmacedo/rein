import hashlib
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from rein.env import PROJECT_ROOT

TRACKED_PACKAGES = ("torch", "transformers", "peft", "flash-linear-attention", "causal-conv1d")


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def collect(config: dict[str, Any], data_files: dict[str, Path | None]) -> dict[str, Any]:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config": config,
        "data_sha256": {name: file_sha256(path) for name, path in data_files.items() if path is not None},
        "packages": {name: _package_version(name) for name in TRACKED_PACKAGES},
        "python": platform.python_version(),
        "gpu": gpu,
        "gpu_count": torch.cuda.device_count(),
    }
