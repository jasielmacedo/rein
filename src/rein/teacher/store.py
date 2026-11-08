import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

KEY_FIELD = "key"


def content_key(*parts: Any) -> str:
    canonical = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class JsonlStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._keys: set[str] = set()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self._load_keys()
            self._terminate_partial_line()

    def _load_keys(self) -> None:
        for row in self.rows():
            key = row.get(KEY_FIELD)
            if isinstance(key, str):
                self._keys.add(key)

    def _terminate_partial_line(self) -> None:
        with self._path.open("rb") as fh:
            fh.seek(0, 2)
            if fh.tell() == 0:
                return
            fh.seek(-1, 2)
            last = fh.read(1)
        if last != b"\n":
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write("\n")

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    def append(self, key: str, value: dict[str, Any]) -> bool:
        if key in self._keys:
            return False
        line = json.dumps({KEY_FIELD: key, **value}, ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self._keys.add(key)
        return True

    def rows(self) -> Iterator[dict[str, Any]]:
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row
