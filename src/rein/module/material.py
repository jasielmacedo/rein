"""The files of the action material: `rein.module.pairs` writes them, `rein.module.act` reads them on the GPU image."""

import json
from pathlib import Path
from typing import Any

SAMPLES_FILE = "samples.jsonl"
AUDITOR_FILE = "auditor.jsonl"
VERDICTS_FILE = "verdicts.jsonl"
PAIRS_FILE = "pairs.jsonl"
ANCHORS_FILE = "anchors.jsonl"
SUMMARY_FILE = "summary.json"
LONG = "long"


def is_long(view_id: str) -> bool:
    return view_id.endswith(f"|{LONG}")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
