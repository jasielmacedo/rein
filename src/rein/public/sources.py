import random
import tarfile
import urllib.request
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from rein.public.common import (
    MAX_PUBLIC_OPTIONS,
    PUBLIC_SPLIT,
    class_names_from_card,
    intent_question,
    read_jsonl,
    render_value,
    single_message_state,
    try_record,
)
from rein.schema import Question, TrainingRecord

KEV_REPO = "jaredpalmer/kev-suites"
KEV_TRAIN = "v7/decision-v7/train.jsonl"
BANKING_REPO = "PolyAI/banking77"
BANKING_TRAIN = "data/train-00000-of-00001.parquet"
CLINC_REPO = "clinc/clinc_oos"
CLINC_TRAIN = "plus/train-00000-of-00001.parquet"
CLINC_OUT_OF_SCOPE = "oos"
MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MASSIVE_LOCALES = (
    "en-US",
    "pt-PT",
    "es-ES",
    "fr-FR",
    "de-DE",
    "it-IT",
    "ja-JP",
    "zh-CN",
    "ko-KR",
    "ru-RU",
    "ar-SA",
    "hi-IN",
)
MASSIVE_TRAIN_PARTITION = "train"
DATASET_REPO = "dataset"


def _as_text(value: Any) -> Any:
    return value if value is None or isinstance(value, str) else render_value(value).strip()


def _normalize(raw: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Text for structured instructions and descriptions, and at most MAX_PUBLIC_OPTIONS choice options."""
    fields: dict[str, Any] = {key: raw[key] for key in ("type", "instructions", "criteria", "label") if key in raw}
    fields["instructions"] = _as_text(fields.get("instructions"))
    criteria = fields.get("criteria")
    if isinstance(criteria, dict):
        criteria = {str(key): _as_text(value) for key, value in criteria.items()}
        label = fields.get("label")
        if fields["type"] == "choice" and len(criteria) > MAX_PUBLIC_OPTIONS and isinstance(label, str):
            others = [key for key in criteria if key != label]
            kept = [label, *rng.sample(others, MAX_PUBLIC_OPTIONS - 1)]
            rng.shuffle(kept)
            criteria = {key: criteria[key] for key in kept}
        fields["criteria"] = criteria
    elif isinstance(criteria, list):
        fields["criteria"] = [_as_text(item) for item in criteria]
    return fields


def kev_decision_v7(count: int, seed: int, rejected: Counter[str]) -> Iterator[TrainingRecord]:
    rows = read_jsonl(Path(hf_hub_download(KEV_REPO, KEV_TRAIN, repo_type=DATASET_REPO)))
    rng = random.Random(seed)
    for index, row in enumerate(rng.sample(rows, min(count, len(rows)))):
        questions = {}
        for qid, raw in row["questions"].items():
            try:
                questions[qid] = Question(**_normalize(raw, rng))
            except ValueError:
                rejected["kev:question"] += 1
        if not questions:
            continue
        state = row["state"] if isinstance(row["state"], str) else render_value(row["state"])
        record = try_record(
            rejected,
            "kev:record",
            id=f"kev-v7-{index:05d}",
            source="kev_decision_v7",
            scenario_id=f"kev-v7-{index:05d}",
            split=PUBLIC_SPLIT,
            state=single_message_state(state),
            questions=questions,
            meta={"src": sorted({raw.get("src") for raw in row["questions"].values()} - {None})},
        )
        if record is not None:
            yield record


def _parquet_intents(repo: str, filename: str) -> tuple[list[str], list[tuple[str, str]]]:
    readme = Path(hf_hub_download(repo, "README.md", repo_type=DATASET_REPO)).read_text(encoding="utf-8")
    table = pq.read_table(hf_hub_download(repo, filename, repo_type=DATASET_REPO)).to_pylist()
    label_field = "label" if "label" in table[0] else "intent"
    names = class_names_from_card(readme, label_field)
    return names, [(row["text"], names[row[label_field]]) for row in table]


def intent_records(
    source: str, repo: str, filename: str, count: int, seed: int, rejected: Counter[str], none_label: str | None = None
) -> Iterator[TrainingRecord]:
    names, rows = _parquet_intents(repo, filename)
    intents = [name for name in names if name != none_label]
    rng = random.Random(seed)
    for index, (text, label) in enumerate(rng.sample(rows, min(count, len(rows)))):
        gold = None if label == none_label else label
        record = try_record(
            rejected,
            f"{source}:record",
            id=f"{source}-{index:05d}",
            source=source,
            scenario_id=f"{source}-{index:05d}",
            split=PUBLIC_SPLIT,
            state=single_message_state(text),
            questions={"intent": intent_question(gold, intents, rng)},
            meta={"lang_conv": "en"},
        )
        if record is not None:
            yield record


def _massive_dir(cache_dir: Path) -> Path:
    target = cache_dir / "massive-1.1"
    if not target.exists():
        archive = cache_dir / "massive-1.1.tar.gz"
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MASSIVE_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter="data")
    return target


def massive_records(per_locale: int, seed: int, cache_dir: Path, rejected: Counter[str]) -> Iterator[TrainingRecord]:
    root = _massive_dir(cache_dir)
    rng = random.Random(seed)
    for locale in MASSIVE_LOCALES:
        files = list(root.rglob(f"{locale}.jsonl"))
        if not files:
            rejected[f"massive:missing_{locale}"] += 1
            continue
        rows = [row for row in read_jsonl(files[0]) if row.get("partition") == MASSIVE_TRAIN_PARTITION]
        intents = sorted({row["intent"] for row in rows})
        for index, row in enumerate(rng.sample(rows, min(per_locale, len(rows)))):
            record = try_record(
                rejected,
                "massive:record",
                id=f"massive-{locale}-{index:05d}",
                source="massive",
                scenario_id=f"massive-{locale}-{index:05d}",
                split=PUBLIC_SPLIT,
                state=single_message_state(row["utt"]),
                questions={"intent": intent_question(row["intent"], intents, rng)},
                meta={"lang_conv": locale},
            )
            if record is not None:
                yield record
