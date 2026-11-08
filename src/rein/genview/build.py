import argparse
import json
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from transformers import AutoTokenizer

from rein.genview import constants as c
from rein.genview.keys import load_bases
from rein.genview.records import MissingBase, record_spec
from rein.genview.render import GeneratorView, View
from rein.genview.scenarios import ScenarioIndex
from rein.schema import TrainingRecord
from rein.train.data import load_records

NO_THINKING = "nothink"
THINKING = "think"
DRAFT = "draft"
TRAIN_FILE = "train.jsonl"
EVAL_FILES = {"development": "development.jsonl", "calibration": "calibration.jsonl", "test": "test.jsonl"}
GOLD_FILE = "gold.jsonl"
SAMPLE_SEED = 0


@dataclass(frozen=True)
class ViewRow:
    view_id: str
    record_id: str
    split: str
    mode: str
    decision: str
    ids: tuple[int, ...]
    prompt_length: int
    positions: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["ids"] = list(self.ids)
        payload["positions"] = {name: list(indices) for name, indices in self.positions.items()}
        return json.dumps(payload)

    @classmethod
    def from_json(cls, line: str) -> "ViewRow":
        payload = json.loads(line)
        payload["ids"] = tuple(payload["ids"])
        payload["positions"] = {name: tuple(indices) for name, indices in payload["positions"].items()}
        return cls(**payload)


def _row(record: TrainingRecord, mode: str, decision: str, rendered: View) -> ViewRow:
    return ViewRow(
        view_id=f"{record.id}|{mode}",
        record_id=record.id,
        split=record.split,
        mode=mode,
        decision=decision,
        ids=rendered.ids,
        prompt_length=rendered.prompt_length,
        positions=rendered.positions,
    )


def record_views(
    record: TrainingRecord,
    bases: Mapping[str, str],
    index: ScenarioIndex,
    view: GeneratorView,
    prompt: str = c.PROMPT_BASES,
    modes: Sequence[str] = (NO_THINKING, THINKING),
) -> list[ViewRow]:
    """A record with a reply in progress is read at the draft's end; any other record at the reply start, in both
    thinking modes."""
    spec = record_spec(record, bases, index.lookup(record), prompt)
    if record.state.draft is not None:
        rendered = view.render_draft(spec, record.state.draft, key=record.id)
        return [_row(record, DRAFT, c.CONTINUATION_END, rendered)]
    return [
        _row(record, mode, c.REPLY_START, view.render(spec, thinking=mode == THINKING, key=record.id)) for mode in modes
    ]


def build_views(
    records: Iterable[TrainingRecord],
    bases: Mapping[str, str],
    index: ScenarioIndex,
    view: GeneratorView,
    prompt: str = c.PROMPT_BASES,
    modes: Sequence[str] = (NO_THINKING, THINKING),
) -> Iterator[ViewRow | str]:
    """Yields each view, or the id of a record that could not be rendered (a rule without a base)."""
    for record in records:
        try:
            yield from record_views(record, bases, index, view, prompt, modes)
        except MissingBase:
            yield record.id


def load_all_records(data_dir: Path) -> dict[str, TrainingRecord]:
    """Every record of a data directory by id: the training and evaluation splits and the gold set, when present."""
    files = [TRAIN_FILE, *EVAL_FILES.values(), GOLD_FILE]
    return {
        record.id: record for name in files if (data_dir / name).exists() for record in load_records(data_dir / name)
    }


def viewed_records(views: Path) -> set[str]:
    with views.open(encoding="utf-8") as handle:
        return {json.loads(line)["record_id"] for line in handle if line.strip()}


def select_records(
    data_dir: Path, train_sample: int | None, splits: Sequence[str], sources: Sequence[str] = ()
) -> list[TrainingRecord]:
    """Training records (optionally sampled) and the evaluation splits, from `sources` only when given."""

    def wanted(record: TrainingRecord) -> bool:
        return not sources or record.source in sources

    train = [record for record in load_records(data_dir / TRAIN_FILE) if wanted(record)]
    if train_sample is not None and train_sample < len(train):
        train = random.Random(SAMPLE_SEED).sample(train, train_sample)
    evaluated = [record for split in splits for record in load_records(data_dir / EVAL_FILES[split])]
    return train + [record for record in evaluated if wanted(record)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render records the way Qwen3.5 sees them before replying.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bases", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-sample", type=int, default=None)
    parser.add_argument("--splits", nargs="*", default=list(EVAL_FILES), choices=list(EVAL_FILES))
    parser.add_argument("--sources", nargs="*", default=[], help="keep only these record sources")
    parser.add_argument("--prompt", choices=c.PROMPTS, default=c.PROMPT_BASES)
    parser.add_argument("--modes", nargs="+", choices=(NO_THINKING, THINKING), default=[NO_THINKING, THINKING])
    parser.add_argument("--no-drafts", action="store_true", help="leave out records with a reply in progress")
    parser.add_argument("--chat-template", type=Path, default=None, help="render with this template instead")
    parser.add_argument("--ids-from", type=Path, default=None, help="keep only the records of these views")
    args = parser.parse_args()
    template = args.chat_template.read_text(encoding="utf-8") if args.chat_template else None
    view = GeneratorView(AutoTokenizer.from_pretrained(c.CHAT_TOKENIZER_REPO), chat_template=template)
    records = select_records(args.data, args.train_sample, args.splits, args.sources)
    if args.ids_from:
        wanted = viewed_records(args.ids_from)
        records = [record for record in records if record.id in wanted]
    if args.no_drafts:
        records = [record for record in records if record.state.draft is None]
    counts: Counter[str] = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        index = ScenarioIndex.from_directory(args.synthetic)
        for item in build_views(records, load_bases(args.bases), index, view, args.prompt, args.modes):
            if isinstance(item, str):
                counts["skipped_without_base"] += 1
                continue
            handle.write(item.to_json() + "\n")
            counts[f"{item.split}:{item.mode}"] += 1
    print(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()
