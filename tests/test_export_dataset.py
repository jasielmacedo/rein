import json
from pathlib import Path

from rein.datagen.mix import SYNTHETIC_SPLITS
from rein.export.dataset import CARD_FILE, STATS_FILE, build_release
from rein.schema import Message, Question, State, TrainingRecord

SOURCES = ("synthetic", "banking77", "kev_decision_v7")


def _record(number: int, source: str, split: str) -> TrainingRecord:
    return TrainingRecord(
        id=f"r{number}",
        source=source,
        scenario_id=f"s{number}",
        split=split,
        state=State(conversation=(Message(role="user", text="send me a photo"),)),
        questions={"photo": Question(type="noul", instructions="Did the user ask for a photo?", label=True)},
    )


def test_the_release_holds_only_the_synthetic_records(tmp_path: Path) -> None:
    mix = tmp_path / "mix"
    mix.mkdir()
    for split in SYNTHETIC_SPLITS:
        records = [_record(number, source, split) for number, source in enumerate(SOURCES)]
        (mix / f"{split}.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    card = tmp_path / "card.md"
    card.write_text("# card\n", encoding="utf-8")
    out = tmp_path / "release"
    stats = build_release(mix, out, card)
    assert stats == {split: {"synthetic": 1} for split in SYNTHETIC_SPLITS}
    assert json.loads((out / STATS_FILE).read_text(encoding="utf-8")) == stats
    for split in SYNTHETIC_SPLITS:
        rows = [json.loads(line) for line in (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [row["source"] for row in rows] == ["synthetic"]
    assert (out / CARD_FILE).read_text(encoding="utf-8") == "# card\n"
