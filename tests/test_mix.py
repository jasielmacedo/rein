from pathlib import Path

import pytest

from rein.datagen.mix import MixConfig, PublicSource, SyntheticRun, build_mix
from rein.schema import TrainingRecord
from rein.train.toy import toy_records


def _write(path: Path, records: list[TrainingRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8")


def test_mix_merges_runs_tags_prompt_version_and_guards_share(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write(run / "train.jsonl", toy_records("train", 8))
    _write(run / "test.jsonl", toy_records("test", 2))
    public = [
        r.model_copy(update={"id": f"pub-{r.id}", "scenario_id": f"pub-{r.scenario_id}", "source": "massive"})
        for r in toy_records("train", 2)
    ]
    _write(tmp_path / "public.jsonl", public)
    config = MixConfig(
        output_dir=tmp_path / "mix",
        synthetic=(SyntheticRun(path=run, prompt_version="v9"),),
        public=(PublicSource(path=tmp_path / "public.jsonl"),),
    )
    stats = build_mix(config)
    assert stats["synthetic_share_of_train"] == round(10 / 12, 3)
    assert stats["rule_disjoint_moves"] == {"test:to_train": 2}
    first = TrainingRecord.model_validate_json((tmp_path / "mix" / "train.jsonl").read_text("utf-8").splitlines()[0])
    assert first.meta["prompt_version"] == "v9"

    strict = config.model_copy(update={"min_synthetic_share": 0.9})
    with pytest.raises(ValueError, match="synthetic share"):
        build_mix(strict)


def test_scenarios_sharing_a_rule_with_train_move_to_train() -> None:
    from rein.datagen.mix import enforce_rule_disjoint, rule_key

    train = toy_records("train", 2)
    shared = [
        r.model_copy(update={"scenario_id": "dev-shared", "id": f"dev-{r.id}"}) for r in toy_records("development", 2)
    ]
    holdout = [
        r.model_copy(update={"scenario_id": "test-holdout", "id": f"t-{r.id}", "meta": {"holdout": True}})
        for r in toy_records("test", 1)
    ]
    splits = {"train": list(train), "development": shared, "calibration": [], "test": holdout}
    moved = enforce_rule_disjoint(splits)
    assert splits["development"] == [] and splits["test"] == []
    assert moved == {"development:to_train": 2, "test:dropped_holdout": 1}
    assert rule_key("Did the user ask for a PHOTO?!") == rule_key("did the user ask for a photo")
