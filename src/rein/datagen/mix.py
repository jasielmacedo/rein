import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rein.schema import TrainingRecord
from rein.train.data import load_records

SYNTHETIC_SPLITS = ("train", "development", "calibration", "test")
PUBLIC_SPLIT = "train"


class SyntheticRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    prompt_version: str


class PublicSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    limit: int | None = Field(default=None, ge=1)


class MixConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output_dir: Path
    synthetic: tuple[SyntheticRun, ...] = Field(min_length=1)
    public: tuple[PublicSource, ...] = ()
    seed: int = 0
    min_synthetic_share: float = Field(default=0.7, gt=0, le=1)


def _synthetic(run: SyntheticRun) -> dict[str, list[TrainingRecord]]:
    by_split: dict[str, list[TrainingRecord]] = {}
    for split in SYNTHETIC_SPLITS:
        path = run.path / f"{split}.jsonl"
        records = load_records(path) if path.exists() else []
        by_split[split] = [
            record.model_copy(update={"meta": {"prompt_version": run.prompt_version, **record.meta}})
            for record in records
        ]
    return by_split


_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def rule_key(text: str) -> str:
    return _NON_WORD.sub(" ", text.lower()).strip()


def _rule_keys(record: TrainingRecord) -> set[str]:
    return {rule_key(question.instructions) for question in record.questions.values()}


def enforce_rule_disjoint(splits: dict[str, list[TrainingRecord]]) -> dict[str, int]:
    """Moves every held-out scenario that shares a rule text with train into train (holdout scenarios are dropped).

    Repeats until stable, since each moved scenario brings its other rules into train.
    """
    moved: Counter[str] = Counter()
    changed = True
    while changed:
        train_rules = {key for record in splits[PUBLIC_SPLIT] for key in _rule_keys(record)}
        changed = False
        for split in SYNTHETIC_SPLITS[1:]:
            leaking = {record.scenario_id for record in splits[split] if _rule_keys(record) & train_rules}
            if not leaking:
                continue
            changed = True
            kept = [record for record in splits[split] if record.scenario_id not in leaking]
            for record in splits[split]:
                if record.scenario_id not in leaking:
                    continue
                if record.meta.get("holdout"):
                    moved[f"{split}:dropped_holdout"] += 1
                else:
                    splits[PUBLIC_SPLIT].append(record.model_copy(update={"split": PUBLIC_SPLIT}))
                    moved[f"{split}:to_train"] += 1
            splits[split] = kept
    return dict(moved)


def build_mix(config: MixConfig) -> dict[str, object]:
    splits: dict[str, list[TrainingRecord]] = defaultdict(list)
    for run in config.synthetic:
        for split, records in _synthetic(run).items():
            splits[split].extend(records)
    moved = enforce_rule_disjoint(splits)
    synthetic_train = len(splits[PUBLIC_SPLIT])
    rng = random.Random(config.seed)
    for source in config.public:
        records = load_records(source.path)
        if source.limit is not None and source.limit < len(records):
            records = rng.sample(records, source.limit)
        splits[PUBLIC_SPLIT].extend(records)
    share = synthetic_train / max(1, len(splits[PUBLIC_SPLIT]))
    if share < config.min_synthetic_share:
        raise ValueError(f"synthetic share of train is {share:.2f}, below {config.min_synthetic_share}")
    ids = [record.id for records in splits.values() for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("record ids collide across runs")
    train_scenarios = {record.scenario_id for record in splits[PUBLIC_SPLIT]}
    for split in SYNTHETIC_SPLITS[1:]:
        if train_scenarios & {record.scenario_id for record in splits[split]}:
            raise ValueError(f"scenarios leak from train into {split}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, object] = {"synthetic_share_of_train": round(share, 3), "rule_disjoint_moves": moved}
    for split, records in splits.items():
        with (config.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.model_dump_json() + "\n")
        stats[split] = {
            "records": len(records),
            "sources": dict(Counter(record.source for record in records)),
            "languages": dict(Counter(str(record.meta.get("lang_conv")) for record in records)),
        }
    (config.output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge synthetic runs and public grounding data into the final mix.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = MixConfig.model_validate(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    print(json.dumps(build_mix(config), indent=2))


if __name__ == "__main__":
    main()
