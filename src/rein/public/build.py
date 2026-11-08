import argparse
import json
from collections import Counter
from pathlib import Path

from rein.public.common import write_records
from rein.public.sources import (
    BANKING_REPO,
    BANKING_TRAIN,
    CLINC_OUT_OF_SCOPE,
    CLINC_REPO,
    CLINC_TRAIN,
    intent_records,
    kev_decision_v7,
    massive_records,
)

DEFAULT_OUT = Path("data/public")
KEV_COUNT = 4000
BANKING_COUNT = 1000
CLINC_COUNT = 1000
MASSIVE_PER_LOCALE = 170
SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the public grounding datasets to Rein training records.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rejected: Counter[str] = Counter()
    counts = {
        "kev_decision_v7": write_records(
            kev_decision_v7(KEV_COUNT, SEED, rejected), args.out / "kev_decision_v7.jsonl"
        ),
        "banking77": write_records(
            intent_records("banking77", BANKING_REPO, BANKING_TRAIN, BANKING_COUNT, SEED, rejected),
            args.out / "banking77.jsonl",
        ),
        "clinc150": write_records(
            intent_records("clinc150", CLINC_REPO, CLINC_TRAIN, CLINC_COUNT, SEED, rejected, CLINC_OUT_OF_SCOPE),
            args.out / "clinc150.jsonl",
        ),
        "massive": write_records(
            massive_records(MASSIVE_PER_LOCALE, SEED, args.out / "cache", rejected), args.out / "massive.jsonl"
        ),
    }
    report = {"records": counts, "rejected": dict(rejected)}
    (args.out / "stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
