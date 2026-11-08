import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi

from rein.datagen.assemble import SYNTHETIC_SOURCE
from rein.datagen.mix import SYNTHETIC_SPLITS
from rein.env import load_env, optional
from rein.train.data import load_records

RELEASED_SOURCES = frozenset({SYNTHETIC_SOURCE})
CARD_FILE = "README.md"
STATS_FILE = "stats.json"


def build_release(mix_dir: Path, out_dir: Path, card: Path) -> dict[str, dict[str, int]]:
    """Copies the mix's synthetic records, the only ones the dataset releases, plus the dataset card."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    stats: dict[str, dict[str, int]] = {}
    for split in SYNTHETIC_SPLITS:
        records = [record for record in load_records(mix_dir / f"{split}.jsonl") if record.source in RELEASED_SOURCES]
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.model_dump_json() + "\n")
        stats[split] = dict(Counter(record.source for record in records))
    (out_dir / STATS_FILE).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    shutil.copy(card, out_dir / CARD_FILE)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the Rein decision dataset (private) to the Hub.")
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--message", default=None, help="the commit message (the mix name by default)")
    args = parser.parse_args()
    stats = build_release(args.mix, args.out, args.card)
    print(json.dumps(stats, indent=2))
    api = HfApi(token=optional(load_env(), "HF_TOKEN"))
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo, repo_type="dataset", folder_path=str(args.out), commit_message=args.message or args.mix.name
    )
    print(f"pushed {args.out} to {args.repo} (private dataset)")


if __name__ == "__main__":
    main()
