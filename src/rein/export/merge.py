import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi

from rein.env import load_env, optional
from rein.model.checkpoint import CONFIG_FILE, MERGED_FORMAT_VERSION, save_head
from rein.model.decider import Decider
from rein.model.loader import load_merged, load_unmerged
from rein.train.data import load_records

MAX_PROBABILITY_GAP = 0.02
MIN_DECISION_AGREEMENT = 0.995
VERIFY_RECORDS = 64
MODEL_CARD = "README.md"
EXPORT_DTYPE = torch.bfloat16
HUB_DIR = Path(__file__).parent / "hub"
HUB_FILES = ("modeling_rein.py", "rein_control.py")
HF_CONFIG = "config.json"
AUTO_MAP = {"AutoModel": "modeling_rein.ReinDecider"}
PUSH_IGNORE = ["__pycache__/*", "*.pyc"]


def add_hub_runtime(out_dir: Path, meta: dict[str, Any]) -> None:
    """Makes the export load on its own: `AutoModel.from_pretrained(repo, trust_remote_code=True)`."""
    for name in HUB_FILES:
        shutil.copy(HUB_DIR / name, out_dir / name)
    config = json.loads((out_dir / HF_CONFIG).read_text(encoding="utf-8"))
    config["auto_map"] = AUTO_MAP
    config["rein"] = {
        "format_version": MERGED_FORMAT_VERSION,
        "head": meta["head"],
        "temperature": meta["temperature"],
        "max_state_tokens": meta["max_state_tokens"],
        "readout_layer": meta.get("readout_layer"),
    }
    (out_dir / HF_CONFIG).write_text(json.dumps(config, indent=2), encoding="utf-8")


def export_merged(run_dir: Path, out_dir: Path, device: torch.device, token: str | None) -> dict[str, Any]:
    decider, peft_model, meta = load_unmerged(run_dir, device, token, dtype=torch.float32)
    if peft_model is None:
        raise ValueError(f"{run_dir} has no adapter to merge")
    backbone = peft_model.merge_and_unload().to(EXPORT_DTYPE)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    backbone.save_pretrained(out_dir)
    decider.encoder.tokenizer.save_pretrained(out_dir)
    save_head(out_dir, decider.head)
    exported = {**meta, "format_version": MERGED_FORMAT_VERSION, "merged_from": str(run_dir.name)}
    exported.pop("lora", None)
    (out_dir / CONFIG_FILE).write_text(json.dumps(exported, indent=2, default=str), encoding="utf-8")
    add_hub_runtime(out_dir, exported)
    return exported


def fidelity(reference: Decider, candidate: Decider, records_file: Path, count: int) -> dict[str, float]:
    """How closely two deciders agree on the same records: worst and mean option gap, and same top option."""
    gaps: list[torch.Tensor] = []
    agree = total = 0
    for record in load_records(records_file)[:count]:
        expected = reference.decide(record)
        actual = candidate.decide(record)
        gaps.append((expected.probs - actual.probs).abs()[expected.mask])
        agree += int((expected.probs.argmax(-1) == actual.probs.argmax(-1)).sum())
        total += expected.probs.shape[0]
    every = torch.cat(gaps)
    return {"max_gap": float(every.max()), "mean_gap": float(every.mean()), "decision_agreement": agree / total}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a Rein run into a standalone model and push it to the Hub.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-data", type=Path, required=True)
    parser.add_argument("--repo", help="Hugging Face repo id; the repo is created private")
    parser.add_argument("--card", type=Path, help="model card to upload as README.md")
    args = parser.parse_args()
    token = optional(load_env(), "HF_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = export_merged(args.run, args.out, device, token)
    exact = fidelity(
        load_unmerged(args.run, device, token, dtype=torch.float32)[0],
        load_merged(args.out, device, dtype=torch.float32),
        args.verify_data,
        VERIFY_RECORDS,
    )
    served = fidelity(
        load_unmerged(args.run, device, token)[0], load_merged(args.out, device), args.verify_data, VERIFY_RECORDS
    )
    report: dict[str, object] = {**exact, "served_bf16": served}
    (args.out / "export_fidelity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"exported": str(args.out), "readout_layer": meta.get("readout_layer"), **report}))
    if exact["max_gap"] > MAX_PROBABILITY_GAP or exact["decision_agreement"] < MIN_DECISION_AGREEMENT:
        raise SystemExit(f"merged model drifts from the run: {report}")
    if args.card is not None:
        shutil.copy(args.card, args.out / MODEL_CARD)
    if args.repo:
        api = HfApi(token=token)
        api.create_repo(args.repo, private=True, exist_ok=True)
        message = f"Rein export of {args.run.name}"
        api.upload_folder(
            repo_id=args.repo, folder_path=str(args.out), commit_message=message, ignore_patterns=PUSH_IGNORE
        )
        print(f"pushed {args.out} to {args.repo} (private)")


if __name__ == "__main__":
    main()
