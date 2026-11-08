import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from rein.env import load_env, optional
from rein.model.backbone import load_text_backbone
from rein.model.batch import collate
from rein.model.kernels import kernel_report
from rein.train.config import TrainConfig
from rein.train.data import token_batches
from rein.train.loss import decision_loss
from rein.train.toy import toy_records
from rein.train.trainer import build_decider

SMOKE_RECORDS = 64
SMOKE_FILLER_TURNS = 40
SMOKE_STEPS = 20
WARMUP_STEPS = 3


def run(base_repo: str, out: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("the Pod smoke test needs CUDA")
    token = optional(load_env(), "HF_TOKEN")
    config = TrainConfig(run_name="smoke", base_repo=base_repo, train_file=out / "unused.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(base_repo, token=token)
    backbone = load_text_backbone(base_repo, dtype=torch.bfloat16, token=token)
    decider, _ = build_decider(config, backbone, tokenizer)
    device = torch.device("cuda")
    decider.to(device).train()
    optimizer = torch.optim.AdamW([p for p in decider.parameters() if p.requires_grad], lr=1e-5)

    records = toy_records("train", SMOKE_RECORDS, filler_turns=SMOKE_FILLER_TURNS)
    encoded = [decider.encoder.encode(record) for record in records]
    batches = token_batches(encoded, config.batch_tokens, config.max_records_per_batch, random.Random(0))
    tokens = 0
    torch.cuda.reset_peak_memory_stats()
    started = 0.0
    for step in range(SMOKE_STEPS + WARMUP_STEPS):
        if step == WARMUP_STEPS:
            torch.cuda.synchronize()
            started = time.monotonic()
            tokens = 0
        batch = collate([encoded[i] for i in batches[step % len(batches)]], decider.encoder.pad_id).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = decider.logits(batch)
        loss = decision_loss(logits, batch.option_mask, batch.targets, batch.question_owner, config.brier_weight)
        loss.total.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens += int(batch.state_mask.sum()) + int(batch.row_mask.sum())
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    report: dict[str, object] = {
        "base_repo": base_repo,
        "gpu": torch.cuda.get_device_name(0),
        "kernels": {status.name: status.error or "ok" for status in kernel_report()},
        "train_tokens_per_s": round(tokens / elapsed, 1),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "final_loss": float(loss.total),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure real-model training throughput on the Pod GPU.")
    parser.add_argument("--base", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--out", type=Path, default=Path("runs/pod-smoke"))
    args = parser.parse_args()
    print(json.dumps(run(args.base, args.out), indent=2))


if __name__ == "__main__":
    main()
