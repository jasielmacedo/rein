import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from rein.env import load_env, optional
from rein.model.backbone import load_text_backbone, placed
from rein.model.kernels import kernel_report, require_fast_kernels
from rein.train.config import TrainConfig
from rein.train.data import load_records
from rein.train.provenance import collect
from rein.train.resume import RESUME_DIR, clear_resume
from rein.train.trainer import build_decider, prepare_silence, run_training, save_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Rein decision model.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = TrainConfig.from_yaml(args.config)
    token = optional(load_env(), "HF_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernels = require_fast_kernels() if device.type == "cuda" and config.require_fast_kernels else kernel_report()
    run_dir = config.output_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance = collect(
        config.model_dump(mode="json"),
        {"train": config.train_file, "dev": config.dev_file, "calibration": config.calibration_file},
    )
    provenance["kernels"] = [asdict(status) for status in kernels]
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(config.base_repo, revision=config.base_revision, token=token)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    backbone = placed(load_text_backbone(config.base_repo, config.base_revision, dtype=dtype, token=token), device)
    train_records = load_records(config.train_file)
    silence = prepare_silence(config, backbone, tokenizer, train_records)
    decider, peft_model = build_decider(config, backbone, tokenizer, silence)
    result = run_training(
        config,
        decider,
        peft_model,
        train_records,
        load_records(config.dev_file) if config.dev_file else (),
        load_records(config.calibration_file) if config.calibration_file else (),
        device=device,
        resume_dir=run_dir / RESUME_DIR,
        silence=silence,
    )
    save_run(run_dir, config, decider, peft_model, result)
    clear_resume(run_dir / RESUME_DIR)
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
