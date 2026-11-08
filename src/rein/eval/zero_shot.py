"""A run directory for a decider with no trained weights: the base model reads the lettered layout and answers with the
label tokens, so `rein.eval.run_eval` can score what the prompt alone gives."""

import argparse
import json
from pathlib import Path

from safetensors.torch import save_file

from rein.model import constants as mc
from rein.model.checkpoint import CONFIG_FILE, FORMAT_VERSION, HEAD_FILE

DEFAULT_MAX_STATE_TOKENS = 8192
UNCALIBRATED = 1.0


def write_zero_shot_run(out: Path, base_repo: str, readout_layer: int | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "format_version": FORMAT_VERSION,
        "base_repo": base_repo,
        "base_revision": None,
        "lora": None,
        "head": {"dim": mc.HEAD_DIM, "heads": mc.HEAD_HEADS, "set_block": True},
        "readout": mc.READOUT_LABELS,
        "adapter_scope": mc.ADAPTER_ROWS,
        "max_state_tokens": DEFAULT_MAX_STATE_TOKENS,
        "readout_layer": readout_layer,
        "temperature": UNCALIBRATED,
    }
    (out / CONFIG_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    save_file({}, out / HEAD_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a zero-shot decider run directory.")
    parser.add_argument("--base", required=True, help="the Qwen3.5 repo whose own logits decide")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--readout-layer", type=int, default=None)
    args = parser.parse_args()
    write_zero_shot_run(args.out, args.base, args.readout_layer)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
