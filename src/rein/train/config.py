from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rein.model import constants as mc


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_name: str
    base_repo: str
    base_revision: str | None = None
    train_file: Path
    dev_file: Path | None = None
    calibration_file: Path | None = None
    output_dir: Path = Path("runs")

    epochs: int = Field(default=2, ge=1)
    lora_rank: int = Field(default=16, ge=0)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    lora_lr: float = Field(default=5e-5, gt=0)
    head_lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_fraction: float = Field(default=0.1, gt=0, lt=1)
    grad_clip: float = Field(default=1.0, gt=0)
    brier_weight: float = Field(default=0.25, ge=0)

    max_state_tokens: int = Field(default=8192, ge=16)
    batch_tokens: int = Field(default=16384, ge=256)
    max_records_per_batch: int = Field(default=16, ge=1)
    accumulation_steps: int = Field(default=1, ge=1)
    checkpointing: bool = True
    bf16: bool = True
    require_fast_kernels: bool = True

    head_dim: int = Field(default=256, ge=8)
    head_heads: int = Field(default=4, ge=1)
    set_block: bool = True
    readout: Literal["set", "labels"] = mc.READOUT_SET
    adapter_scope: Literal["state_and_rows", "rows"] = mc.ADAPTER_STATE_AND_ROWS
    soft_tokens_front: int = Field(default=0, ge=0)
    soft_tokens_tail: int = Field(default=0, ge=0)
    soft_lr: float = Field(default=1e-3, gt=0)
    soft_silence_margin: float = Field(default=0.0, ge=0)
    soft_silence_weight: float = Field(default=1.0, ge=0)
    soft_silence_records: int = Field(default=256, ge=1)
    soft_silence_positions: int = Field(default=32768, ge=1)
    soft_silence_rank: int = Field(default=0, ge=0)
    soft_silence_tail: float = Field(default=1.0, gt=0, le=1.0)
    readout_layer: int | None = Field(default=None, ge=1)
    pad_state_multiple: int = Field(default=1, ge=1)
    pad_row_multiple: int = Field(default=1, ge=1)
    pad_row_count_multiple: int = Field(default=1, ge=1)

    criteria_dropout: float = Field(default=0.3, ge=0, le=1)
    option_subsample: float = Field(default=0.5, ge=0, le=1)
    seed: int = 0
    log_every_steps: int = Field(default=20, ge=1)
    step_timeout_s: int = Field(default=600, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    resume_every_steps: int | None = Field(default=200, ge=1)

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
