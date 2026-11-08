import faulthandler
import json
import math
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from rein.eval.metrics import summarize
from rein.eval.predict import predict_rows
from rein.model import constants as mc
from rein.model.backbone import LoraSettings, attach_lora, freeze, truncate_backbone
from rein.model.batch import BatchPadding, collate
from rein.model.checkpoint import save_checkpoint
from rein.model.decider import Decider
from rein.model.encode import RequestEncoder
from rein.model.head import build_head, layout_for
from rein.model.soft import SoftPrompt, add_soft_tokens, input_embeddings
from rein.schema import TrainingRecord
from rein.train.calibrate import fit_temperature
from rein.train.config import TrainConfig
from rein.train.data import epoch_batches, token_batches
from rein.train.loss import decision_loss
from rein.train.resume import ResumePoint, epoch_rng, load_resume, save_resume
from rein.train.silence import OutputBank, build_output_bank, output_directions, silence_loss

Logger = Callable[[dict[str, Any]], None]
RUNNING_PARTS = 4


@dataclass
class TrainResult:
    history: list[dict[str, Any]] = field(default_factory=list)
    temperature: float = 1.0
    dev_metrics: dict[str, float] = field(default_factory=dict)
    optimizer_steps: int = 0
    silence_gap: float | None = None


def prepare_silence(
    config: TrainConfig,
    backbone: Qwen3_5TextModel,
    tokenizer: PreTrainedTokenizerBase,
    records: Sequence[TrainingRecord],
) -> OutputBank | None:
    """The generator's output states the soft rows must stay silent on; needs the backbone with all its layers."""
    if not config.soft_silence_margin or not (config.soft_tokens_front or config.soft_tokens_tail):
        return None
    front, tail = add_soft_tokens(tokenizer, config.soft_tokens_front, config.soft_tokens_tail)
    sample = random.Random(config.seed).sample(list(records), min(config.soft_silence_records, len(records)))
    return build_output_bank(
        backbone,
        tokenizer,
        [record.state for record in sample],
        front + tail,
        config.soft_silence_positions,
        config.max_state_tokens,
        config.seed,
    )


def build_decider(
    config: TrainConfig,
    backbone: Qwen3_5TextModel,
    tokenizer: PreTrainedTokenizerBase,
    silence: OutputBank | None = None,
) -> tuple[Decider, PeftModel | None]:
    """The decider a config describes: an adapter (rank 0 means none), soft tokens (0 means none), the head. With
    `soft_silence_rank`, the soft rows live off the bank's dominant output directions."""
    backbone = truncate_backbone(backbone, config.readout_layer)
    torch.manual_seed(config.seed)
    peft_model = (
        attach_lora(backbone, LoraSettings(config.lora_rank, config.lora_alpha, config.lora_dropout))
        if config.lora_rank
        else None
    )
    if peft_model is None:
        freeze(backbone)
    front, tail = add_soft_tokens(tokenizer, config.soft_tokens_front, config.soft_tokens_tail)
    soft = SoftPrompt(front + tail, input_embeddings(backbone), seed=config.seed) if front or tail else None
    if soft is not None and config.soft_silence_rank:
        if silence is None:
            raise ValueError("soft_silence_rank needs the silence bank (set soft_silence_margin)")
        soft.set_basis(output_directions(silence.hidden, config.soft_silence_rank))
    torch.manual_seed(config.seed)
    head = build_head(config.readout, backbone, config.head_dim, config.head_heads, config.set_block)
    encoder = RequestEncoder(
        tokenizer,
        max_state_tokens=config.max_state_tokens,
        layout=layout_for(config.readout),
        soft_front=front,
        soft_tail=tail,
    )
    decider = Decider(
        backbone,
        head,
        encoder,
        checkpointing=config.checkpointing,
        adapter_on_state=config.adapter_scope == mc.ADAPTER_STATE_AND_ROWS,
        soft=soft,
    )
    return decider, peft_model


def _optimizer(decider: Decider, config: TrainConfig) -> torch.optim.Optimizer:
    """One group per kind of trained weight, each at its own learning rate: adapter, head, soft tokens."""
    groups: list[dict[str, Any]] = []
    adapter = [param for param in decider.backbone.parameters() if param.requires_grad]
    if adapter:
        groups.append({"params": adapter, "lr": config.lora_lr})
    head = list(decider.head.parameters())
    if head:
        groups.append({"params": head, "lr": config.head_lr})
    if decider.soft is not None:
        groups.append({"params": list(decider.soft.parameters()), "lr": config.soft_lr})
    if not groups:
        raise ValueError("the config trains nothing: no adapter, no head weights, no soft tokens")
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def _calibration_logits(
    decider: Decider, records: Sequence[TrainingRecord], config: TrainConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = [decider.encoder.encode(record) for record in records]
    logits: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    labels: list[int] = []
    width = max(len(question.option_keys) for request in encoded for question in request.questions)
    for batch in token_batches(encoded, config.batch_tokens, config.max_records_per_batch, random.Random(config.seed)):
        output = decider.decide_encoded([encoded[i] for i in batch], temperature=1.0)
        pad = width - output.logits.shape[1]
        logits.append(torch.nn.functional.pad(output.logits, (0, pad), value=torch.finfo(output.logits.dtype).min))
        masks.append(torch.nn.functional.pad(output.mask, (0, pad), value=False))
        labels.extend(
            -1 if question.label_index is None else question.label_index
            for i in batch
            for question in encoded[i].questions
        )
    return torch.cat(logits), torch.cat(masks), torch.tensor(labels, dtype=torch.long)


def run_training(
    config: TrainConfig,
    decider: Decider,
    peft_model: PeftModel | None,
    train_records: Sequence[TrainingRecord],
    dev_records: Sequence[TrainingRecord] = (),
    calibration_records: Sequence[TrainingRecord] = (),
    device: torch.device | None = None,
    log: Logger = print,
    resume_dir: Path | None = None,
    silence: OutputBank | None = None,
) -> TrainResult:
    device = device or decider.device
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    decider.to(device)
    if silence is not None:
        silence = silence.to(device)
    use_bf16 = config.bf16 and device.type == "cuda"

    probe = [decider.encoder.encode(record) for record in train_records]
    batches_per_epoch = len(token_batches(probe, config.batch_tokens, config.max_records_per_batch, random.Random(0)))
    total_steps = max(1, config.epochs * math.ceil(batches_per_epoch / config.accumulation_steps))
    optimizer = _optimizer(decider, config)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group["lr"] for group in optimizer.param_groups],
        total_steps=total_steps,
        pct_start=config.warmup_fraction,
    )

    result = TrainResult()
    step = 0
    config_key = config.model_dump_json(
        exclude={"max_steps", "log_every_steps", "step_timeout_s", "resume_every_steps"}
    )
    point = load_resume(resume_dir, decider, optimizer, scheduler, config_key) if resume_dir else None
    start_epoch = 0
    if point is not None:
        step, start_epoch, result.history = point.step, point.epoch, point.history
        rng = epoch_rng(point)
    log({"event": "ready", "records": len(train_records), "batches_per_epoch": batches_per_epoch, "step": step})
    padding = BatchPadding(config.pad_state_multiple, config.pad_row_multiple, config.pad_row_count_multiple)
    started = time.monotonic()
    for epoch in range(start_epoch, config.epochs):
        if config.max_steps is not None and step >= config.max_steps:
            break
        decider.train()
        epoch_state = rng.getstate()
        resumed = point if point is not None and point.epoch == epoch else None
        saved = list(resumed.running) if resumed else []
        running = torch.tensor(saved + [0.0] * (RUNNING_PARTS - len(saved)), device=device)
        gap = torch.zeros((), device=device)
        micro = resumed.micro_done if resumed else 0
        skip = micro
        for encoded in epoch_batches(
            train_records,
            decider.encoder,
            config.batch_tokens,
            config.max_records_per_batch,
            config.criteria_dropout,
            rng,
            config.option_subsample,
        ):
            if skip:
                skip -= 1
                continue
            faulthandler.dump_traceback_later(config.step_timeout_s, exit=True, file=sys.stderr)
            step_started = time.monotonic()
            batch = collate(encoded, decider.encoder.pad_id, padding).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = decider.logits(batch)
            parts = decision_loss(logits, batch.option_mask, batch.targets, batch.question_owner, config.brier_weight)
            total = parts.total
            quiet = torch.zeros((), device=device)
            if silence is not None and decider.soft is not None:
                quiet, gap = silence_loss(
                    silence, decider.soft.rows, config.soft_silence_margin, config.soft_silence_tail
                )
                total = total + config.soft_silence_weight * quiet
            (total / config.accumulation_steps).backward()
            running += torch.stack([total.detach(), parts.ce.detach(), parts.brier.detach(), quiet.detach()])
            micro += 1
            if micro % config.accumulation_steps:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in decider.parameters() if p.requires_grad], config.grad_clip)
            optimizer.step()
            if step < total_steps - 1:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % config.log_every_steps == 0:
                loss, ce, brier, quiet_mean = (running / micro).tolist()
                entry = {"epoch": epoch, "step": step, "loss": loss, "ce": ce, "brier": brier}
                if silence is not None:
                    entry["silence"] = quiet_mean
                    entry["gap_min"] = round(float(gap), 3)
                entry["elapsed_s"] = round(time.monotonic() - started, 1)
                entry["step_s"] = round(time.monotonic() - step_started, 2)
                entry["shapes"] = [list(batch.state_ids.shape), list(batch.row_ids.shape)]
                result.history.append(entry)
                log(entry)
            if resume_dir and config.resume_every_steps and step % config.resume_every_steps == 0:
                snapshot = ResumePoint(
                    config_key, epoch, micro, step, epoch_state, running.tolist(), list(result.history)
                )
                save_resume(resume_dir, decider, optimizer, scheduler, snapshot)
            if config.max_steps is not None and step >= config.max_steps:
                break
        if dev_records:
            decider.eval()
            rows = predict_rows(decider, dev_records, config.batch_tokens, config.max_records_per_batch)
            result.dev_metrics = summarize(rows)
            log({"epoch": epoch, "dev": result.dev_metrics})

    faulthandler.cancel_dump_traceback_later()
    result.optimizer_steps = step
    decider.eval()
    if silence is not None and decider.soft is not None:
        with torch.no_grad():
            result.silence_gap = float(silence_loss(silence, decider.soft.rows, config.soft_silence_margin)[1])
        log({"silence_gap": result.silence_gap})
    if calibration_records:
        logits, masks, labels = _calibration_logits(decider, calibration_records, config)
        result.temperature = fit_temperature(logits, masks, labels)
        decider.temperature = result.temperature
        log({"temperature": result.temperature})
    return result


def save_run(
    directory: Path, config: TrainConfig, decider: Decider, peft_model: PeftModel | None, result: TrainResult
) -> None:
    lora = {"rank": config.lora_rank, "alpha": config.lora_alpha, "dropout": config.lora_dropout}
    meta = {
        "base_repo": config.base_repo,
        "base_revision": config.base_revision,
        "lora": lora if peft_model is not None else None,
        "head": {"dim": config.head_dim, "heads": config.head_heads, "set_block": config.set_block},
        "readout": config.readout,
        "adapter_scope": config.adapter_scope,
        "soft_tokens": {
            "front": config.soft_tokens_front,
            "tail": config.soft_tokens_tail,
            "silence_margin": config.soft_silence_margin,
            "silence_rank": config.soft_silence_rank,
            "silence_weight": config.soft_silence_weight,
            "silence_tail": config.soft_silence_tail,
        },
        "max_state_tokens": config.max_state_tokens,
        "readout_layer": config.readout_layer,
        "temperature": result.temperature,
    }
    save_checkpoint(directory, peft_model, decider.head, meta, decider.soft)
    metrics = {
        "history": result.history,
        "dev": result.dev_metrics,
        "silence_gap": result.silence_gap,
        "temperature": result.temperature,
        "optimizer_steps": result.optimizer_steps,
    }
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
