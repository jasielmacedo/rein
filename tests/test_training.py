import copy
from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerBase, Qwen3_5TextModel

from rein.model.checkpoint import load_weights, read_meta
from rein.schema import Split, TrainingRecord
from rein.train.config import TrainConfig
from rein.train.toy import toy_records
from rein.train.trainer import build_decider, run_training, save_run
from tests.devices import ON_GPU

TOY_RECORDS = 12
TOY_EPOCHS = 12


def _toy_records(split: Split) -> list[TrainingRecord]:
    return toy_records(split, TOY_RECORDS)


@pytest.fixture
def toy_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        run_name="toy",
        base_repo="tiny",
        train_file=tmp_path / "unused.jsonl",
        output_dir=tmp_path,
        epochs=TOY_EPOCHS,
        lora_lr=5e-3,
        head_lr=5e-3,
        warmup_fraction=0.1,
        batch_tokens=4096,
        max_records_per_batch=4,
        checkpointing=False,
        bf16=False,
        head_dim=16,
        head_heads=2,
        criteria_dropout=0.0,
        log_every_steps=1,
    )


def test_training_reduces_loss_and_checkpoint_round_trips(
    toy_config: TrainConfig, tiny_backbone: Qwen3_5TextModel, tokenizer: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    decider, peft_model = build_decider(toy_config, tiny_backbone, tokenizer)
    trainable = sum(p.numel() for p in decider.parameters() if p.requires_grad)
    frozen = [n for n, p in decider.backbone.named_parameters() if p.requires_grad and "lora_" not in n]
    assert trainable > 0
    assert not frozen

    result = run_training(
        toy_config,
        decider,
        peft_model,
        _toy_records("train"),
        dev_records=_toy_records("development"),
        calibration_records=_toy_records("calibration"),
        log=lambda entry: None,
    )
    first, last = result.history[0]["loss"], result.history[-1]["loss"]
    assert last < first * 0.5
    assert result.dev_metrics["accuracy"] > 0.9
    assert result.temperature > 0

    save_run(tmp_path / "toy", toy_config, decider, peft_model, result)
    meta = read_meta(tmp_path / "toy")
    assert meta["temperature"] == result.temperature

    reference = decider.decide_encoded([decider.encoder.encode(_toy_records("test")[0])])
    with torch.no_grad():
        for param in decider.head.parameters():
            param.zero_()
    load_weights(tmp_path / "toy", peft_model, decider.head, decider.device)
    restored = decider.decide_encoded([decider.encoder.encode(_toy_records("test")[0])])
    torch.testing.assert_close(restored.probs, reference.probs)


RESUME_EVERY = 3
RESUME_RELATIVE_GAP = 0.05 if ON_GPU else 1e-5
INTERRUPT_AT = 7


def _train(config: TrainConfig, backbone: Qwen3_5TextModel, tokenizer: PreTrainedTokenizerBase, resume_dir: Path):
    decider, peft_model = build_decider(config, backbone, tokenizer)
    result = run_training(
        config, decider, peft_model, _toy_records("train"), log=lambda entry: None, resume_dir=resume_dir
    )
    return decider, result


def test_interrupted_training_resumes_to_the_same_weights(
    toy_config: TrainConfig, tiny_backbone: Qwen3_5TextModel, tokenizer: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    config = toy_config.model_copy(update={"epochs": 3, "resume_every_steps": RESUME_EVERY, "criteria_dropout": 0.3})
    straight, straight_result = _train(config, copy.deepcopy(tiny_backbone), tokenizer, tmp_path / "straight")

    resume_dir = tmp_path / "resumed"
    _train(config.model_copy(update={"max_steps": INTERRUPT_AT}), copy.deepcopy(tiny_backbone), tokenizer, resume_dir)
    resumed, resumed_result = _train(config, copy.deepcopy(tiny_backbone), tokenizer, resume_dir)

    assert resumed_result.optimizer_steps == straight_result.optimizer_steps
    assert [e["step"] for e in resumed_result.history] == [e["step"] for e in straight_result.history]
    start = {
        name: param.detach().clone()
        for name, param in build_decider(config, copy.deepcopy(tiny_backbone), tokenizer)[0].named_parameters()
    }
    expected = dict(straight.named_parameters())
    trained = [name for name, param in resumed.named_parameters() if param.requires_grad]
    gap = sum(float((dict(resumed.named_parameters())[n] - expected[n]).pow(2).sum()) for n in trained) ** 0.5
    moved = sum(float((expected[n].detach().cpu() - start[n].cpu()).pow(2).sum()) for n in trained) ** 0.5
    assert moved > 0
    assert gap / moved < RESUME_RELATIVE_GAP
