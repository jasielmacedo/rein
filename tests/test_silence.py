import copy
from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerBase, Qwen3_5TextModel

from rein.model.checkpoint import load_weights, save_checkpoint
from rein.model.soft import SoftPrompt, add_soft_tokens
from rein.schema import DecisionRequest, TrainingRecord
from rein.train.config import TrainConfig
from rein.train.silence import OutputBank, build_output_bank, output_directions, real_logsumexp, silence_loss
from rein.train.toy import toy_records
from rein.train.trainer import build_decider, prepare_silence, run_training
from tests.devices import DEVICE

FRONT = 3
TAIL = 2
RANK = 4
MARGIN = 10.0
MAX_POSITIONS = 12
MAX_TOKENS = 256
TOY_RECORDS = 8
LOUD_SCALE = 50.0


@pytest.fixture
def soft_chat_tokenizer(chat_tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    return copy.deepcopy(chat_tokenizer)


def _bank(
    tiny_backbone: Qwen3_5TextModel, tokenizer: PreTrainedTokenizerBase, request: DecisionRequest, soft_ids: list[int]
) -> OutputBank:
    return build_output_bank(
        tiny_backbone, tokenizer, [request.state, request.state], soft_ids, MAX_POSITIONS, MAX_TOKENS, seed=0
    )


def test_the_bank_holds_output_states_and_the_real_vocabulary_mass(
    tiny_backbone: Qwen3_5TextModel, soft_chat_tokenizer: PreTrainedTokenizerBase, roleplay_request: DecisionRequest
) -> None:
    front, tail = add_soft_tokens(soft_chat_tokenizer, FRONT, TAIL)
    soft_ids = list(front + tail)
    bank = _bank(tiny_backbone, soft_chat_tokenizer, roleplay_request, soft_ids)
    assert bank.hidden.shape == (MAX_POSITIONS, tiny_backbone.config.hidden_size)
    assert bank.real_logsumexp.shape == (MAX_POSITIONS,)
    table = tiny_backbone.get_input_embeddings().weight
    logits = bank.hidden @ table.T.float()
    logits[:, soft_ids] = torch.finfo(logits.dtype).min
    torch.testing.assert_close(bank.real_logsumexp, torch.logsumexp(logits, dim=-1))
    with torch.no_grad():
        table[soft_ids] = LOUD_SCALE
    torch.testing.assert_close(real_logsumexp(bank.hidden, table, soft_ids), bank.real_logsumexp)


def test_silence_loss_is_zero_for_quiet_rows_and_positive_for_loud_ones(
    tiny_backbone: Qwen3_5TextModel, soft_chat_tokenizer: PreTrainedTokenizerBase, roleplay_request: DecisionRequest
) -> None:
    front, tail = add_soft_tokens(soft_chat_tokenizer, FRONT, TAIL)
    soft_ids = list(front + tail)
    bank = _bank(tiny_backbone, soft_chat_tokenizer, roleplay_request, soft_ids)
    quiet = torch.zeros(len(soft_ids), tiny_backbone.config.hidden_size, device=DEVICE)
    loss, gap = silence_loss(bank, quiet, MARGIN)
    assert gap > 0
    loud = bank.hidden[: len(soft_ids)] * LOUD_SCALE / bank.hidden[: len(soft_ids)].norm(dim=1, keepdim=True)
    loud = loud.clone().requires_grad_(True)
    loss, gap = silence_loss(bank, loud, MARGIN)
    assert gap < 0 and loss > MARGIN
    loss.backward()
    assert loud.grad is not None and bool((loud.grad != 0).any())
    one_loud = torch.cat([loud[:1].detach(), quiet[1:]])
    everywhere, _ = silence_loss(bank, one_loud, MARGIN)
    worst_only, _ = silence_loss(bank, one_loud, MARGIN, tail=1.0 / MAX_POSITIONS)
    assert worst_only > everywhere
    gaps = bank.real_logsumexp - (bank.hidden @ one_loud.T).logsumexp(-1)
    assert worst_only.item() == pytest.approx(torch.relu(MARGIN - gaps).max().item(), rel=1e-4)


def test_output_directions_are_orthonormal_and_ordered_by_energy(
    tiny_backbone: Qwen3_5TextModel, soft_chat_tokenizer: PreTrainedTokenizerBase, roleplay_request: DecisionRequest
) -> None:
    front, tail = add_soft_tokens(soft_chat_tokenizer, FRONT, TAIL)
    bank = _bank(tiny_backbone, soft_chat_tokenizer, roleplay_request, list(front + tail))
    basis = output_directions(bank.hidden, RANK)
    assert basis.shape == (RANK, tiny_backbone.config.hidden_size)
    torch.testing.assert_close(basis @ basis.T, torch.eye(RANK, device=basis.device), atol=1e-4, rtol=0)
    energy = (bank.hidden @ basis.T).square().mean(dim=0)
    assert bool((energy[:-1] >= energy[1:]).all())
    assert energy[0] > (bank.hidden @ torch.randn_like(basis[0]).div(basis.shape[1] ** 0.5)).square().mean()


def test_a_run_with_a_silence_rank_saves_and_restores_its_basis(
    tiny_backbone: Qwen3_5TextModel, soft_chat_tokenizer: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    config = _config(tmp_path).model_copy(update={"soft_silence_rank": RANK})
    records: list[TrainingRecord] = toy_records("train", TOY_RECORDS)
    bank = prepare_silence(config, tiny_backbone, soft_chat_tokenizer, records)
    decider, peft_model = build_decider(config, tiny_backbone, soft_chat_tokenizer, bank)
    assert decider.soft is not None and decider.soft.basis.shape == (RANK, tiny_backbone.config.hidden_size)
    run_training(config, decider, peft_model, records, log=lambda entry: None, silence=bank)
    save_checkpoint(tmp_path / "run", None, decider.head, {}, decider.soft)
    restored = SoftPrompt(decider.soft.ids.tolist(), tiny_backbone.get_input_embeddings()).to(DEVICE)
    load_weights(tmp_path / "run", None, decider.head, DEVICE, restored)
    torch.testing.assert_close(restored.rows, decider.soft.rows)
    torch.testing.assert_close(restored.rows @ decider.soft.basis.T, torch.zeros(FRONT + TAIL, RANK, device=DEVICE))


def _config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        run_name="silence",
        base_repo="tiny",
        train_file=tmp_path / "unused.jsonl",
        output_dir=tmp_path,
        epochs=2,
        lora_rank=0,
        soft_tokens_front=FRONT,
        soft_tokens_tail=TAIL,
        soft_lr=5e-2,
        soft_silence_margin=MARGIN,
        soft_silence_records=4,
        soft_silence_positions=MAX_POSITIONS,
        readout="labels",
        adapter_scope="rows",
        batch_tokens=4096,
        max_records_per_batch=4,
        checkpointing=False,
        bf16=False,
        criteria_dropout=0.0,
        log_every_steps=1,
    )


def test_training_with_the_silence_term_pushes_the_soft_rows_quiet(
    tiny_backbone: Qwen3_5TextModel, soft_chat_tokenizer: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    config = _config(tmp_path)
    records: list[TrainingRecord] = toy_records("train", TOY_RECORDS)
    bank = prepare_silence(config, tiny_backbone, soft_chat_tokenizer, records)
    assert bank is not None
    decider, peft_model = build_decider(config, tiny_backbone, soft_chat_tokenizer)
    assert decider.soft is not None
    with torch.no_grad():
        decider.soft.weight.copy_(bank.hidden[: decider.soft.weight.shape[0]] * LOUD_SCALE)
    before = silence_loss(bank, decider.soft.weight, MARGIN)[1]
    entries: list[dict] = []
    result = run_training(config, decider, peft_model, records, log=entries.append, silence=bank)
    assert before < 0
    assert result.silence_gap is not None and result.silence_gap > before
    assert any("silence" in entry and "gap_min" in entry for entry in entries)
