import random
from pathlib import Path

import torch
from safetensors.torch import load_file

from rein.genview import constants as gc
from rein.genview.build import NO_THINKING, ViewRow
from rein.genview.reasoning import ANSWER_START
from rein.module.master import MasterWeights
from rein.module.targets import LabelTable, batch_targets, option_targets
from rein.module.train import (
    CHECKPOINT_PREFIX,
    HEAD_FILE,
    MODULE_FILE,
    RESUME_FILE,
    ModuleSettings,
    load_resume,
    module_parameters,
    module_place,
    new_head,
    prepared_backbone,
    train_module,
)
from rein.neurons.layers import insert_layers
from rein.neurons.tiny import TINY_LAYER_TYPES, tiny_qwen_checkpoint

VOCAB = 64
SHARD = "100KB"
PERIOD = 2
SILENCES = (1, 2, 3, 4)
CATEGORIES = ("lore", "tool", "name")
SIGNAL_TOKEN = 7
VIEWS = 12
VIEW_TOKENS = 14
TEST_BATCH_TOKENS = 64
TEST_LR = 1e-3


def _view(record_id: str, ids: list[int], positions: dict[str, tuple[int, ...]], split: str = "train") -> ViewRow:
    return ViewRow(
        view_id=f"{record_id}|{NO_THINKING}",
        record_id=record_id,
        split=split,
        mode=NO_THINKING,
        decision=gc.REPLY_START,
        ids=tuple(ids),
        prompt_length=len(ids),
        positions=positions,
    )


def test_each_decision_comes_with_one_quiet_silence() -> None:
    table = LabelTable(CATEGORIES, {"a": (1, 0, 0), "b": (0, 1, 1)})
    batch = [
        _view("a", list(range(10)), {gc.REPLY_START: (9,), gc.SILENCE: SILENCES}),
        _view("b", list(range(12)), {gc.REPLY_START: (8,), ANSWER_START: (11,), gc.SILENCE: SILENCES}),
    ]
    targets = batch_targets(batch, table, random.Random(0))
    assert targets.rows.tolist() == [0, 0, 1, 1, 1, 1]
    assert targets.columns.tolist()[0] == 9 and targets.columns.tolist()[2:4] == [8, 11]
    assert targets.labels.tolist() == [[1, 0, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 0, 0], [0, 0, 0]]
    assert set(targets.columns.tolist()[4:]) <= set(SILENCES)


def test_every_position_owns_one_question_per_category() -> None:
    one_hot, owners = option_targets(torch.tensor([[1, 0], [0, 0]]))
    assert one_hot.tolist() == [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    assert owners.tolist() == [0, 0, 1, 1]


def test_a_teachers_chance_becomes_the_two_option_target() -> None:
    targets, owners = option_targets(torch.tensor([[0.75, 0.0]]))
    assert targets.tolist() == [[0.25, 0.75], [1.0, 0.0]]
    assert owners.tolist() == [0, 0]


def test_master_weights_keep_steps_bf16_would_round_away() -> None:
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    master = MasterWeights({"weight": param})
    param.grad = torch.ones(4, dtype=torch.bfloat16)
    master.pull_gradients()
    assert param.grad is None and master.master[0].grad is not None
    with torch.no_grad():
        master.master[0] -= 1e-4
    master.push_weights()
    assert master.master[0][0].item() != 1.0
    assert param.dtype == torch.bfloat16


def test_master_weights_add_up_gradients_until_the_step() -> None:
    param = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    master = MasterWeights({"weight": param})
    for _ in range(2):
        param.grad = torch.ones(2, dtype=torch.bfloat16)
        master.pull_gradients()
    assert master.master[0].grad is not None and master.master[0].grad.tolist() == [2.0, 2.0]


def _signal_views(rng: random.Random) -> tuple[list[ViewRow], LabelTable]:
    views, labels = [], {}
    for number in range(VIEWS):
        ids = [rng.randrange(SIGNAL_TOKEN + 1, VOCAB) for _ in range(VIEW_TOKENS)]
        fires = number % 2
        if fires:
            ids[rng.randrange(len(SILENCES) + 1, VIEW_TOKENS - 1)] = SIGNAL_TOKEN
        record_id = f"r{number}"
        labels[record_id] = (fires, 0, 0)
        positions = {gc.REPLY_START: (VIEW_TOKENS - 1,), gc.SILENCE: SILENCES}
        views.append(_view(record_id, ids, positions))
    return views, LabelTable(CATEGORIES, labels)


def test_training_changes_only_the_module_and_resumes(tmp_path: Path) -> None:
    source = tiny_qwen_checkpoint(tmp_path / "original", VOCAB, SHARD)
    checkpoint = tmp_path / "inserted"
    insert_layers(source, checkpoint, at=PERIOD, group=PERIOD)
    place = module_place(checkpoint)
    assert list(place.layers) == [PERIOD, PERIOD + 1] and place.read_layer == 2 * PERIOD
    backbone = prepared_backbone(checkpoint, place, torch.device("cpu"), torch.float32)
    assert len(backbone.layers) == 2 * PERIOD + 1 <= len(TINY_LAYER_TYPES) + PERIOD
    frozen = {name: param.detach().clone() for name, param in backbone.named_parameters() if not param.requires_grad}
    trainable = module_parameters(backbone)
    assert trainable and all(name.startswith(f"{CHECKPOINT_PREFIX}layers.") for name in trainable)
    linear_layer, full_layer = (f"{CHECKPOINT_PREFIX}layers.{layer}." for layer in place.layers)
    assert TINY_LAYER_TYPES[0] == "linear_attention" and TINY_LAYER_TYPES[1] == "full_attention"
    assert not any(name.startswith(f"{linear_layer}linear_attn.") for name in trainable)
    assert any(f"{CHECKPOINT_PREFIX}{name}".startswith(f"{linear_layer}linear_attn.out_proj") for name in frozen)
    assert any(name.startswith(f"{linear_layer}mlp.") for name in trainable)
    assert any(name.startswith(f"{full_layer}self_attn.") for name in trainable)
    views, table = _signal_views(random.Random(0))
    settings = ModuleSettings(lr=TEST_LR, batch_tokens=TEST_BATCH_TOKENS, log_every_steps=1, resume_every_steps=1)
    head = new_head(backbone.config.hidden_size, len(table.categories), settings.seed)
    before = {name: param.detach().clone() for name, param in trainable.items()}
    out = tmp_path / "module"
    logged: list[dict] = []
    progress = train_module(backbone, place, head, views, views[:4], table, settings, out, logged.append)
    assert progress.epoch == settings.epochs and progress.best_epoch >= 0 and progress.step > 0
    for name, param in backbone.named_parameters():
        if name in frozen:
            assert torch.equal(param, frozen[name]), name
    assert any(not torch.equal(param, before[name]) for name, param in trainable.items())
    assert set(load_file(str(out / MODULE_FILE))) == set(trainable)
    assert (out / HEAD_FILE).exists() and (out / RESUME_FILE).exists()
    assert any("held_loss" in entry for entry in logged)
    again = new_head(backbone.config.hidden_size, len(table.categories), settings.seed)
    master = MasterWeights(trainable)
    optimizer = torch.optim.AdamW([{"params": master.master}, {"params": list(again.parameters())}])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[1.0, 1.0], total_steps=progress.step)
    resumed = load_resume(out, settings, master, again, optimizer, scheduler)
    assert resumed is not None and resumed.step == progress.step and resumed.epoch == settings.epochs
    assert torch.equal(again.weight, head.weight)
