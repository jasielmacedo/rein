import json
import random
from dataclasses import replace
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from rein.model.backbone import load_text_backbone
from rein.module.act import (
    ActSettings,
    Example,
    load_examples,
    load_teacher,
    read_size,
    train_action,
    writes_residual,
)
from rein.module.install import install_module
from rein.module.losses import bypassed, decision_rows, last_hidden, log_probs_at, reply_log_probs, stacked
from rein.module.opening_loss import Opening, opening_loss
from rein.module.targets import LabelTable
from rein.module.train import (
    MODULE_FILE,
    ModulePlace,
    module_parameters,
    module_place,
    new_head,
    prepared_backbone,
    save_weights,
)
from rein.neurons.layers import insert_layers
from rein.neurons.tiny import tiny_qwen_checkpoint

VOCAB = 64
SHARD = "100KB"
PERIOD = 2
CATEGORIES = ("lore", "tool")
PROMPT_TOKENS = 10
REPLY_TOKENS = 4
EXAMPLES = 6
TEST_BATCH_TOKENS = 64
TEST_LR = 1e-3
OPENING_EPOCHS = 3
CALL = (3, 4)
AVOIDED = 5


def _tokens(rng: random.Random, count: int) -> tuple[int, ...]:
    return tuple(rng.randrange(1, VOCAB) for _ in range(count))


def _examples(rng: random.Random, replies: int) -> list[Example]:
    return [
        Example(f"r{n}", _tokens(rng, PROMPT_TOKENS), tuple(_tokens(rng, REPLY_TOKENS) for _ in range(replies)))
        for n in range(EXAMPLES)
    ]


def test_each_kind_of_pair_weighs_the_same_in_total(tmp_path: Path) -> None:
    kinds = ["unneeded_call"] * 6 + ["needed_call"] * 2
    rows = [
        {"record_id": f"r{n}", "prompt": [1, 2], "chosen": [3], "rejected": [4], "kind": kind}
        for n, kind in enumerate(kinds)
    ]
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    examples = load_examples([path], ("chosen", "rejected"), "kind")
    totals = {kind: sum(e.weight for e, k in zip(examples, kinds, strict=True) if k == kind) for kind in set(kinds)}
    assert totals["unneeded_call"] == totals["needed_call"]
    assert sum(example.weight for example in examples) / len(examples) == 1.0


def test_the_action_trains_only_the_module_and_bypassing_it_gives_qwen(tmp_path: Path) -> None:
    source = tiny_qwen_checkpoint(tmp_path / "original", VOCAB, SHARD)
    checkpoint = tmp_path / "inserted"
    insert_layers(source, checkpoint, at=PERIOD, group=PERIOD)
    place = module_place(checkpoint)
    backbone = prepared_backbone(checkpoint, place, torch.device("cpu"), torch.float32, whole=True)
    table = LabelTable(CATEGORIES, {f"r{n}": (n % 2, 0) for n in range(EXAMPLES)})
    settings = ActSettings(
        lr=TEST_LR, batch_tokens=TEST_BATCH_TOKENS, accumulation=1, log_every_steps=1, resume_every_steps=1
    )
    head = new_head(backbone.config.hidden_size, len(CATEGORIES), settings.seed)
    start = tmp_path / "start"
    start.mkdir()
    save_weights(start, module_parameters(backbone), head)
    rng = random.Random(0)
    pairs, anchors = _examples(rng, 2), _examples(rng, 1)
    probe = stacked([(anchors[0].prompt, anchors[0].replies[0])], 0, random.Random(0), 0)
    with torch.no_grad():
        before = reply_log_probs(backbone, probe)
        with bypassed(backbone, place.layers):
            assert torch.equal(reply_log_probs(backbone, probe), before)
    frozen = {name: param.detach().clone() for name, param in backbone.named_parameters() if not param.requires_grad}
    logged: list[dict] = []
    out = tmp_path / "act"
    progress = train_action(backbone, place, head, pairs, anchors, table, settings, out, logged.append)
    assert progress.step > 0
    assert any("kl" in entry for entry in logged) and any("prompt_kl" in entry for entry in logged)
    for name, param in backbone.named_parameters():
        if name in frozen:
            assert torch.equal(param, frozen[name]), name
    original = load_text_backbone(str(source), dtype=torch.float32).eval()
    with torch.no_grad():
        trained = reply_log_probs(backbone, probe)
        assert not torch.allclose(trained, before)
        with bypassed(backbone, place.layers):
            assert torch.allclose(reply_log_probs(backbone, probe), reply_log_probs(original, probe), atol=1e-6)
    installed = tmp_path / "installed"
    assert install_module(checkpoint, out / MODULE_FILE, installed) == len(module_parameters(backbone))
    model = AutoModelForCausalLM.from_pretrained(installed, dtype=torch.float32).eval()
    with torch.no_grad():
        logits = model(input_ids=probe.ids).logits[0, probe.columns]
    assert torch.allclose(torch.log_softmax(logits, dim=-1), trained, atol=1e-5)


def test_the_teacher_labels_only_the_views_it_read(tmp_path: Path) -> None:
    path = tmp_path / "teacher.jsonl"
    path.write_text(json.dumps({"view_id": "a", "record_id": "r0", "fire": {"tool": 0.75}}) + "\n", encoding="utf-8")
    teacher = load_teacher(path, CATEGORIES)
    assert teacher == {"a": (0.0, 0.75)}
    two = _examples(random.Random(0), 1)[:2]
    examples = [replace(example, view_id=view_id) for example, view_id in zip(two, "ab", strict=True)]
    kept, labels = decision_rows(examples, LabelTable(CATEGORIES, {}), teacher)
    assert kept == [0]
    assert labels.tolist() == [[0.0, 0.75]]


def test_the_read_inside_the_module_learns_from_the_teacher(tmp_path: Path) -> None:
    source = tiny_qwen_checkpoint(tmp_path / "original", VOCAB, SHARD)
    checkpoint = tmp_path / "inserted"
    insert_layers(source, checkpoint, at=PERIOD, group=PERIOD)
    place = module_place(checkpoint)
    backbone = prepared_backbone(checkpoint, place, torch.device("cpu"), torch.float32, whole=True)
    settings = ActSettings(
        lr=TEST_LR, batch_tokens=TEST_BATCH_TOKENS, accumulation=1, log_every_steps=1, read_inside=True
    )
    head = new_head(read_size(backbone, inside=True), len(CATEGORIES), settings.seed)
    assert head.in_features == backbone.config.intermediate_size
    rng = random.Random(0)
    pairs = [replace(example, view_id=f"p{n}") for n, example in enumerate(_examples(rng, 2))]
    anchors = [replace(example, view_id=f"a{n}") for n, example in enumerate(_examples(rng, 1))]
    teacher = {example.view_id: (0.9, 0.1) for example in (*pairs[::2], *anchors)}
    before = head.weight.detach().clone()
    logged: list[dict] = []
    table = LabelTable(CATEGORIES, {})
    progress = train_action(
        backbone, place, head, pairs, anchors, table, settings, tmp_path / "act", logged.append, teacher
    )
    assert progress.step > 0 and any(entry.get("decision", 0) > 0 for entry in logged)
    assert not torch.equal(head.weight, before)


def _inserted_backbone(tmp_path: Path) -> tuple[torch.nn.Module, ModulePlace]:
    source = tiny_qwen_checkpoint(tmp_path / "original", VOCAB, SHARD)
    checkpoint = tmp_path / "inserted"
    insert_layers(source, checkpoint, at=PERIOD, group=PERIOD)
    place = module_place(checkpoint)
    return prepared_backbone(checkpoint, place, torch.device("cpu"), torch.float32, whole=True), place


def test_the_opening_loss_is_minus_the_call_s_log_chance_or_of_not_opening_with_it(tmp_path: Path) -> None:
    backbone, _ = _inserted_backbone(tmp_path)
    rng = random.Random(0)
    called = Opening("c", "r", _tokens(rng, PROMPT_TOKENS), CALL, True)
    held = Opening("h", "r", _tokens(rng, PROMPT_TOKENS), (AVOIDED,), False)
    plain = Opening("p", "r", _tokens(rng, PROMPT_TOKENS))
    batch = stacked([(called.prompt, called.call), (held.prompt, ()), (plain.prompt, ())], 0, rng, 0)
    with torch.no_grad():
        hidden = last_hidden(backbone, batch)
        loss = opening_loss(backbone, hidden, batch, [called, held, plain])
        needed = log_probs_at(backbone, hidden, batch.rows, batch.columns).gather(-1, batch.targets[:, None]).sum()
        chance = log_probs_at(backbone, hidden, torch.tensor([1]), batch.openings[1:2])[0, AVOIDED].exp()
    assert torch.allclose(loss, (-needed - torch.log1p(-chance)) / 2)


def test_the_reply_learns_to_open_with_the_needed_call(tmp_path: Path) -> None:
    backbone, place = _inserted_backbone(tmp_path)
    settings = ActSettings(
        epochs=OPENING_EPOCHS, lr=TEST_LR, batch_tokens=TEST_BATCH_TOKENS, accumulation=1, log_every_steps=1
    )
    head = new_head(backbone.config.hidden_size, len(CATEGORIES), settings.seed)
    rng = random.Random(0)
    called = [Opening(f"c{n}", f"r{n}", _tokens(rng, PROMPT_TOKENS), CALL, True) for n in range(EXAMPLES)]
    held = [Opening(f"h{n}", f"r{n}", _tokens(rng, PROMPT_TOKENS), (AVOIDED,), False) for n in range(EXAMPLES)]
    plain = [Opening(f"p{n}", f"r{n}", _tokens(rng, PROMPT_TOKENS)) for n in range(EXAMPLES)]
    calls = stacked([(opening.prompt, opening.call) for opening in called], 0, random.Random(0), 0)
    with torch.no_grad():
        before = reply_log_probs(backbone, calls).gather(-1, calls.targets[:, None]).mean()
    logged: list[dict] = []
    table = LabelTable(CATEGORIES, {})
    openings = [*called, *held, *plain]
    train_action(backbone, place, head, [], [], table, settings, tmp_path / "act", logged.append, None, openings)
    with torch.no_grad():
        after = reply_log_probs(backbone, calls).gather(-1, calls.targets[:, None]).mean()
    assert any("opening" in entry and "opening_prompt_kl" in entry for entry in logged)
    assert after > before


def test_only_the_output_projections_count_as_writing_into_qwen() -> None:
    assert writes_residual("model.language_model.layers.23.self_attn.o_proj.weight")
    assert writes_residual("model.language_model.layers.20.mlp.down_proj.weight")
    assert not writes_residual("model.language_model.layers.23.self_attn.q_proj.weight")
    assert not writes_residual("model.language_model.layers.21.mlp.up_proj.weight")
    assert not writes_residual("model.language_model.layers.23.post_attention_layernorm.weight")
