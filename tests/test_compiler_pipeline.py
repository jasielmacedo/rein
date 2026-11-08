import math
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase, Qwen3_5TextModel

from rein.compiler.model import Compiler
from rein.compiler.reader import RuleReader, RuleReading
from rein.compiler.readings import load_readings, read_all, rules_with_bases, save_readings
from rein.compiler.report import layer_report, layer_temperatures
from rein.compiler.table import Item, StateTable, table_items
from rein.compiler.train import TrainSettings, predict_logits, train_compiler
from rein.genview.build import record_views
from rein.genview.keys import rule_key
from rein.genview.render import GeneratorView
from rein.genview.scenarios import ScenarioIndex
from rein.model.encode import RequestEncoder
from rein.neurons.extract import StateWriter, extract
from rein.schema import Message, Question, State, TrainingRecord
from tests.conftest import TINY_HIDDEN

TOY_HIDDEN = 12
TOY_RULES = 40
TOY_STATES = 30
TOY_DIM, TOY_HEADS, TOY_LAYERS, TOY_RANK = 32, 4, 2, 12
HELD_OUT = 8
TOY_ACCURACY = 0.8
LAYERS = (1, 3)


def _toy_items(rules: torch.Tensor, states: torch.Tensor, first_rule: int, last_rule: int, split: str) -> list[Item]:
    """Rule r says yes when its own direction points with the state: learnable only by reading the rule."""
    items = []
    for rule in range(first_rule, last_rule):
        for row in range(states.shape[0]):
            yes = float(rules[rule] @ states[row, 0]) > 0
            items.append(
                Item(
                    row=row,
                    reading=rule,
                    question_id=f"q{rule}",
                    question_type="noul",
                    option_keys=("no", "yes"),
                    label=int(yes),
                    target=(0.0, 1.0) if yes else (1.0, 0.0),
                    null=0,
                    record_id=f"s{row}",
                    split=split,
                    mode="nothink",
                    kind="reply_start",
                    tags={"split": split},
                )
            )
    return items


def test_the_compiler_learns_to_read_rules_it_never_saw() -> None:
    torch.manual_seed(0)
    directions = torch.nn.functional.normalize(torch.randn(TOY_RULES, TOY_HIDDEN), dim=-1)
    options = torch.stack([torch.zeros(TOY_RULES, TOY_HIDDEN), directions], dim=1)
    readings = RuleReading(
        directions, options, torch.ones(TOY_RULES, 2, dtype=torch.bool), (("no", "yes"),) * TOY_RULES
    )
    states = torch.randn(TOY_STATES, 1, TOY_HIDDEN)
    states[:, 0, -1] = 1.0
    channels = torch.zeros(1, TOY_HIDDEN)
    channels[0, -1] = 1.0
    train = _toy_items(directions, states, 0, TOY_RULES - HELD_OUT, "train")
    unseen = _toy_items(directions, states, TOY_RULES - HELD_OUT, TOY_RULES, "development")
    compiler = Compiler(TOY_HIDDEN, (0,), dim=TOY_DIM, heads=TOY_HEADS, layers=TOY_LAYERS, rank=TOY_RANK)
    settings = TrainSettings(epochs=60, rows_per_batch=8, learning_rate=3e-3, silence_weight=0.0)
    train_compiler(compiler, readings, states, channels, train, [], settings)
    logits, _ = predict_logits(compiler, readings, states, channels, unseen)
    predicted = logits[:, 0].argmax(-1)
    accuracy = float((predicted == torch.tensor([item.label for item in unseen])).float().mean())
    assert accuracy >= TOY_ACCURACY


def _record(number: int, text: str, fires: bool, split: str) -> TrainingRecord:
    return TrainingRecord(
        id=f"rec{number}",
        source="synthetic",
        scenario_id=f"sc{number}",
        split=split,
        state=State(context="A tavern keeper.", conversation=(Message(role="user", text=text),)),
        questions={
            "politics": Question(type="noul", instructions="Is the visitor asking about politics?", label=fires),
            "tool": Question(
                type="choice",
                instructions="Which tool is needed?",
                criteria={"give_item": "hand over an item", "none": None},
                label="none",
            ),
        },
        meta={"case_form": "hard_negative" if not fires else "direct", "category": "age_rating"},
    )


def test_the_stage_one_pipeline_runs_end_to_end_on_cpu(
    tiny_backbone: Qwen3_5TextModel,
    chat_tokenizer: PreTrainedTokenizerBase,
    encoder: RequestEncoder,
    tmp_path: Path,
) -> None:
    records = [
        _record(number, text, fires, split)
        for number, (text, fires, split) in enumerate(
            [
                ("Who should win the election?", True, "train"),
                ("A pint of ale, please.", False, "train"),
                ("What do you think of the president?", True, "calibration"),
                ("Tell me about your cat.", False, "calibration"),
                ("Which party do you vote for?", True, "development"),
                ("Is the soup warm?", False, "development"),
            ]
        )
    ]
    bases = {
        rule_key(question): "You keep to the house rules."
        for record in records
        for question in record.questions.values()
    }
    view = GeneratorView(chat_tokenizer, silence_positions=2)
    views = [row for record in records for row in record_views(record, bases, ScenarioIndex({}), view)]
    assert {row.mode for row in views} == {"nothink", "think"}

    states_dir = tmp_path / "states"
    extract(tiny_backbone, views, LAYERS, StateWriter(states_dir, shard_rows=4), pad_id=0)
    (states_dir / "extraction.json").write_text(
        f'{{"model": "tiny", "layers": [1, 3], "hidden_size": {TINY_HIDDEN}, "views": {len(views)}}}', encoding="utf-8"
    )
    rules = rules_with_bases(records, bases)
    reading = read_all(RuleReader(tiny_backbone, encoder), list(rules.values()), batch_rules=2)
    save_readings(tmp_path / "readings", list(rules), reading, layer=3)
    reading_index, readings = load_readings(tmp_path / "readings")

    table = StateTable.load(states_dir)
    decisions, silences = table_items(table, {record.id: record for record in records}, reading_index)
    assert decisions and silences
    assert all(item.label == item.null for item in silences)

    channels = torch.stack([torch.nn.functional.normalize(torch.randn(TINY_HIDDEN), dim=0) for _ in LAYERS])
    compiler = Compiler(TINY_HIDDEN, LAYERS, dim=16, heads=2, layers=1, rank=8)
    train = [item for item in decisions if item.split == "train"]
    quiet = [item for item in silences if item.split == "train"]
    train_compiler(compiler, readings, table.states, channels, train, quiet, TrainSettings(epochs=2, rows_per_batch=2))

    calibration = [item for item in decisions if item.split == "calibration"]
    cal_logits, cal_masks = predict_logits(compiler, readings, table.states, channels, calibration)
    temperatures = layer_temperatures(cal_logits, cal_masks, calibration)
    assert len(temperatures) == len(LAYERS) and all(math.isfinite(value) for value in temperatures)

    development = [item for item in decisions if item.split == "development"]
    dev_quiet = [item for item in silences if item.split == "development"]
    logits, masks = predict_logits(compiler, readings, table.states, channels, development)
    quiet_logits, quiet_masks = predict_logits(compiler, readings, table.states, channels, dev_quiet)
    report = layer_report(
        LAYERS, logits, masks, development, temperatures, (quiet_logits, quiet_masks, dev_quiet), table.states, channels
    )
    assert set(report) == {"1", "3"}
    assert {"nothink:reply_start", "think:reply_start"} <= set(report["1"]["positions"])
    assert 0.0 <= report["3"]["false_fire_at_gate"] <= 1.0
    assert "variation" in report["1"]["bias_channel"]
