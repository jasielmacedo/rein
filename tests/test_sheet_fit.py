import pytest
import torch

from rein.compiler.holdout import held_out_readings
from rein.compiler.model import Compiler, decision_logits
from rein.compiler.reader import RuleReading
from rein.compiler.sheet_fit import (
    SheetExamples,
    compiled_neurons,
    fit_offsets,
    fit_sheet,
    layer_comparison,
    oracle_tool_rules,
    sheet_loss,
)
from rein.compiler.table import Item
from rein.compiler.tool_binary import ANY_SUFFIX, with_tool_questions
from rein.genview.keys import rule_key
from rein.schema import Message, Question, State, TrainingRecord

HIDDEN = 16
OPTIONS = 3
EXAMPLES = 24
SILENCES = 12
TEMPERATURE = 1.3
NULL = 2
STRONG_PRIOR = 1e6
WEAK_PRIOR = 1e-6
SAME_ATOL = 1e-4
SILENCE_SHIFT = 6.0
TOOLS = Question(
    type="choice",
    instructions="Which tool does the request need?",
    criteria={"ring_bell": "ring the signal bell", "give_item": "hand over an item", "none": None},
    label="ring_bell",
)


def _compiler() -> Compiler:
    torch.manual_seed(0)
    return Compiler(HIDDEN, (5,), dim=8, heads=2, layers=1, rank=4).eval()


def _reading() -> RuleReading:
    torch.manual_seed(1)
    return RuleReading(
        query=torch.randn(1, HIDDEN),
        options=torch.randn(1, OPTIONS, HIDDEN),
        mask=torch.ones(1, OPTIONS, dtype=torch.bool),
        option_keys=(("ring_bell", "give_item", "none"),),
    )


def _examples() -> tuple[SheetExamples, torch.Tensor]:
    torch.manual_seed(2)
    states = torch.randn(EXAMPLES, HIDDEN) + 3.0
    labels = (states[:, 0] > 3.0).long()
    quiet = torch.randn(SILENCES, HIDDEN) + 3.0
    quiet[:, 1] -= SILENCE_SHIFT
    return SheetExamples(states, labels, quiet, NULL), torch.full((HIDDEN,), 1 / HIDDEN)


def test_the_sheet_neurons_are_the_compilers_own_decision() -> None:
    compiler, reading = _compiler(), _reading()
    prior = compiled_neurons(compiler, reading, 0, 0, OPTIONS)
    states, channel = torch.randn(7, HIDDEN), torch.randn(HIDDEN)
    with torch.no_grad():
        compiled = compiler(reading).take(torch.zeros(7, dtype=torch.long))
        expected = decision_logits(compiled, states[:, None], channel[None])[:, 0]
    assert torch.allclose(prior.logits(states, channel), expected, atol=SAME_ATOL)


def test_a_strong_prior_keeps_the_compilers_neurons_and_a_weak_one_fits_the_sheet() -> None:
    prior = compiled_neurons(_compiler(), _reading(), 0, 0, OPTIONS)
    examples, channel = _examples()
    kept = fit_sheet(prior, examples, channel, TEMPERATURE, STRONG_PRIOR)
    assert torch.allclose(kept.detectors, prior.detectors, atol=SAME_ATOL)
    fitted = fit_sheet(prior, examples, channel, TEMPERATURE, WEAK_PRIOR)
    with torch.no_grad():
        before = sheet_loss(prior, examples, channel, TEMPERATURE)
        after = sheet_loss(fitted, examples, channel, TEMPERATURE)
    assert after < before / 2


def test_silences_teach_the_fit_to_stay_on_the_null_option() -> None:
    prior = compiled_neurons(_compiler(), _reading(), 0, 0, OPTIONS)
    examples, channel = _examples()
    fitted = fit_sheet(prior, examples, channel, TEMPERATURE, WEAK_PRIOR)
    with torch.no_grad():
        quiet = torch.softmax(fitted.logits(examples.quiet, channel) / TEMPERATURE, dim=-1)
    assert float(quiet[:, NULL].mean()) == pytest.approx(1.0, abs=0.2)


def _record(record_id: str, scenario: str) -> TrainingRecord:
    return TrainingRecord(
        id=record_id,
        source="synthetic",
        scenario_id=scenario,
        split="train",
        state=State(conversation=(Message(role="user", text="ring it"),)),
        questions={"tool": TOOLS},
    )


def test_held_out_rules_take_their_yes_no_versions_with_them() -> None:
    records = with_tool_questions({"r": _record("r", "s1")}, {"r": "tool"})
    keys = [rule_key(question) for question in records["r"].questions.values()]
    index = {key: position for position, key in enumerate(keys)}
    assert held_out_readings(records.values(), {("s1", "tool")}, index) == set(index.values())
    assert held_out_readings(records.values(), {("s2", "tool")}, index) == set()


def test_oracle_records_find_their_tool_rule_through_the_original_scenario() -> None:
    records = {"o": _record("o", "oracle-s1"), "p": _record("p", "oracle-s9")}
    rules = [{"scenario_id": "s1", "rule_id": "tool", "category": "tool_trigger"}]
    mapping = oracle_tool_rules(records, rules)
    assert mapping == {"o": "tool"}
    assert f"tool{ANY_SUFFIX}" in with_tool_questions(records, mapping)["o"].questions


RECORDS = 40
KEYS = ("ring_bell", "give_item", "none")


def _item(row: int, record: int, label: int, kind: str) -> Item:
    return Item(
        row=row,
        reading=0,
        question_id="tool",
        question_type="choice",
        option_keys=KEYS,
        label=label,
        target=tuple(float(position == label) for position in range(OPTIONS)),
        null=NULL,
        record_id=f"rec{record}",
        split="train",
        mode="nothink",
        kind=kind,
        tags={"target_rule": "tool", "category": "tool_trigger"},
    )


def test_every_way_of_getting_the_neurons_is_scored_on_the_same_held_out_cases() -> None:
    torch.manual_seed(3)
    decisions = torch.randn(RECORDS, HIDDEN)
    labels = [int(value) for value in (decisions[:, 0] > 0).long() + (decisions[:, 1] > 1).long()]
    quiet_states = torch.randn(RECORDS, HIDDEN)
    quiet_states[:, 2] -= SILENCE_SHIFT
    states = torch.cat([decisions, quiet_states])[:, None]
    items = [_item(index, index, labels[index], "reply_start") for index in range(RECORDS)]
    quiet = [_item(RECORDS + index, index, NULL, "silence") for index in range(RECORDS)]
    report = layer_comparison(
        _compiler(), _reading(), items, quiet, states, 0, torch.full((HIDDEN,), 1 / HIDDEN), TEMPERATURE
    )
    assert {"compiler", "fit_4", "fit_8", "fit_16", "fit_all", "oracle"} <= set(report)
    assert report["fit_all"]["l2"] and report["compiler"]["silence_false_fire"]["all"]["n"] > 0


def test_an_offsets_only_fit_keeps_the_detectors_and_moves_the_thresholds() -> None:
    prior = compiled_neurons(_compiler(), _reading(), 0, 0, OPTIONS)
    examples, channel = _examples()
    fitted = fit_offsets(prior, examples, channel, TEMPERATURE, WEAK_PRIOR)
    assert torch.equal(fitted.detectors, prior.detectors)
    assert not torch.allclose(fitted.offsets, prior.offsets)
    with torch.no_grad():
        assert sheet_loss(fitted, examples, channel, TEMPERATURE) < sheet_loss(prior, examples, channel, TEMPERATURE)
