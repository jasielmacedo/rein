import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from rein.datagen.labels import resolve
from rein.datagen.models import RuleDraft
from rein.datagen.pipeline import DatagenConfig, Pipeline, SliceOverrides
from rein.datagen.taxonomy import load_taxonomy
from rein.schema import TrainingRecord
from rein.teacher.client import Completion

CATEGORY_LINE = re.compile(r"^- (\w+) \(type: (\w+)")
RULE_LINE = re.compile(r'^- "(\w+)": .*?Options: "(\w+)"(?:.*?; "(\w+)")?')
TURNS = re.compile(r"about (\d+) messages")
DISAGREEING_CATEGORY = "break_character"


class FakeTeacher:
    def __init__(self) -> None:
        self.calls = 0
        self.thinking_calls = 0

    async def chat(
        self, messages: Sequence[Mapping[str, str]], *, max_tokens: int, thinking: bool = False, **kwargs: Any
    ) -> Completion:
        self.calls += 1
        self.thinking_calls += int(thinking)
        prompt = messages[0]["content"]
        if prompt.startswith("You design rule sheets"):
            body = self._scenario(prompt)
        elif prompt.startswith("You write realistic conversations"):
            body = self._case(prompt)
        else:
            body = self._judge(prompt, thinking)
        return Completion(json.dumps(body), 10, 20, "stop", 0.0)

    def _scenario(self, prompt: str) -> dict[str, Any]:
        rules = []
        for line in prompt.splitlines():
            match = CATEGORY_LINE.match(line)
            if not match:
                continue
            category, kind = match.groups()
            criteria: Any = None
            if kind == "choice":
                criteria = {"send_photo": "share an image", "none": None}
            elif kind == "score":
                criteria = ["on tone", "slightly off", "drifting", "far off", "another character"]
            rules.append(
                {
                    "id": f"rule_{category}",
                    "category": category,
                    "type": kind,
                    "instructions": f"Does the latest turn trigger {category}?",
                    "criteria": criteria,
                    "fires_when": "the user does it",
                    "does_not_fire_when": "anything else",
                }
            )
        return {
            "title": "Fake scenario",
            "context": "A barista named Luna works in Lisbon. " * 2,
            "character_name": "Luna",
            "tools": ["send_photo"],
            "rules": rules,
        }

    def _case(self, prompt: str) -> dict[str, Any]:
        turns = int(TURNS.search(prompt).group(1))  # type: ignore[union-attr]
        conversation = [
            {"role": "user" if (turns - i) % 2 == 1 else "assistant", "name": None, "text": f"*message {i}*"}
            for i in range(turns)
        ]
        draft: Any = None
        if 'Set "draft" to null' not in prompt:
            calls = [{"name": "send_photo", "arguments": {}}] if "tool call in" in prompt else []
            draft = {"text": "*sends it*", "tool_calls": calls}
        return {"conversation": conversation, "draft": draft}

    def _judge(self, prompt: str, thinking: bool) -> dict[str, Any]:
        reviewer = prompt.startswith("You are a skeptical reviewer")
        answers = {}
        for line in prompt.splitlines():
            match = RULE_LINE.match(line)
            if match:
                rule_id, first, second = match.groups()
                flip = reviewer and DISAGREEING_CATEGORY in rule_id and second
                answers[rule_id] = second if flip else first
        return {"answers": answers}


def _pipeline(tmp_path: Path, teacher: FakeTeacher) -> Pipeline:
    config = DatagenConfig(run_name="fake", scenarios=6, cases_per_scenario=4, output_dir=tmp_path, seed=3)
    return Pipeline(config, teacher, load_taxonomy())


async def test_pipeline_produces_rule_split_records_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rein.datagen.generate.matches_language", lambda messages, code: True)
    teacher = FakeTeacher()
    stats = await _pipeline(tmp_path, teacher).run()
    assert stats.accepted["record"] > 0

    records = []
    for split_file in (tmp_path / "fake").glob("*.jsonl"):
        if split_file.stem in {"train", "development", "calibration", "test"}:
            records += [TrainingRecord.model_validate_json(line) for line in split_file.read_text("utf-8").splitlines()]
    assert len(records) == stats.accepted["record"]

    splits_by_scenario: dict[str, set[str]] = defaultdict(set)
    for record in records:
        splits_by_scenario[record.scenario_id].add(record.split)
        assert record.state.conversation[-1].role == "user"
        assert record.meta["unseen_rule"] == (record.split == "test")
    assert all(len(splits) == 1 for splits in splits_by_scenario.values())
    assert teacher.thinking_calls > 0
    assert any(question.target is not None for record in records for question in record.questions.values())
    assert any(question.label is not None for record in records for question in record.questions.values())

    calls = teacher.calls
    again = await _pipeline(tmp_path, teacher).run()
    assert teacher.calls == calls
    assert again.accepted["record"] == 0


def test_resolve_gives_hard_labels_only_when_unanimous() -> None:
    rule = RuleDraft(
        id="tool",
        category="tool_trigger",
        type="choice",
        instructions="Which tool is needed now?",
        criteria={"send_photo": None, "none": None},
        fires_when="asks for a photo",
        does_not_fire_when="anything else",
    )
    hard = resolve(rule, ["send_photo", "send_photo", "send_photo"])
    soft = resolve(rule, ["send_photo", "none", "send_photo", "send_photo"])
    assert hard.question.label == "send_photo" and hard.unanimous
    assert soft.question.target == {"send_photo": 0.75, "none": 0.25} and not soft.unanimous


async def test_eval_slice_overrides_force_split_languages_and_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rein.datagen.generate.matches_language", lambda messages, code: True)
    overrides = SliceOverrides(
        conversation_languages={"hi": 1.0},
        length_bands=("long",),
        required_categories=("narrated_action",),
        target_categories=("narrated_action",),
        force_split="test",
    )
    config = DatagenConfig(
        run_name="slice", scenarios=3, cases_per_scenario=3, output_dir=tmp_path, seed=5, overrides=overrides
    )
    await Pipeline(config, FakeTeacher(), load_taxonomy()).run()
    records = [
        TrainingRecord.model_validate_json(line)
        for line in (tmp_path / "slice" / "test.jsonl").read_text("utf-8").splitlines()
    ]
    assert records
    assert {record.split for record in records} == {"test"}
    assert {record.meta["lang_conv"] for record in records} == {"hi"}
    assert {record.meta["category"] for record in records} == {"narrated_action"}
    assert all(record.meta["decision_turn"] >= 7 for record in records)


def test_conversation_is_cut_after_the_latest_user_message() -> None:
    from rein.datagen.models import ConversationDraft

    draft = ConversationDraft.model_validate(
        {
            "conversation": [
                {"role": "user", "text": "hi"},
                {"role": "user", "text": "send a photo"},
                {"role": "assistant", "text": "*sends it*"},
            ],
            "draft": None,
        }
    )
    assert [message.text for message in draft.conversation] == ["hi", "send a photo"]


def test_tool_names_are_normalized_to_snake_case() -> None:
    from rein.datagen.models import _snake_case

    assert _snake_case("sendPhoto") == "send_photo"
    assert _snake_case("Check-Inventory!") == "check_inventory"
    assert _snake_case("3d_render") == "tool_3d_render"
