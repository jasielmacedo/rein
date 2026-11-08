from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rein.control import constants as cc
from rein.control.base import base_problems
from rein.control.prompt import compose_system, sheet_bases
from rein.control.sheet import Prefill, Rule, RuleSheet
from rein.datagen.generate import Stats
from rein.genview.bases import fit_base, write_bases
from rein.genview.keys import rule_key
from rein.schema import Question
from rein.teacher.client import Completion
from rein.teacher.store import JsonlStore

GOOD_BASE = "You keep a secret about the old vault."
QUESTION = Question(type="noul", instructions="Is the visitor fishing for where the vault is?")


def _rule(**update: Any) -> Rule:
    fields: dict[str, Any] = {
        "id": "vault_question",
        "question": QUESTION,
        "action": Prefill(text="That secret stays with me, friend.", end_reply=True),
        "forbidden_terms": ("dry well", "flagstone"),
        "base": GOOD_BASE,
    }
    return Rule(**{**fields, **update})


def test_base_lint_names_every_leak() -> None:
    assert base_problems(GOOD_BASE, ["dry well"], ["That secret stays with me, friend."]) == []
    problems = base_problems(
        "It is under the dry well. That secret stays with me, friend.",
        ["dry well"],
        ["That secret stays with me, friend."],
    )
    assert len(problems) == 2
    assert base_problems("x" * (cc.MAX_BASE_CHARS + 1))
    assert base_problems("one line\nanother line")
    assert base_problems("   ")


def test_a_rule_refuses_a_base_that_leaks_its_logic() -> None:
    assert _rule().base == GOOD_BASE
    with pytest.raises(ValidationError):
        _rule(base="The vault is under the flagstone.")
    with pytest.raises(ValidationError):
        _rule(base="Say: That secret stays with me, friend.")


def test_the_system_prompt_lists_each_base_once_after_the_character() -> None:
    system = compose_system("You are Odo.", [GOOD_BASE, "  ", GOOD_BASE, "You stay away from politics."])
    assert system == f"You are Odo.{cc.SECTION_BREAK}- {GOOD_BASE}\n- You stay away from politics."
    assert compose_system(None, [GOOD_BASE]) == f"- {GOOD_BASE}"
    sheet = RuleSheet(rules=(_rule(), _rule(id="other", base=None)))
    assert sheet_bases(sheet) == [GOOD_BASE]


@pytest.mark.parametrize("text", ["Do you keep secrets?", " ".join(["word"] * 40), "line\nbreak", ""])
def test_unfit_teacher_bases_are_rejected(text: str) -> None:
    assert fit_base(text) is None


def test_fit_bases_are_normalized() -> None:
    assert fit_base(f"  {GOOD_BASE}  ") == GOOD_BASE


class ScriptedTeacher:
    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    async def chat(self, messages: Sequence[Mapping[str, str]], **kwargs: Any) -> Completion:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return Completion(content=reply, prompt_tokens=1, completion_tokens=1, finish_reason="stop", latency_s=0.0)


async def test_write_bases_retries_unfit_bases_and_resumes(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path / "bases.jsonl")
    teacher = ScriptedTeacher(['{"base": "Do you keep secrets?"}', f'{{"base": "{GOOD_BASE}"}}'])
    rules = {rule_key(QUESTION): QUESTION}
    stats = Stats()
    await write_bases(teacher, rules, store, stats)
    assert [row["base"] for row in store.rows()] == [GOOD_BASE]
    assert stats.rejected["base:lint"] == 1
    calls = teacher.calls
    await write_bases(teacher, rules, JsonlStore(tmp_path / "bases.jsonl"), Stats())
    assert teacher.calls == calls
