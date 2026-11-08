import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from rein.datagen.audit import Auditor, QuestionAudit, apply_audit, audited_record, judge, reference_option
from rein.datagen.pipeline import Stage
from rein.datagen.prompts import audit_prompt
from rein.schema import Message, Question, State, TrainingRecord
from rein.teacher.client import Completion

NOUL = Question(type="noul", instructions="Did the user ask for a photo?", label=True)
CHOICE = Question(
    type="choice",
    instructions="Which tool?",
    criteria={"send_photo": None, "web_search": None, "none": None},
    target={"send_photo": 0.6, "none": 0.4},
)


def _record() -> TrainingRecord:
    return TrainingRecord(
        id="r1",
        source="synthetic",
        scenario_id="s1",
        split="test",
        state=State(conversation=(Message(role="user", text="send me a photo"),)),
        questions={"photo": NOUL, "tool": CHOICE},
    )


class ScriptedModel:
    def __init__(self, replies: Sequence[Mapping[str, str]]) -> None:
        self.replies = list(replies)
        self.calls = 0

    async def chat(self, messages: Sequence[Mapping[str, str]], **kwargs: object) -> Completion:
        reply = self.replies[self.calls]
        self.calls += 1
        return Completion(json.dumps({"answers": reply}), 0, 10, "stop", 0.0)


def test_reference_option_uses_label_or_target_peak() -> None:
    assert reference_option(NOUL) == "yes"
    assert reference_option(CHOICE) == "send_photo"


SCORE = Question(type="score", instructions="How far did the tone drift?", criteria=["none", "a bit", "a lot"], label=0)


def test_agreeing_thinking_votes_overturn_a_label() -> None:
    assert judge(NOUL, "yes", "yes", ()).status == "confirmed"
    corrected = judge(NOUL, "yes", "no", ("no", "no"))
    assert corrected.status == "corrected" and corrected.corrected == "no"
    assert judge(NOUL, "yes", "no", ("no", "yes")).status == "disputed"
    assert judge(NOUL, "yes", "no", ("yes", "yes")).status == "confirmed"
    assert judge(NOUL, "yes", None, (None, None)).status == "abstained"
    assert judge(NOUL, "yes", "no", ("no", None)).status == "disputed"


def test_score_corrections_need_a_two_level_gap() -> None:
    assert judge(SCORE, "0", "1", ("1", "1")).status == "disputed"
    far = judge(SCORE, "0", "2", ("2", "2"))
    assert far.status == "corrected" and far.corrected == "2"


def test_corrections_and_disputes_change_the_question() -> None:
    corrected = apply_audit(NOUL, QuestionAudit("corrected", "yes", ("no", "no", "no"), "no"))
    assert corrected.label is False and corrected.target is None
    disputed = apply_audit(CHOICE, QuestionAudit("disputed", "send_photo", ("none", "none", "web_search")))
    assert disputed.target is not None
    assert abs(sum(disputed.target.values()) - 1) < 1e-9
    assert disputed.target["send_photo"] > disputed.target["none"]


def test_auditor_thinks_only_when_the_screen_disagrees(tmp_path: Path) -> None:
    screen = {"photo": "no", "tool": "send_photo"}
    thinking = {"photo": "no", "tool": "none"}
    model = ScriptedModel([screen, thinking, thinking])
    auditor = Auditor(model, Stage(tmp_path / "votes.jsonl"))
    audits = asyncio.run(auditor.audit(_record()))
    assert model.calls == 3
    assert audits["photo"].status == "corrected"
    assert audits["tool"].status == "confirmed" and audits["tool"].votes == ("send_photo",)
    record = audited_record(_record(), audits)
    assert record.questions["photo"].label is False
    assert record.meta["audit"]["questions"]["photo"]["votes"] == ["no", "no", "no"]

    again = Auditor(ScriptedModel([]), Stage(tmp_path / "votes.jsonl"))
    assert asyncio.run(again.audit(_record()))["photo"].status == "corrected"

    agreeing = ScriptedModel([{"photo": "yes", "tool": "send_photo"}])
    asyncio.run(Auditor(agreeing, Stage(tmp_path / "other.jsonl")).audit(_record()))
    assert agreeing.calls == 1


def test_auditor_ignores_answers_outside_the_options(tmp_path: Path) -> None:
    model = ScriptedModel([{"photo": "maybe", "tool": "send_photo"}, {"tool": "send_photo"}, {"tool": "send_photo"}])
    audits = asyncio.run(Auditor(model, Stage(tmp_path / "votes.jsonl")).audit(_record()))
    assert audits["photo"].status == "abstained" and model.calls == 3


def test_audit_prompt_never_reveals_the_current_label() -> None:
    flipped = _record().model_copy(
        update={
            "questions": {
                "photo": NOUL.model_copy(update={"label": False}),
                "tool": CHOICE.model_copy(update={"target": {"web_search": 1.0}}),
            }
        }
    )
    prompt = audit_prompt(_record())
    assert prompt == audit_prompt(flipped)
    assert "label" not in prompt.lower() and "target" not in prompt.lower()
