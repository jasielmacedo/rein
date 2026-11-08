"""The Rein arm of the sheets' end-to-end test. The generator sees the persona alone; at every turn each rule of the
sheet goes to the decider as a question in Rein's own layout, and the app acts on the answers: the most probable tool's
call is opened when the tool rule leaves "none" under half the mass, tool calls are vetoed when it is sure there is
none and right after a tool result (one call a turn), and a rule's base joins the turn as a reminder when the rule
fires."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rein.compiler.constants import NULL_OPTIONS
from rein.control.induction import QwenXmlToolCall
from rein.control.plan import DecisionBackend, Probabilities
from rein.genview.render import GeneratorView
from rein.schema import DecisionRequest, Message, State
from rein.sheets.e2e import REIN_ARM, TOOL_ROLE, Session, render_prompt
from rein.sheets.model import Sheet, ToolAction

ACT_PROBABILITY = 0.5
VETO_PROBABILITY = 0.9
PLAYER_SCOPE = "player"
RECORD_KEY = "rein"
YES = "yes"


@dataclass(frozen=True)
class TurnActions:
    opening: str = ""
    reminders: tuple[str, ...] = ()
    veto_tools: bool = False
    decisions: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "opening": self.opening,
            "reminders": list(self.reminders),
            "veto_tools": self.veto_tools,
            "decisions": self.decisions,
        }


AFTER_TOOL_RESULT = TurnActions(veto_tools=True)


def decision_request(sheet: Sheet, history: Sequence[Message]) -> DecisionRequest:
    """Every rule that watches the player, asked on the conversation so far, with the persona as the context."""
    return DecisionRequest(
        state=State(context=sheet.context(), conversation=tuple(history)),
        questions={rule.id: rule.as_question() for rule in sheet.rules if rule.looks_at == PLAYER_SCOPE},
    )


def rule_fires(kind: str, answer: str) -> bool:
    """A yes/no rule fires on "yes", a choice rule on any option but the null ones; scales never act here."""
    if kind == "yes_no":
        return answer == YES
    return kind == "choice" and answer not in NULL_OPTIONS


def null_mass(probs: Mapping[str, float]) -> float:
    return sum(value for key, value in probs.items() if key in NULL_OPTIONS)


def tool_choice(probs: Mapping[str, float]) -> str | None:
    """The tool the arm opens on a tool rule's answers: the most probable one, as long as the null options hold under
    half the mass."""
    tools = {key: value for key, value in probs.items() if key not in NULL_OPTIONS}
    if not tools or null_mass(probs) >= ACT_PROBABILITY:
        return None
    return max(tools, key=lambda key: tools[key])


def plan_actions(sheet: Sheet, probabilities: Probabilities) -> TurnActions:
    opening = ""
    reminders: list[str] = []
    veto = False
    for rule in sheet.rules:
        probs = probabilities.get(rule.id)
        if not probs:
            continue
        best = max(probs, key=lambda key: probs[key])
        if isinstance(rule.action, ToolAction):
            chosen = tool_choice(probs)
            if chosen is not None:
                opening = QwenXmlToolCall().opening(chosen)
            elif null_mass(probs) >= VETO_PROBABILITY:
                veto = True
        elif rule_fires(rule.kind, best) and probs[best] >= ACT_PROBABILITY and rule.base:
            reminders.append(rule.base)
    decisions = {
        rule_id: {key: round(value, 4) for key, value in probs.items()} for rule_id, probs in probabilities.items()
    }
    return TurnActions(opening, tuple(reminders), veto, decisions)


class ReinSession(Session):
    """A session whose prompts carry the decider's actions; the decisions are kept with every turn."""

    def __init__(
        self, sheet: Sheet, view: GeneratorView, seed: int, decide: DecisionBackend, tool_call_token: int
    ) -> None:
        super().__init__(sheet, REIN_ARM, view, seed)
        self.decide = decide
        self.tool_call_token = tool_call_token
        self.pending = TurnActions()

    def prompt(self) -> list[int]:
        if self.history and self.history[-1].role == TOOL_ROLE:
            self.pending = AFTER_TOOL_RESULT
            return render_prompt(self.view, self.sheet, self.arm, self.history)
        self.pending = plan_actions(self.sheet, self.decide.decide(decision_request(self.sheet, self.history)))
        return render_prompt(
            self.view, self.sheet, self.arm, self.history, self.pending.reminders, self.pending.opening
        )

    def bans(self) -> tuple[int, ...]:
        return (self.tool_call_token,) if self.pending.veto_tools else ()

    def take_reply(self, index: int, raw: str) -> bool:
        called = super().take_reply(index, self.pending.opening + raw)
        self.turns[-1][RECORD_KEY] = self.pending.as_record()
        return called
