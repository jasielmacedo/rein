import time
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any

from rein.control.backend import DecisionBackend, rule_request
from rein.control.generator import Chunks, Generator, GeneratorRequest, chat_messages
from rein.control.induction import Opening, ToolCallSyntax, render_opening
from rein.control.plan import Directive, TurnPlan, plan_turn
from rein.control.sheet import ForceTool, Prefill, RuleSheet
from rein.control.visible import TOOL_CALL_MARKER, VisibleText
from rein.control.watcher import Check, Emit, WatchConfig, watch_stream
from rein.schema import Draft, State, ToolCall

WATCH_ACTIONS = (ForceTool, Prefill)


def _directive_log(directive: Directive) -> dict[str, Any]:
    return {
        "rule": directive.rule.id,
        "band": directive.band,
        "option": directive.option,
        "probability": round(directive.probability, 4),
    }


@dataclass
class TurnResult:
    reply: str
    visible: str
    tool_call: ToolCall | None
    escalate: bool
    plan: TurnPlan
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleSession:
    """One generator turn under a rule sheet: decide, induce or reinject, then watch the draft as it streams."""

    sheet: RuleSheet
    backend: DecisionBackend
    generator: Generator
    syntax: ToolCallSyntax
    system: str | None = None
    watch: WatchConfig = field(default_factory=WatchConfig)

    def _plan(self, state: State, scope: str) -> tuple[TurnPlan, float]:
        rules = self.sheet.scoped(scope)
        started = time.monotonic()
        if not rules:
            return TurnPlan(()), 0.0
        probabilities = self.backend.decide(rule_request(state, rules))
        return plan_turn(rules, probabilities), time.monotonic() - started

    def _watch_check(self, state: State, prefix: str) -> Check:
        def check(text: str) -> Directive | None:
            if TOOL_CALL_MARKER in text:
                return None
            plan, _ = self._plan(state.model_copy(update={"draft": Draft(text=prefix + text)}), "draft")
            opening = plan.opening
            return opening if opening is not None and isinstance(opening.rule.action, WATCH_ACTIONS) else None

        return check

    def _stream(
        self, messages: tuple[dict[str, Any], ...], prefix: str, tools_enabled: bool, banned: tuple[str, ...] = ()
    ) -> Chunks:
        return self.generator.stream(
            GeneratorRequest(messages, prefix, tools_enabled=tools_enabled, banned_terms=banned)
        )

    def _suppressed(self, plan: TurnPlan, opening: Opening) -> bool:
        """No tool this turn when every tool rule is confidently off: tools leave the prompt and stray calls are cut."""
        tools = plan.tool_probabilities()
        floor = self.sheet.tool_suppress_below
        return floor is not None and bool(tools) and opening.tool is None and max(tools.values()) < floor

    def _requested(self, plan: TurnPlan, fired: Directive) -> bool:
        """The watcher only turns narration into a call when the turn itself plausibly asked for that tool."""
        tools = plan.tool_probabilities()
        floor = self.sheet.watch_request_floor
        if floor is None or not tools or not isinstance(fired.rule.action, ForceTool):
            return True
        return tools.get(fired.rule.action.tool_for(fired.option), 0.0) >= floor

    def _banned(self, plan: TurnPlan) -> tuple[str, ...]:
        """Terms the generator may not write this turn: those of every rule decided into a tight or loose band."""
        active = (d.rule.forbidden_terms for d in plan.directives if d.band != "off")
        return tuple(dict.fromkeys(term for terms in active for term in terms))

    def _collect(self, stream: Chunks, visible: VisibleText) -> str:
        text = ""
        with closing(stream) as chunks:
            for chunk in chunks:
                text += chunk
                visible(chunk)
        return text

    def _without_action(
        self, state: State, reminders: list[str], tools_enabled: bool, banned: tuple[str, ...], visible: VisibleText
    ) -> str:
        notes = [*reminders, self.sheet.unrequested_action_reminder]
        retry = chat_messages(state, self.system, notes, self.sheet.reinject_placement)
        return self._collect(self._stream(retry, "", tools_enabled, banned), visible)

    def run_turn(self, state: State, emit: Emit = lambda text: None) -> TurnResult:
        plan, decide_seconds = self._plan(state, "turn")
        opening = render_opening(plan.opening, self.sheet.variables, self.syntax) if plan.opening else Opening("")
        reminders = [directive.rule.reminder for directive in plan.reinject]
        messages = chat_messages(state, self.system, reminders, self.sheet.reinject_placement)
        suppressed = self._suppressed(plan, opening)
        tools_enabled = not suppressed
        banned = self._banned(plan)
        visible = VisibleText(emit)
        visible(opening.text)

        watching = bool(self.sheet.scoped("draft")) and opening.tool is None and not plan.escalates
        reply = opening.text
        interventions: list[Directive] = []
        dropped: Directive | None = None
        regenerated = False
        checks = 0
        check_seconds: list[float] = []
        cut_at: int | None = None
        ended = _ends_reply(plan.opening)
        while not ended:
            stream = self._stream(messages, reply, tools_enabled, banned)
            if not (watching and len(interventions) < self.watch.max_interventions):
                reply += self._collect(stream, visible)
                break
            with closing(stream) as chunks:
                watched = watch_stream(chunks, self._watch_check(state, reply), self.watch, visible)
            checks += watched.checks
            check_seconds += watched.check_seconds
            reply += watched.text
            fired = watched.fired
            if fired is None:
                break
            cut_at = len(reply)
            if not self._requested(plan, fired):
                dropped = fired
                if not reply.strip():
                    regenerated = True
                    reply += self._without_action(state, reminders, tools_enabled, banned, visible)
                break
            interventions.append(fired)
            induced = render_opening(fired, self.sheet.variables, self.syntax)
            reply += induced.text
            visible(induced.text)
            if isinstance(fired.rule.action, ForceTool):
                reply += self._collect(self._stream(messages, reply, True, banned), visible)
                break
            ended = _ends_reply(fired)
        if suppressed and TOOL_CALL_MARKER in reply:
            reply = reply[: reply.index(TOOL_CALL_MARKER)]
        visible.flush()

        audit = {
            "decisions": [_directive_log(directive) for directive in plan.directives],
            "opening": plan.opening.rule.id if plan.opening else None,
            "reinjected": [directive.rule.id for directive in plan.reinject],
            "decide_seconds": round(decide_seconds, 4),
            "tools_suppressed": suppressed,
            "banned_terms": len(banned),
            "watch_checks": checks,
            "watch_seconds": [round(seconds, 4) for seconds in check_seconds],
            "intervention": _directive_log(interventions[0]) if interventions else None,
            "interventions": [_directive_log(directive) for directive in interventions],
            "unrequested_action_dropped": _directive_log(dropped) if dropped else None,
            "regenerated": regenerated,
            "cut_at": cut_at,
        }
        return TurnResult(
            reply=reply,
            visible=visible.text,
            tool_call=self.syntax.parse(reply),
            escalate=plan.escalates,
            plan=plan,
            audit=audit,
        )


def _ends_reply(directive: Directive | None) -> bool:
    """A prefill marked end_reply closes the reply: the generator is not allowed to continue past it."""
    return directive is not None and isinstance(directive.rule.action, Prefill) and directive.rule.action.end_reply
