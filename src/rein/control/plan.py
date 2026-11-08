from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from rein.control.sheet import Escalate, ForceTool, Prefill, Rule
from rein.schema import DecisionRequest

Band = Literal["tight", "loose", "off"]
Probabilities = Mapping[str, Mapping[str, float]]
OPENING_ACTIONS = (ForceTool, Prefill, Escalate)


class DecisionBackend(Protocol):
    def decide(self, request: DecisionRequest) -> Probabilities: ...


@dataclass(frozen=True)
class Directive:
    rule: Rule
    band: Band
    probability: float
    option: str


@dataclass(frozen=True)
class TurnPlan:
    directives: tuple[Directive, ...]
    opening: Directive | None = None
    reinject: tuple[Directive, ...] = field(default_factory=tuple)
    probabilities: Probabilities = field(default_factory=dict)

    def tool_probabilities(self) -> dict[str, float]:
        """How likely each tool is to be needed this turn, from every tool rule's firing options."""
        tools: dict[str, float] = {}
        for directive in self.directives:
            action = directive.rule.action
            if not isinstance(action, ForceTool):
                continue
            for option in directive.rule.fire_on:
                tool = action.tool_for(option)
                tools[tool] = max(tools.get(tool, 0.0), self.probabilities[directive.rule.id].get(option, 0.0))
        return tools

    @property
    def escalates(self) -> bool:
        return self.opening is not None and isinstance(self.opening.rule.action, Escalate)


def directive(rule: Rule, probabilities: Probabilities) -> Directive:
    """The rule fires on its most likely firing option, with that option's probability."""
    options = probabilities.get(rule.id)
    if options is None:
        raise KeyError(f"no decision for rule {rule.id}")
    option = max(rule.fire_on, key=lambda key: options.get(key, 0.0))
    probability = options.get(option, 0.0)
    if probability >= rule.gate.fire:
        band: Band = "tight"
    elif rule.gate.reinject is not None and probability >= rule.gate.reinject:
        band = "loose"
    else:
        band = "off"
    return Directive(rule, band, probability, option)


def plan_turn(rules: Sequence[Rule], probabilities: Probabilities) -> TurnPlan:
    """One action opens the reply; every other active rule is reinjected so no decision is dropped."""
    directives = tuple(directive(rule, probabilities) for rule in rules)
    openers = [d for d in directives if d.band == "tight" and isinstance(d.rule.action, OPENING_ACTIONS)]
    opening = max(openers, key=lambda d: (d.rule.priority, d.probability), default=None)
    reinject = tuple(d for d in directives if d is not opening and d.band != "off")
    return TurnPlan(directives, opening, reinject, probabilities)
