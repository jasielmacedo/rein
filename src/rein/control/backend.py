from collections.abc import Sequence
from dataclasses import dataclass

from rein.control.plan import DecisionBackend, Probabilities  # noqa: F401  (the protocol stays importable here)
from rein.control.sheet import Rule
from rein.model.decider import Decider
from rein.schema import DecisionRequest, State


@dataclass
class DeciderBackend:
    decider: Decider

    def decide(self, request: DecisionRequest) -> Probabilities:
        return self.decider.decide(request).by_key()


def rule_request(state: State, rules: Sequence[Rule]) -> DecisionRequest:
    return DecisionRequest(state=state, questions={rule.id: rule.question for rule in rules})
