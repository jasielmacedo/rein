import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from rein.schema import constants as c
from rein.schema.conversation import State

QuestionType = Literal["noul", "choice", "score"]
QuestionId = Annotated[str, StringConstraints(pattern=c.ID_PATTERN)]
OptionKey = Annotated[str, StringConstraints(min_length=1, max_length=c.MAX_OPTION_KEY_CHARS)]
OptionText = Annotated[str, StringConstraints(max_length=c.MAX_OPTION_TEXT_CHARS)]


class Question(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: QuestionType
    instructions: Annotated[str, StringConstraints(min_length=1, max_length=c.MAX_INSTRUCTION_CHARS)]
    criteria: dict[OptionKey, OptionText | None] | list[OptionText] | None = None
    label: bool | int | str | None = None
    target: dict[OptionKey, float] | None = None

    @model_validator(mode="after")
    def _validate(self) -> "Question":
        _validate_criteria(self)
        if self.label is not None:
            self.label_index()
        if self.target is not None:
            _validate_target(self)
        return self

    def option_keys(self) -> tuple[str, ...]:
        if self.type == "noul":
            return c.NOUL_OPTION_KEYS
        if self.type == "choice":
            assert isinstance(self.criteria, dict)
            return tuple(self.criteria)
        assert isinstance(self.criteria, list)
        return tuple(str(level) for level in range(len(self.criteria)))

    def option_texts(self) -> tuple[str, ...]:
        if self.type == "noul":
            described = self.criteria if isinstance(self.criteria, dict) else {}
            return tuple(
                _join(key, described.get(flag)) for key, flag in zip(c.NOUL_OPTION_KEYS, ("false", "true"), strict=True)
            )
        if self.type == "choice":
            assert isinstance(self.criteria, dict)
            return tuple(_join(key, description) for key, description in self.criteria.items())
        assert isinstance(self.criteria, list)
        return tuple(f"{level}: {description}" for level, description in enumerate(self.criteria))

    def label_index(self) -> int | None:
        if self.label is None:
            return None
        keys = self.option_keys()
        if self.type == "noul":
            if not isinstance(self.label, bool):
                raise ValueError("noul label must be true or false")
            return int(self.label)
        if self.type == "choice":
            if not isinstance(self.label, str) or self.label not in keys:
                raise ValueError(f"choice label must be one of the options: {self.label!r}")
            return keys.index(self.label)
        if isinstance(self.label, bool) or not isinstance(self.label, int) or not 0 <= self.label < len(keys):
            raise ValueError(f"score label must be a level index in [0, {len(keys) - 1}]")
        return self.label

    def target_vector(self) -> tuple[float, ...] | None:
        if self.target is None:
            return None
        return tuple(self.target.get(key, 0.0) for key in self.option_keys())


def _join(key: str, description: str | None) -> str:
    return f"{key}: {description}" if description else key


def _validate_criteria(question: Question) -> None:
    criteria = question.criteria
    if question.type == "noul":
        if criteria is not None and (not isinstance(criteria, dict) or not set(criteria) <= c.NOUL_CRITERIA_KEYS):
            raise ValueError("noul criteria may only describe 'true' and 'false'")
    elif question.type == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= c.MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice needs 1 to {c.MAX_CHOICE_OPTIONS} named options")
    elif not isinstance(criteria, list) or not c.MIN_SCORE_LEVELS <= len(criteria) <= c.MAX_SCORE_LEVELS:
        raise ValueError(f"score needs {c.MIN_SCORE_LEVELS} to {c.MAX_SCORE_LEVELS} ordered levels")


def _validate_target(question: Question) -> None:
    assert question.target is not None
    keys = set(question.option_keys())
    if not set(question.target) <= keys:
        raise ValueError("target has keys that are not options")
    values = list(question.target.values())
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("target probabilities must be finite and non-negative")
    if abs(sum(values) - 1.0) > c.TARGET_SUM_TOLERANCE:
        raise ValueError("target probabilities must sum to 1")


class DecisionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    state: State
    questions: Annotated[dict[QuestionId, Question], Field(min_length=1, max_length=c.MAX_QUESTIONS)]


Split = Literal["train", "development", "calibration", "test", "gold"]


class TrainingRecord(DecisionRequest):
    id: Annotated[str, StringConstraints(min_length=1, max_length=c.MAX_RECORD_ID_CHARS)]
    source: Annotated[str, StringConstraints(min_length=1, max_length=c.MAX_SOURCE_CHARS)]
    scenario_id: Annotated[str, StringConstraints(min_length=1, max_length=c.MAX_SCENARIO_ID_CHARS)]
    split: Split
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_supervision(self) -> "TrainingRecord":
        missing = [
            qid for qid, question in self.questions.items() if question.label is None and question.target is None
        ]
        if missing:
            raise ValueError(f"questions without label or target: {missing}")
        return self
