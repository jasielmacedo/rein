from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rein.schema import QuestionType

TAXONOMY_FILE = Path(__file__).with_name("taxonomy.yaml")
WEIGHT_TOLERANCE = 1e-6


class Category(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    types: tuple[QuestionType, ...]
    description: str
    needs_draft: bool = False
    needs_assistant_turns: bool = False


class CaseForm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fires: bool
    guidance: str
    two_rules: bool = False


class LengthBand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min: int = Field(ge=1)
    max: int = Field(ge=1)
    weight: float = Field(gt=0)


class Holdout(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    domains: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    scenario_fraction: float = Field(default=0.0, ge=0, lt=1)


class Languages(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation: dict[str, float]
    rules: dict[str, float]
    names: dict[str, str]

    @model_validator(mode="after")
    def _check(self) -> "Languages":
        for name, weights in (("conversation", self.conversation), ("rules", self.rules)):
            if abs(sum(weights.values()) - 1.0) > WEIGHT_TOLERANCE:
                raise ValueError(f"{name} language weights must sum to 1")
            missing = set(weights) - set(self.names)
            if missing:
                raise ValueError(f"languages without a display name: {sorted(missing)}")
        return self


class Taxonomy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    categories: dict[str, Category]
    domains: tuple[str, ...] = Field(min_length=1)
    case_forms: dict[str, CaseForm]
    rule_styles: tuple[str, ...] = Field(min_length=1)
    conversation_lengths: dict[str, LengthBand]
    holdout: Holdout = Holdout()
    languages: Languages

    @model_validator(mode="after")
    def _holdout_exists(self) -> "Taxonomy":
        if not set(self.holdout.domains) <= set(self.domains):
            raise ValueError("holdout domains must be listed domains")
        if not set(self.holdout.categories) <= set(self.categories):
            raise ValueError("holdout categories must be listed categories")
        return self


@cache
def load_taxonomy(path: Path = TAXONOMY_FILE) -> Taxonomy:
    return Taxonomy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
