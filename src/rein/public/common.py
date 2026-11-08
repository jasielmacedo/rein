import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from rein.schema import Message, Question, State, TrainingRecord

PUBLIC_SPLIT = "train"
MAX_PUBLIC_OPTIONS = 24
NONE_OPTION = "none"
NONE_RATE = 0.1
INTENT_INSTRUCTIONS = (
    "Which intent best describes the latest user message?",
    "What is the user asking for in the latest message?",
    "Classify the latest user message into one of the listed intents.",
    "Qual intenção descreve melhor a última mensagem do usuário?",
    "O que o usuário está pedindo na última mensagem?",
)


def render_value(value: Any, indent: int = 0) -> str:
    """Kev-style rendering of structured states: `key: value` lines and `- ` bullets."""
    pad = "  " * indent
    if isinstance(value, Mapping):
        lines = []
        for key, item in value.items():
            if isinstance(item, Mapping | list):
                lines.append(f"{pad}{key}:")
                lines.append(render_value(item, indent + 1))
            else:
                lines.append(f"{pad}{key}: {item}")
        return "\n".join(lines)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {render_value(item, indent + 1).strip()}" for item in value)
    return f"{pad}{value}"


def single_message_state(text: str) -> State:
    return State(conversation=(Message(role="user", text=text),))


def intent_question(
    gold: str | None, intents: list[str], rng: random.Random, instructions: str | None = None
) -> Question:
    """A choice over a random subset of intents that always holds the answer, or `none` when it is left out."""
    others = [intent for intent in intents if intent != gold]
    shown = rng.sample(others, min(len(others), rng.randint(3, MAX_PUBLIC_OPTIONS - 2)))
    label = NONE_OPTION
    if gold is not None and rng.random() >= NONE_RATE:
        shown.append(gold)
        label = gold
    rng.shuffle(shown)
    criteria: dict[str, str | None] = {intent: None for intent in shown}
    criteria[NONE_OPTION] = "none of the listed intents"
    return Question(
        type="choice",
        instructions=instructions or rng.choice(INTENT_INSTRUCTIONS),
        criteria=criteria,
        label=label,
    )


def class_names_from_card(readme: str, feature: str = "label") -> list[str]:
    """Class label names from a Hugging Face dataset card's YAML front matter."""
    front = yaml.safe_load(readme.split("---")[1])
    infos = front["dataset_info"]
    for info in infos if isinstance(infos, list) else [infos]:
        for item in info["features"]:
            if item["name"] == feature:
                names = item["dtype"]["class_label"]["names"]
                return [names[key] for key in sorted(names, key=int)]
    raise ValueError(f"no class label named {feature!r} in the card")


def write_records(records: Iterable[TrainingRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
            count += 1
    return count


def try_record(rejected: Counter[str], reason: str, **fields: Any) -> TrainingRecord | None:
    try:
        return TrainingRecord(**fields)
    except ValidationError:
        rejected[reason] += 1
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
