import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path

from transformers import AutoTokenizer

from rein.genview import constants as c
from rein.genview.build import EVAL_FILES, NO_THINKING, THINKING
from rein.genview.render import GeneratorView, PromptSpec
from rein.neurons.runtimes import Prompt
from rein.schema import State, TrainingRecord
from rein.train.data import load_records

SAMPLE_SEED = 0


def plain_prompts(records: Sequence[TrainingRecord], view: GeneratorView) -> list[Prompt]:
    """Real conversations under a generic system prompt, in both thinking modes: nothing here asks for a rule."""
    prompts: list[Prompt] = []
    for record in records:
        spec = PromptSpec(system=c.GENERIC_SYSTEM, state=State(conversation=record.state.conversation))
        for mode in (NO_THINKING, THINKING):
            rendered = view.render(spec, thinking=mode == THINKING, key=record.id)
            prompts.append(Prompt(f"{record.id}|{mode}", rendered.ids))
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Token ids of plain chat prompts for the runtime checks.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=list(EVAL_FILES), default="development")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = [record for record in load_records(args.data / EVAL_FILES[args.split]) if record.state.draft is None]
    chosen = random.Random(SAMPLE_SEED).sample(records, min(args.count, len(records)))
    prompts = plain_prompts(chosen, GeneratorView(AutoTokenizer.from_pretrained(c.CHAT_TOKENIZER_REPO)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps({"prompt_id": prompt.prompt_id, "ids": list(prompt.ids)}) + "\n" for prompt in prompts)
    args.out.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"prompts": len(prompts), "longest": max(len(prompt.ids) for prompt in prompts)}))


if __name__ == "__main__":
    main()
