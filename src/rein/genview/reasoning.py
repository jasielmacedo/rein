import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from rein.genview import constants as c
from rein.genview.build import THINKING, ViewRow
from rein.neurons.extract import load_views

REASONING = "reasoning"
ANSWER_START = "answer_start"
REASONING_SAMPLES = 2
MAX_REASONING_TOKENS = 768
MAX_MODEL_LEN = 16384
THINKING_TEMPERATURE = 0.6
THINKING_TOP_P = 0.95
THINKING_TOP_K = 20
SAMPLING_SEED = 0
ANSWER_GAP = "\n\n"
STOPPED = "stop"


def merge_reasoning(
    view: ViewRow, reasoning: Sequence[int], close_id: int, gap_ids: Sequence[int], samples: int = REASONING_SAMPLES
) -> ViewRow:
    """Appends the model's own reasoning to a thinking-mode view, with positions inside it and, when the reasoning
    closed, the position that opens the answer."""
    body = list(reasoning)
    finished = close_id in body
    if finished:
        body = body[: body.index(close_id) + 1]
    thought = len(body) - (1 if finished else 0)
    inside = range(view.prompt_length, view.prompt_length + thought)
    picked = tuple(sorted(random.Random(view.view_id).sample(inside, min(samples, len(inside)))))
    ids = (*view.ids, *body, *(gap_ids if finished else ()))
    positions = {**view.positions, REASONING: picked}
    if finished:
        positions[ANSWER_START] = (len(ids) - 1,)
    return replace(view, ids=ids, positions=positions)


def generate_reasoning(model: str, views: Sequence[ViewRow], max_tokens: int) -> list[list[int]]:
    """On-policy reasoning from Qwen3.5 itself with vLLM, sampled with Qwen's thinking settings."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=model, max_model_len=MAX_MODEL_LEN, limit_mm_per_prompt={"image": 0, "video": 0})
    tokenizer = llm.get_tokenizer()
    close_id = tokenizer.convert_tokens_to_ids(c.THINK_CLOSE)
    params = SamplingParams(
        temperature=THINKING_TEMPERATURE,
        top_p=THINKING_TOP_P,
        top_k=THINKING_TOP_K,
        max_tokens=max_tokens,
        stop_token_ids=[close_id],
        seed=SAMPLING_SEED,
    )
    prompts: list[Any] = [TokensPrompt(prompt_token_ids=list(view.ids)) for view in views]
    reasoning = []
    for output in llm.generate(prompts, params):
        ids = list(output.outputs[0].token_ids)
        if output.outputs[0].finish_reason == STOPPED and close_id not in ids:
            ids.append(close_id)
        reasoning.append(ids)
    return reasoning


def main() -> None:
    parser = argparse.ArgumentParser(description="Add Qwen3.5's own reasoning to the thinking-mode views.")
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=MAX_REASONING_TOKENS)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    close_id = int(tokenizer.get_added_vocab()[c.THINK_CLOSE])
    gap_ids = list(tokenizer(ANSWER_GAP, add_special_tokens=False)["input_ids"])
    views = load_views(args.views)
    thinking = [view for view in views if view.mode == THINKING]
    generated = dict(
        zip(
            (view.view_id for view in thinking),
            generate_reasoning(args.model, thinking, args.max_tokens),
            strict=True,
        )
    )
    finished = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for view in views:
            if view.view_id in generated:
                view = merge_reasoning(view, generated[view.view_id], close_id, gap_ids)
                finished += ANSWER_START in view.positions
            handle.write(view.to_json() + "\n")
    print(json.dumps({"thinking_views": len(thinking), "reasoning_closed": finished}), flush=True)


if __name__ == "__main__":
    main()
