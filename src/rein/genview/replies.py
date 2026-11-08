"""Qwen3.5's own reply appended to each non-thinking view, with positions inside it.

Rules whose action lasts the whole reply (steer) must keep firing there, and rules that act once at the reply opening
(forcing a tool call) must stay silent there, so the per-sheet fit needs states from inside on-policy replies.
"""

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from rein.genview.build import NO_THINKING, ViewRow
from rein.neurons.extract import load_views

REPLY = "reply"
REPLY_SAMPLES = 6
MAX_REPLY_TOKENS = 48
MAX_MODEL_LEN = 32768
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
SEED = 0


def merge_reply(view: ViewRow, reply: Sequence[int], samples: int = REPLY_SAMPLES) -> ViewRow:
    """The view continued by the reply, with up to `samples` positions whose next token is part of the reply."""
    ids = (*view.ids, *reply)
    inside = range(view.prompt_length, len(ids) - 1)
    picked = tuple(sorted(random.Random(view.view_id).sample(inside, min(samples, len(inside)))))
    return replace(view, ids=ids, positions={**view.positions, REPLY: picked})


def sample_replies(model: str, views: Sequence[ViewRow], max_tokens: int, samples: int) -> list[list[list[int]]]:
    """`samples` replies per view with Qwen3.5's non-thinking sampling."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=model, max_model_len=MAX_MODEL_LEN, limit_mm_per_prompt={"image": 0, "video": 0}, seed=SEED)
    params = SamplingParams(
        n=samples, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=max_tokens, seed=SEED
    )
    prompts: list[Any] = [TokensPrompt(prompt_token_ids=list(view.ids)) for view in views]
    return [[list(choice.token_ids) for choice in output.outputs] for output in llm.generate(prompts, params)]


def generate_replies(model: str, views: Sequence[ViewRow], max_tokens: int) -> list[list[int]]:
    return [replies[0] for replies in sample_replies(model, views, max_tokens, 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Add Qwen3.5's own reply to the non-thinking views.")
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=MAX_REPLY_TOKENS)
    args = parser.parse_args()
    views = load_views(args.views)
    opening = [view for view in views if view.mode == NO_THINKING]
    replies = dict(
        zip((view.view_id for view in opening), generate_replies(args.model, opening, args.max_tokens), strict=True)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for view in views:
            if view.view_id in replies:
                view = merge_reply(view, replies[view.view_id])
            handle.write(view.to_json() + "\n")
    print(json.dumps({"views": len(views), "with_reply": len(replies)}), flush=True)


if __name__ == "__main__":
    main()
