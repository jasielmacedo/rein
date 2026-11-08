"""Greedy generation from token ids in each runtime, for the stage 1A checks.

Every runtime lives in its own image, so each one is imported only inside its own function.
"""

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_MODEL_LEN = 8192
NO_IMAGES = {"image": 0, "video": 0}
HTTP_TIMEOUT_S = 600.0
MS_PER_S = 1000.0
RUNTIMES = ("transformers", "vllm", "sglang", "mlx", "llamacpp", "ollama")
OLLAMA_MODEL = "rein"
NS_PER_S = 1e9


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    ids: tuple[int, ...]


@dataclass(frozen=True)
class Generation:
    prompt_id: str
    ids: tuple[int, ...]
    text: str


@dataclass(frozen=True)
class Run:
    """Generations and the wall time of the generation calls alone, without model loading."""

    generations: list[Generation]
    seconds: float

    @property
    def tokens_per_s(self) -> float:
        return sum(len(generation.ids) for generation in self.generations) / max(self.seconds, 1e-9)


def load_prompts(path: Path) -> list[Prompt]:
    with path.open(encoding="utf-8") as handle:
        return [Prompt(row["prompt_id"], tuple(row["ids"])) for row in map(json.loads, handle) if row]


def transformers_generate(model_dir: str, prompts: Sequence[Prompt], max_tokens: int) -> Run:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, device_map=device).eval()
    generations = []
    started = time.perf_counter()
    for prompt in prompts:
        tokens = torch.tensor([list(prompt.ids)], device=device)
        output = model.generate(tokens, max_new_tokens=max_tokens, do_sample=False)[0, tokens.shape[1] :].tolist()
        text = str(tokenizer.decode(output, skip_special_tokens=True))
        generations.append(Generation(prompt.prompt_id, tuple(output), text))
    return Run(generations, time.perf_counter() - started)


def vllm_generate(model_dir: str, prompts: Sequence[Prompt], max_tokens: int) -> Run:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=model_dir, max_model_len=MAX_MODEL_LEN, limit_mm_per_prompt=NO_IMAGES, seed=0)
    inputs: list[Any] = [TokensPrompt(prompt_token_ids=list(prompt.ids)) for prompt in prompts]
    started = time.perf_counter()
    outputs = llm.generate(inputs, SamplingParams(temperature=0.0, max_tokens=max_tokens))
    seconds = time.perf_counter() - started
    generations = [
        Generation(prompt.prompt_id, tuple(output.outputs[0].token_ids), output.outputs[0].text)
        for prompt, output in zip(prompts, outputs, strict=True)
    ]
    return Run(generations, seconds)


def sglang_generate(model_dir: str, prompts: Sequence[Prompt], max_tokens: int) -> Run:
    import sglang
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    engine = sglang.Engine(model_path=model_dir, context_length=MAX_MODEL_LEN, random_seed=0)
    try:
        started = time.perf_counter()
        outputs = engine.generate(
            input_ids=[list(prompt.ids) for prompt in prompts],
            sampling_params={"temperature": 0.0, "max_new_tokens": max_tokens},
        )
        seconds = time.perf_counter() - started
    finally:
        engine.shutdown()
    generations = []
    for prompt, output in zip(prompts, outputs, strict=True):
        ids = output.get("output_ids") or tokenizer(output["text"], add_special_tokens=False)["input_ids"]
        generations.append(Generation(prompt.prompt_id, tuple(ids), output["text"]))
    return Run(generations, seconds)


def mlx_generate(model_dir: str, prompts: Sequence[Prompt], max_tokens: int) -> Run:
    from mlx_lm import load, stream_generate

    model, tokenizer = load(model_dir)
    generations = []
    started = time.perf_counter()
    for prompt in prompts:
        ids: list[int] = []
        text = ""
        for response in stream_generate(model, tokenizer, prompt=list(prompt.ids), max_tokens=max_tokens):
            ids.append(int(response.token))
            text += response.text
        generations.append(Generation(prompt.prompt_id, tuple(ids), text))
    return Run(generations, time.perf_counter() - started)


def llamacpp_generate(server: str, prompts: Sequence[Prompt], max_tokens: int) -> Run:
    """Generations from llama-server, timed by the server's own decode timings."""
    import httpx

    generations, seconds = [], 0.0
    with httpx.Client(timeout=HTTP_TIMEOUT_S) as client:
        for prompt in prompts:
            body = {
                "prompt": list(prompt.ids),
                "n_predict": max_tokens,
                "temperature": 0.0,
                "top_k": 1,
                "cache_prompt": False,
                "return_tokens": True,
            }
            reply = client.post(f"{server.rstrip('/')}/completion", json=body)
            reply.raise_for_status()
            payload = reply.json()
            generations.append(Generation(prompt.prompt_id, tuple(payload.get("tokens", [])), payload["content"]))
            seconds += float(payload.get("timings", {}).get("predicted_ms", 0.0)) / MS_PER_S
    return Run(generations, seconds)


def ollama_generate(server: str, prompts: Sequence[Prompt], max_tokens: int) -> Run:
    """Ollama's raw generation of each prompt's exact text (its model passes prompts through untouched), the reply
    tokenized back with Qwen's tokenizer; timed by Ollama's own eval duration."""
    import httpx
    from transformers import AutoTokenizer

    from rein.genview.constants import CHAT_TOKENIZER_REPO

    tokenizer = AutoTokenizer.from_pretrained(CHAT_TOKENIZER_REPO)
    generations, seconds = [], 0.0
    with httpx.Client(timeout=HTTP_TIMEOUT_S) as client:
        for prompt in prompts:
            body = {
                "model": OLLAMA_MODEL,
                "prompt": tokenizer.decode(list(prompt.ids), skip_special_tokens=False),
                "raw": True,
                "stream": False,
                "options": {"temperature": 0.0, "top_k": 1, "seed": 0, "num_predict": max_tokens},
            }
            reply = client.post(f"{server.rstrip('/')}/api/generate", json=body)
            reply.raise_for_status()
            payload = reply.json()
            text = str(payload["response"])
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            generations.append(Generation(prompt.prompt_id, tuple(ids), text))
            seconds += float(payload.get("eval_duration", 0)) / NS_PER_S
    return Run(generations, seconds)


def compare(reference: Sequence[Generation], other: Sequence[Generation]) -> dict[str, float]:
    """How closely two runs of the same prompts agree, token by token."""
    pairs = list(zip(reference, other, strict=True))
    identical = sum(first.ids == second.ids for first, second in pairs)
    first_token = sum(bool(first.ids) and first.ids[:1] == second.ids[:1] for first, second in pairs)
    prefixes = []
    for first, second in pairs:
        shared = 0
        for a, b in zip(first.ids, second.ids, strict=False):
            if a != b:
                break
            shared += 1
        prefixes.append(shared)
    return {
        "prompts": float(len(pairs)),
        "identical": identical / max(1, len(pairs)),
        "first_token_agreement": first_token / max(1, len(pairs)),
        "mean_shared_prefix": sum(prefixes) / max(1, len(prefixes)),
    }


def forced_rate(generations: Sequence[Generation], token_id: int, token_text: str) -> dict[str, float]:
    """Share of replies that open with the forced token, by id and as visible text."""
    count = max(1, len(generations))
    return {
        "first_id": sum(generation.ids[:1] == (token_id,) for generation in generations) / count,
        "visible_text": sum(generation.text.lstrip().startswith(token_text) for generation in generations) / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy generations of one checkpoint in one runtime.")
    parser.add_argument("--runtime", choices=RUNTIMES, required=True)
    parser.add_argument("--model", required=True, help="checkpoint directory, or the llama-server URL")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    args = parser.parse_args()
    prompts = load_prompts(args.prompts)
    generate = {
        "transformers": transformers_generate,
        "vllm": vllm_generate,
        "sglang": sglang_generate,
        "mlx": mlx_generate,
        "llamacpp": llamacpp_generate,
        "ollama": ollama_generate,
    }[args.runtime]
    run = generate(args.model, prompts, args.max_tokens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runtime": args.runtime,
        "model": args.model,
        "seconds": run.seconds,
        "tokens_per_s": run.tokens_per_s,
        "generations": [asdict(generation) for generation in run.generations],
    }
    args.out.write_text(json.dumps(payload), encoding="utf-8")


def load_run(path: Path) -> Run:
    payload = json.loads(path.read_text(encoding="utf-8"))
    generations = [Generation(item["prompt_id"], tuple(item["ids"]), item["text"]) for item in payload["generations"]]
    return Run(generations, float(payload["seconds"]))


if __name__ == "__main__":
    main()
