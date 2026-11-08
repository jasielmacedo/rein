"""GPU steps of the behavior vectors: Qwen3.5-4B's hold and give-in replies (vLLM), the vector itself (Transformers),
and the strength sweep, where the vector is added after the steering block from the reply opening on, as the compiled
steer neurons will add it."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from rein.compiler.oracle_probe import is_held_out

HOLD, GIVE_IN = "hold", "give_in"
POOLED = "pooled"
SIDES = (HOLD, GIVE_IN)
REPLY_TOKENS = 64
TEMPERATURE = 0.7
TOP_P = 0.8
STEER_BLOCK_OUTPUT = 21
STEER_LAYER = STEER_BLOCK_OUTPUT - 1
SWEEP_TOKENS = 64
VECTORS_FILE = "vectors.safetensors"
DURATION_FILE = "duration.jsonl"
CONDITION_SEPARATOR = ":"
ALL_STEPS = "all"
TOOL_CALL = "<tool_call>"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _model(name: str) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()


def held_out(row: dict[str, Any]) -> bool:
    """Rows say whether they are held out (an evaluation split), or fall back on the hashed share of their record."""
    return bool(row["held_out"]) if "held_out" in row else is_held_out(row["record_id"])


def vectors(model: Any, rows: Sequence[dict[str, Any]]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """One vector per name, from the usable pairs outside the held-out share, and one over every name's pairs."""
    tensors: dict[str, torch.Tensor] = {}
    stats: dict[str, Any] = {}
    pooled: tuple[list[torch.Tensor], list[torch.Tensor]] = ([], [])
    for name in sorted({row["vector"] for row in rows}):
        mine = [row for row in rows if row["vector"] == name and row.get("usable") and not held_out(row)]
        if not mine:
            continue
        holds = [reply_means(model, row["plain"], [row[f"{HOLD}_reply"]], STEER_BLOCK_OUTPUT)[0] for row in mine]
        gives = [reply_means(model, row["plain"], [row[f"{GIVE_IN}_reply"]], STEER_BLOCK_OUTPUT)[0] for row in mine]
        tensors[name], stats[name] = _vector(holds, gives)
        pooled[0].extend(holds)
        pooled[1].extend(gives)
    if len(tensors) > 1:
        tensors[POOLED], stats[POOLED] = _vector(*pooled)
    return tensors, stats


def _vector(holds: list[torch.Tensor], gives: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
    vector = steer_vector(holds, gives)
    stats = {
        "pairs": len(holds),
        "norm": float(vector.norm()),
        "residual_norm": float(torch.stack(holds + gives).norm(dim=-1).median()),
    }
    return vector.contiguous(), stats


def sample_pairs(model: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Qwen3.5-4B continues each prefilled opening (holding to the rule, or giving in) with its non-thinking settings;
    the reply is the opening plus Qwen's own continuation."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=model, max_model_len=16384, limit_mm_per_prompt={"image": 0, "video": 0}, seed=0)
    params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=REPLY_TOKENS, seed=0)
    prompts: list[Any] = [
        TokensPrompt(prompt_token_ids=[*row["plain"], *row[f"{side}_prefix"]]) for row in rows for side in SIDES
    ]
    outputs = iter(llm.generate(prompts, params))
    tokenizer = llm.get_tokenizer()
    for row in rows:
        for side in SIDES:
            ids = [*row[f"{side}_prefix"], *next(outputs).outputs[0].token_ids]
            row[f"{side}_reply"] = ids
            row[f"{side}_text"] = tokenizer.decode(ids, skip_special_tokens=True)
    return rows


@torch.no_grad()
def reply_means(
    model: Any, prompt: Sequence[int], replies: Sequence[Sequence[int]], output_index: int
) -> list[torch.Tensor]:
    """Mean residual after the steering block over each reply's tokens, every reply read after the same prompt."""
    device = next(model.parameters()).device
    means = []
    for reply in replies:
        ids = torch.tensor([[*prompt, *reply]], device=device)
        hidden = model(input_ids=ids, output_hidden_states=True).hidden_states[output_index][0]
        means.append(hidden[len(prompt) : len(prompt) + len(reply)].float().mean(0).cpu())
    return means


def steer_vector(holds: Sequence[torch.Tensor], give_ins: Sequence[torch.Tensor]) -> torch.Tensor:
    """Mean of the paired differences: what holding adds over giving in, at the steering block."""
    return torch.stack([hold - give for hold, give in zip(holds, give_ins, strict=True)]).mean(0)


class SteerHook:
    """Adds `alpha * vector` to the output of one block at the last prompt position and at the generated tokens after
    it: every one, or only the first `steps` positions (1 is the reply opening alone)."""

    def __init__(self, vector: torch.Tensor, alpha: float, steps: int | None = None) -> None:
        self.vector = vector
        self.alpha = alpha
        self.steps = steps
        self.calls = 0

    def __call__(self, module: Any, args: Any, output: Any) -> Any:
        if self.steps is not None and self.calls >= self.steps:
            return output
        self.calls += 1
        hidden = output[0] if isinstance(output, tuple) else output
        added = (self.alpha * self.vector).to(hidden.dtype)
        hidden[:, -1, :] = hidden[:, -1, :] + added
        return output


@torch.no_grad()
def sweep(
    model: Any, rows: Sequence[dict[str, Any]], tensors: dict[str, torch.Tensor], alphas: Sequence[float]
) -> list[dict[str, Any]]:
    """Greedy replies to the held-out attacks under each strength, decoded one token at a time with the hook on."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    layer = model.model.layers[STEER_LAYER]
    held = [row for row in rows if held_out(row) and row["vector"] in tensors]
    out = []
    for alpha in alphas:
        for row in held:
            handle = layer.register_forward_hook(SteerHook(tensors[row["vector"]].to(model.device), alpha))
            try:
                ids = torch.tensor([row["plain"]], device=model.device)
                reply = model.generate(ids, max_new_tokens=SWEEP_TOKENS, do_sample=False)[0, ids.shape[1] :]
            finally:
                handle.remove()
            out.append(
                {
                    "record_id": row["record_id"],
                    "rule_id": row["rule_id"],
                    "vector": row["vector"],
                    "alpha": alpha,
                    "text": tokenizer.decode(reply, skip_special_tokens=True),
                }
            )
        print(json.dumps({"alpha": alpha, "replies": len(held)}), flush=True)
    return out


def condition(spec: str) -> tuple[float, int | None]:
    """`alpha:steps`, where steps is how many positions from the reply opening get the vector, or ALL_STEPS."""
    alpha, _, steps = spec.partition(CONDITION_SEPARATOR)
    return float(alpha), None if steps == ALL_STEPS else int(steps)


@torch.no_grad()
def duration_sweep(
    model: Any, rows: Sequence[dict[str, Any]], tensors: dict[str, torch.Tensor], conditions: Sequence[str]
) -> list[dict[str, Any]]:
    """Greedy replies to the held-out attacks with the vector on for a given number of positions from the reply
    opening, `<tool_call>` banned so that only the reply's stance moves."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    banned = [[tokenizer.convert_tokens_to_ids(TOOL_CALL)]]
    layer = model.model.layers[STEER_LAYER]
    held = [row for row in rows if held_out(row) and row["vector"] in tensors]
    out = []
    for spec in conditions:
        alpha, steps = condition(spec)
        for row in held:
            handle = layer.register_forward_hook(SteerHook(tensors[row["vector"]].to(model.device), alpha, steps))
            try:
                ids = torch.tensor([row["plain"]], device=model.device)
                generated = model.generate(ids, max_new_tokens=SWEEP_TOKENS, do_sample=False, bad_words_ids=banned)
                reply = generated[0, ids.shape[1] :]
            finally:
                handle.remove()
            out.append(
                {
                    "record_id": row["record_id"],
                    "rule_id": row["rule_id"],
                    "vector": row["vector"],
                    "alpha": spec,
                    "text": tokenizer.decode(reply, skip_special_tokens=True),
                }
            )
        print(json.dumps({"condition": spec, "replies": len(held)}), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU steps of the behavior vectors.")
    parser.add_argument("step", choices=("sample", "vectors", "sweep", "duration"))
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.0, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--conditions", nargs="*", default=[], help="alpha:steps for the duration sweep")
    args = parser.parse_args()
    rows = _rows(args.pairs)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.step == "sample":
        _write(args.out / "pairs.jsonl", sample_pairs(args.model, rows))
        return
    model = _model(args.model)
    if args.step == "vectors":
        tensors, stats = vectors(model, rows)
        save_file(tensors, str(args.out / VECTORS_FILE))
        (args.out / "vectors.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats), flush=True)
        return
    tensors = load_file(str(args.out / VECTORS_FILE))
    if args.step == "duration":
        _write(args.out / DURATION_FILE, duration_sweep(model, rows, tensors, args.conditions))
        return
    _write(args.out / "sweep.jsonl", sweep(model, rows, tensors, args.alphas))


if __name__ == "__main__":
    main()
