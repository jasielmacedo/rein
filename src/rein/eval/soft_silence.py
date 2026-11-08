"""Does the chat ever meet the soft tokens? Teacher-forces conversations through a plain checkpoint that carries
soft rows and reports the probability mass the soft tokens get at every position, as the tied output head sees them.
The conversations come from decision records (rendered as the generator sees them) or from recorded play sessions."""

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from rein.genview.render import GeneratorView, PromptSpec
from rein.model import constants as mc
from rein.model.backbone import placed
from rein.schema import Message, State
from rein.train.data import load_records
from rein.train.silence import real_logsumexp

TOP_RANK = 100
WORST_POSITIONS = 10
CONTEXT_TOKENS = 16
SESSION_SYSTEM = "You are a character in a game. Stay in character and answer the player."
SOFT_PREFIX = mc.SOFT_TOKEN_FORMAT.split("{", maxsplit=1)[0]


def output_head(checkpoint: str, dtype: torch.dtype, device: torch.device) -> tuple[Qwen3_5TextModel, torch.Tensor]:
    """The text backbone and the table its output head reads: the embedding when tied, the head's own when not."""
    causal_lm = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=dtype)
    backbone = causal_lm.model
    if not isinstance(backbone, Qwen3_5TextModel):
        raise TypeError(f"{checkpoint} did not load a Qwen3.5 text model: {type(backbone).__name__}")
    head = causal_lm.get_output_embeddings()
    if not isinstance(head, nn.Linear):
        raise TypeError(f"{checkpoint} has no linear output head")
    placed(backbone, device)
    return backbone, head.weight.detach().to(device)


def soft_token_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    return sorted(token_id for text, token_id in tokenizer.get_added_vocab().items() if text.startswith(SOFT_PREFIX))


def session_state(path: Path) -> State:
    session = json.loads(path.read_text(encoding="utf-8"))
    conversation: list[Message] = []
    for turn in session["turns"]:
        conversation.append(Message(role="user", text=turn["text"]))
        conversation.append(Message(role="assistant", text=turn["reply"]))
    return State(context=SESSION_SYSTEM, conversation=tuple(conversation))


class SilenceMeter:
    def __init__(self, table: torch.Tensor, soft_ids: list[int], tokenizer: PreTrainedTokenizerBase) -> None:
        self.table = table
        self.soft_ids = torch.tensor(soft_ids, device=table.device)
        self.tokenizer = tokenizer
        self.positions = 0
        self.max_prob = 0.0
        self.mass_sum = 0.0
        self.worst_gap: float | None = None
        self.best_rank: int | None = None
        self.in_top = 0
        self.worst_positions: list[dict[str, Any]] = []

    def _context(self, ids: list[int], position: int, gap: float, soft_index: int) -> dict[str, Any]:
        return {
            "gap_nats": round(gap, 2),
            "soft_token": self.tokenizer.convert_ids_to_tokens(int(self.soft_ids[soft_index])),
            "before": self.tokenizer.decode(ids[max(0, position - CONTEXT_TOKENS) : position + 1]),
            "next": self.tokenizer.decode(ids[position + 1 : position + 2]),
        }

    @torch.no_grad()
    def add(self, hidden: torch.Tensor, ids: list[int]) -> None:
        logits = (hidden @ self.table.T).float()
        soft_logits = logits[:, self.soft_ids]
        real = real_logsumexp(hidden, self.table, self.soft_ids.tolist())
        gap = real - torch.logsumexp(soft_logits, dim=-1)
        mass = torch.exp(-gap) / (1.0 + torch.exp(-gap))
        ranks = (logits > soft_logits.max(dim=1).values[:, None]).sum(dim=1)
        self.positions += int(logits.shape[0])
        self.max_prob = max(self.max_prob, mass.max().item())
        self.mass_sum += mass.sum().item()
        worst, best = gap.min().item(), int(ranks.min().item())
        self.worst_gap = worst if self.worst_gap is None else min(self.worst_gap, worst)
        self.best_rank = best if self.best_rank is None else min(self.best_rank, best)
        self.in_top += int((ranks < TOP_RANK).sum().item())
        lowest = torch.topk(gap, min(WORST_POSITIONS, gap.shape[0]), largest=False)
        loudest = soft_logits.argmax(dim=1)
        self.worst_positions.extend(
            self._context(ids, int(position), float(value), int(loudest[position]))
            for value, position in zip(lowest.values.tolist(), lowest.indices.tolist(), strict=True)
        )
        self.worst_positions.sort(key=lambda item: item["gap_nats"])
        del self.worst_positions[WORST_POSITIONS:]

    def report(self) -> dict[str, Any]:
        return {
            "positions": self.positions,
            "max_soft_mass": self.max_prob,
            "mean_soft_mass": self.mass_sum / max(self.positions, 1),
            "worst_gap_nats": self.worst_gap,
            "best_rank_of_a_soft_token": self.best_rank,
            f"positions_with_a_soft_token_in_top_{TOP_RANK}": self.in_top,
            "worst_positions": self.worst_positions,
        }


def measure(checkpoint: Path, states: list[State], max_tokens: int, device: torch.device) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    soft_ids = soft_token_ids(tokenizer)
    if not soft_ids:
        raise ValueError(f"{checkpoint} has no soft tokens in its tokenizer")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    backbone, table = output_head(str(checkpoint), dtype, device)
    meter = SilenceMeter(table, soft_ids, tokenizer)
    view = GeneratorView(tokenizer)
    for index, state in enumerate(states):
        ids = list(view.render(PromptSpec(system="", state=state), thinking=False, key=str(index)).ids[:max_tokens])
        with torch.no_grad():
            hidden = backbone(input_ids=torch.tensor([ids], device=device)).last_hidden_state[0]
        meter.add(hidden, ids)
    return {"checkpoint": str(checkpoint), "soft_ids": soft_ids, "conversations": len(states), **meter.report()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the soft tokens' output mass over chat text.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="plain checkpoint with the soft rows merged")
    parser.add_argument("--data", type=Path, nargs="*", default=[], help="decision records to render as chat")
    parser.add_argument("--sessions", type=Path, default=None, help="directory of recorded play sessions")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    states = [record.state for path in args.data for record in load_records(path)]
    if args.sessions is not None:
        states.extend(session_state(Path(file)) for file in sorted(glob.glob(str(args.sessions / "*.json"))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = measure(args.checkpoint, states[: args.limit], args.max_tokens, device)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
