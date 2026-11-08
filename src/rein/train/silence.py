"""Keeps the soft tokens out of the chat. With tied embeddings the soft rows are output rows too, so at every chat
position the generator could sample them. The bank holds the full generator's final hidden states over conversations
rendered as the generator sees them, with the log-sum-exp of the real vocabulary's logits at each position; the loss
asks the soft rows to stay `margin` nats below that everywhere, and is zero once they do."""

import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from rein.genview.render import GeneratorView, PromptSpec
from rein.model.soft import input_embeddings
from rein.schema import State

LOGIT_CHUNK = 512
NO_SYSTEM = ""


@dataclass(frozen=True)
class OutputBank:
    hidden: torch.Tensor
    real_logsumexp: torch.Tensor

    def to(self, device: torch.device) -> "OutputBank":
        return OutputBank(self.hidden.to(device), self.real_logsumexp.to(device))


def real_logsumexp(hidden: torch.Tensor, table: torch.Tensor, excluded: Sequence[int]) -> torch.Tensor:
    """log-sum-exp of the logits every real token takes at each hidden state; the soft rows are left out."""
    values = []
    for start in range(0, hidden.shape[0], LOGIT_CHUNK):
        logits = (hidden[start : start + LOGIT_CHUNK] @ table.T).float()
        if excluded:
            logits[:, list(excluded)] = torch.finfo(logits.dtype).min
        values.append(torch.logsumexp(logits, dim=-1))
    return torch.cat(values)


@torch.no_grad()
def build_output_bank(
    backbone: Qwen3_5TextModel,
    tokenizer: PreTrainedTokenizerBase,
    states: Sequence[State],
    soft_ids: Sequence[int],
    max_positions: int,
    max_tokens: int,
    seed: int,
) -> OutputBank:
    """Final hidden states of the full generator over the conversations, about `max_positions` of them, sampled
    evenly across the states. The backbone must still have all its layers."""
    view = GeneratorView(tokenizer)
    device = next(backbone.parameters()).device
    table = input_embeddings(backbone).weight
    rng = random.Random(seed)
    per_state = max(1, -(-max_positions // max(len(states), 1)))
    hidden_parts: list[torch.Tensor] = []
    logsumexp_parts: list[torch.Tensor] = []
    was_training = backbone.training
    backbone.eval()
    for index, state in enumerate(states):
        ids = view.render(PromptSpec(system=NO_SYSTEM, state=state), thinking=False, key=str(index)).ids[:max_tokens]
        hidden = backbone(input_ids=torch.tensor([ids], device=device)).last_hidden_state[0]
        chosen = torch.tensor(sorted(rng.sample(range(len(ids)), min(per_state, len(ids)))), device=device)
        picked = hidden.index_select(0, chosen)
        hidden_parts.append(picked.float())
        logsumexp_parts.append(real_logsumexp(picked, table, soft_ids))
    backbone.train(was_training)
    return OutputBank(torch.cat(hidden_parts), torch.cat(logsumexp_parts))


def output_directions(hidden: torch.Tensor, rank: int) -> torch.Tensor:
    """The `rank` directions that carry the most energy of the output states, orthonormal, one per row."""
    moment = hidden.T.float() @ hidden.float() / hidden.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(moment)
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    return eigenvectors[:, order].T.contiguous()


def silence_loss(
    bank: OutputBank, rows: torch.Tensor, margin: float, tail: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """The hinge averaged over the worst `tail` fraction of the bank's positions (1.0 means all of them), and the
    smallest gap: how many nats the real vocabulary leads the soft rows by at the worst position (negative means a
    soft row could be sampled there)."""
    soft = torch.logsumexp(bank.hidden @ rows.T.to(bank.hidden.dtype), dim=-1)
    gap = bank.real_logsumexp - soft
    hinge = torch.relu(margin - gap)
    count = max(1, round(tail * hinge.shape[0]))
    worst = hinge.topk(count).values if count < hinge.shape[0] else hinge
    return worst.mean(), gap.min().detach()
