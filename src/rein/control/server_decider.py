"""Decisions through a llama-server that serves the merged soft-token GGUF: the request is encoded in Rein's layout,
sent as token ids, and the labels' next-token log-probabilities are the decision. No second model, no PyTorch."""

import math
from pathlib import Path

import httpx
from transformers import AutoTokenizer

from rein.control.plan import Probabilities
from rein.model.encode import RequestEncoder
from rein.model.run_encoder import build_encoder
from rein.model.run_meta import read_meta
from rein.schema import DecisionRequest

TOP_LOGPROBS = 60
MISSING_LOGPROB = -60.0
TIMEOUT_S = 600.0


def label_probabilities(
    logprobs: dict[int, float], label_ids: tuple[int, ...], temperature: float
) -> tuple[float, ...]:
    """The labels' share of the next token, renormalized among themselves at the run's temperature."""
    scaled = [logprobs.get(token, MISSING_LOGPROB) / temperature for token in label_ids]
    top = max(scaled)
    weights = [math.exp(value - top) for value in scaled]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


class ServerDecisionBackend:
    def __init__(self, url: str, run_dir: Path, encoder: RequestEncoder | None = None) -> None:
        meta = read_meta(run_dir)
        self.url = url.rstrip("/")
        self.temperature = float(meta["temperature"])
        self.encoder = encoder or build_encoder(meta, AutoTokenizer.from_pretrained(meta["base_repo"]))

    def _next_token_logprobs(self, prompt: list[int]) -> dict[int, float]:
        body = {"prompt": prompt, "n_predict": 1, "n_probs": TOP_LOGPROBS, "temperature": 0.0, "cache_prompt": True}
        reply = httpx.post(f"{self.url}/completion", json=body, timeout=TIMEOUT_S)
        reply.raise_for_status()
        top = reply.json()["completion_probabilities"][0]["top_logprobs"]
        return {int(item["id"]): float(item["logprob"]) for item in top}

    def decide(self, request: DecisionRequest) -> Probabilities:
        encoded = self.encoder.encode(request)
        decisions: dict[str, dict[str, float]] = {}
        for question in encoded.questions:
            if not question.label_token_ids:
                raise ValueError("the server backend needs the lettered layout (readout labels)")
            logprobs = self._next_token_logprobs([*encoded.state_ids, *question.rows[0]])
            probabilities = label_probabilities(logprobs, question.label_token_ids, self.temperature)
            decisions[question.question_id] = dict(zip(question.option_keys, probabilities, strict=True))
        return decisions
