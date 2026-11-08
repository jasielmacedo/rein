from collections.abc import Sequence
from dataclasses import dataclass
from threading import Event, Thread
from typing import Any, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from rein.control.generator import Chunks, GeneratorRequest, template_message, term_variants

DEFAULT_MAX_NEW_TOKENS = 192
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P = 0.8
SAMPLE_TOP_K = 20
SAMPLE_REPETITION_PENALTY = 1.15
TOOL_CALL_END = "</tool_call>"
STOP_TAIL_TOKENS = 12


def _flags(input_ids: torch.LongTensor, value: bool) -> torch.BoolTensor:
    return cast(torch.BoolTensor, torch.full((input_ids.shape[0],), value, dtype=torch.bool, device=input_ids.device))


class _Cancelled(StoppingCriteria):
    """Ends generation once the reader stops consuming, for example after the watcher cut the reply."""

    def __init__(self, cancelled: Event) -> None:
        self.cancelled = cancelled

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> torch.BoolTensor:
        return _flags(input_ids, self.cancelled.is_set())


class _StopOnText(StoppingCriteria):
    """Stops once the newest tokens spell a stop string; decodes only a short tail, never the vocabulary."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, prompt_length: int, stops: Sequence[str]) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.stops = tuple(stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> torch.BoolTensor:
        tail = input_ids[0, max(self.prompt_length, input_ids.shape[1] - STOP_TAIL_TOKENS) :]
        text = self.tokenizer.decode(tail, skip_special_tokens=False)
        return _flags(input_ids, any(stop in text for stop in self.stops))


@dataclass(frozen=True)
class Decoding:
    """Greedy by default; sampling follows the Qwen non-thinking recommendation plus a repetition penalty."""

    sample: bool = False
    temperature: float = SAMPLE_TEMPERATURE
    top_p: float = SAMPLE_TOP_P
    top_k: int = SAMPLE_TOP_K
    repetition_penalty: float = SAMPLE_REPETITION_PENALTY

    def kwargs(self) -> dict[str, Any]:
        if not self.sample:
            return {"do_sample": False}
        return {
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }


GREEDY = Decoding()


class TransformersGenerator:
    """A local chat model that continues a prefilled reply and stops after a complete tool call."""

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        tools: Sequence[dict[str, Any]] = (),
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        token: str | None = None,
        decoding: Decoding = GREEDY,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_id, token=token)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, token=token, device_map=device).eval()
        self.device = device
        self.tools: list[Any] = list(tools)
        self.max_new_tokens = max_new_tokens
        self.decoding = decoding

    def prompt(self, request: GeneratorRequest) -> str:
        messages = [template_message(message) for message in request.messages]
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=(self.tools or None) if request.tools_enabled else None,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        return str(text) + request.assistant_prefix

    def banned_ids(self, terms: Sequence[str]) -> list[list[int]]:
        """Token sequences the sampler may not complete; a multi-token term loses only its last token."""
        encoded = (self.tokenizer.encode(form, add_special_tokens=False) for form in term_variants(terms))
        return [list(ids) for ids in {tuple(ids) for ids in encoded if ids}]

    def stream(self, request: GeneratorRequest) -> Chunks:
        inputs = self.tokenizer(self.prompt(request), return_tensors="pt", add_special_tokens=False).to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)
        cancelled = Event()
        banned = self.banned_ids(request.banned_terms)
        failures: list[BaseException] = []

        def generate(**kwargs: Any) -> None:
            try:
                self.model.generate(**kwargs)
            except BaseException as exc:
                failures.append(exc)
                streamer.end()

        worker = Thread(
            target=generate,
            kwargs={
                **inputs,
                "streamer": streamer,
                "max_new_tokens": self.max_new_tokens,
                **self.decoding.kwargs(),
                **({"bad_words_ids": banned} if banned else {}),
                "stopping_criteria": StoppingCriteriaList(
                    [
                        _Cancelled(cancelled),
                        _StopOnText(self.tokenizer, inputs["input_ids"].shape[1], (TOOL_CALL_END, *request.stop)),
                    ]
                ),
            },
            daemon=True,
        )
        worker.start()
        eos = self.tokenizer.eos_token or ""
        finished = False
        try:
            for chunk in streamer:
                text = chunk.replace(eos, "") if eos else chunk
                if text:
                    yield text
            finished = True
        finally:
            cancelled.set()
            if not finished:
                for _ in streamer:
                    pass
            worker.join()
        if failures:
            raise RuntimeError("generation failed") from failures[0]
