import asyncio
from collections.abc import Sequence
from typing import Any

import httpx

from rein.eval.metrics import PredictionRow
from rein.public.common import render_value
from rein.schema import TrainingRecord

SYSTEMONE_PATH = "/v1/systemone"
REQUEST_TIMEOUT_S = 300.0
DEFAULT_CONCURRENCY = 8
NOUL_KEYS = ("no", "yes")


def render_state(record: TrainingRecord) -> str:
    """Plain-text state for baselines that take a string: context, transcript and reply in progress."""
    parts = []
    if record.state.context:
        parts.append(f"Context:\n{record.state.context}")
    lines = [f"{message.name or message.role}: {message.text}" for message in record.state.conversation]
    parts.append("Conversation:\n" + "\n".join(lines))
    if record.state.draft is not None:
        calls = render_value([call.model_dump() for call in record.state.draft.tool_calls]) or "none"
        parts.append(f"Reply in progress:\n{record.state.draft.text}\nTool calls emitted: {calls}")
    return "\n\n".join(parts)


def to_request(record: TrainingRecord) -> dict[str, Any]:
    questions = {}
    for qid, question in record.questions.items():
        body: dict[str, Any] = {"type": question.type, "instructions": question.instructions}
        if question.criteria is not None:
            body["criteria"] = question.criteria
        questions[qid] = body
    return {"state": render_state(record), "questions": questions}


def probabilities(answer: dict[str, Any], option_keys: tuple[str, ...]) -> tuple[float, ...]:
    if answer.get("type") == "noul":
        yes = float(answer["noul"])
        return (1.0 - yes, yes)
    table = answer.get("probabilities") or {}
    values = [float(table.get(key, 0.0)) for key in option_keys]
    total = sum(values)
    return tuple(value / total for value in values) if total > 0 else tuple(1.0 / len(values) for _ in values)


def rows_from_answers(record: TrainingRecord, answers: dict[str, Any]) -> list[PredictionRow]:
    rows = []
    for qid, question in record.questions.items():
        label = question.label_index()
        if label is None or qid not in answers:
            continue
        keys = question.option_keys()
        rows.append(
            PredictionRow(
                record_id=record.id,
                question_id=qid,
                question_type=question.type,
                option_keys=keys,
                probs=probabilities(answers[qid], keys),
                label_index=label,
                tags={"split": record.split, "source": record.source, **record.meta},
            )
        )
    return rows


async def score_remote(
    base_url: str, records: Sequence[TrainingRecord], concurrency: int = DEFAULT_CONCURRENCY, model: str | None = None
) -> list[PredictionRow]:
    """Scores records on any System One compatible server (TinyJev, Kev) for a same-items comparison."""
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=REQUEST_TIMEOUT_S) as client:

        async def one(record: TrainingRecord) -> list[PredictionRow]:
            payload = to_request(record)
            if model:
                payload["model"] = model
            async with semaphore:
                response = await client.post(SYSTEMONE_PATH, json=payload)
            response.raise_for_status()
            return rows_from_answers(record, response.json()["answers"])

        results = await asyncio.gather(*(one(record) for record in records))
    return [row for rows in results for row in rows]
