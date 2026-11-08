import argparse
import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from rein.datagen.generate import ChatModel, Stats, ask_json
from rein.datagen.mix import SYNTHETIC_SPLITS
from rein.datagen.pipeline import Stage
from rein.datagen.prompts import PROMPT_VERSION, audit_prompt
from rein.env import load_env, require
from rein.schema import Question, TrainingRecord
from rein.teacher import constants as tc
from rein.teacher.client import TeacherClient
from rein.teacher.modal_ops import stop_app_containers
from rein.teacher.settings import TeacherSettings
from rein.teacher.store import content_key
from rein.train.data import load_records

AUDITOR_APP = "rein-auditor"
AUDITOR_MODEL = "Qwen/Qwen3.8-27B-FP8"
ENV_AUDITOR_URL = "REIN_AUDITOR_URL"
AUDIT_VERSION = "a2"
SCREEN_MAX_TOKENS = 1024
SCREEN_TEMPERATURE = 0.0
THINKING_MAX_TOKENS = 16384
THINKING_TEMPERATURE = 1.0
THINKING_TOP_P = 0.95
THINKING_VOTES = 2
SCORE_CORRECTION_MIN_GAP = 2
DEFAULT_WORKERS = 64
COLD_START_TIMEOUT_S = 30 * 60.0
DEFAULT_SOURCES = ("synthetic",)
PROGRESS_EVERY = 200
DISPUTED_LABEL_SHARE = 2 / 3

Status = Literal["confirmed", "corrected", "disputed", "abstained"]


class AuditVerdict(BaseModel):
    answers: dict[str, str]


@dataclass(frozen=True)
class QuestionAudit:
    status: Status
    reference: str
    votes: tuple[str | None, ...]
    corrected: str | None = None


def reference_option(question: Question) -> str:
    """The option the record currently teaches: its hard label, or the top of its soft target."""
    keys = question.option_keys()
    index = question.label_index()
    if index is not None:
        return keys[index]
    target = question.target_vector()
    assert target is not None
    return keys[max(range(len(keys)), key=lambda i: target[i])]


def label_value(question: Question, option: str) -> bool | int | str:
    if question.type == "noul":
        return option == "yes"
    if question.type == "score":
        return int(option)
    return option


def judge(question: Question, reference: str, screen: str | None, thinking: tuple[str | None, ...]) -> QuestionAudit:
    """A cheap screen vote clears agreements; overturning a label needs every thinking vote to agree on one option.

    On a score scale, a one-level disagreement is a matter of degree, so it softens the target instead.
    """
    votes = (screen, *thinking)
    if screen == reference:
        return QuestionAudit("confirmed", reference, votes)
    valid = [vote for vote in thinking if vote is not None]
    if not valid:
        return QuestionAudit("abstained", reference, votes)
    if all(vote == reference for vote in valid):
        return QuestionAudit("confirmed", reference, votes)
    unanimous = len(valid) == THINKING_VOTES and len(set(valid)) == 1
    near = question.type == "score" and abs(int(valid[0]) - int(reference)) < SCORE_CORRECTION_MIN_GAP
    if unanimous and not near:
        return QuestionAudit("corrected", reference, votes, valid[0])
    return QuestionAudit("disputed", reference, votes)


def apply_audit(question: Question, audit: QuestionAudit) -> Question:
    if audit.status == "corrected":
        assert audit.corrected is not None
        return question.model_copy(update={"label": label_value(question, audit.corrected), "target": None})
    if audit.status == "disputed":
        others = sorted({vote for vote in audit.votes[1:] if vote is not None and vote != audit.reference})
        share = (1 - DISPUTED_LABEL_SHARE) / len(others)
        target = {audit.reference: DISPUTED_LABEL_SHARE, **{vote: share for vote in others}}
        return question.model_copy(update={"target": target})
    return question


def _valid(record: TrainingRecord, verdict: AuditVerdict | None) -> dict[str, str | None]:
    answers = verdict.answers if verdict else {}
    return {
        qid: answers.get(qid) if answers.get(qid) in question.option_keys() else None
        for qid, question in record.questions.items()
    }


@dataclass
class Auditor:
    model: ChatModel
    stage: Stage
    stats: Stats = field(default_factory=Stats)

    async def _vote(self, record: TrainingRecord, index: int) -> dict[str, str | None]:
        prompt = audit_prompt(record)
        thinking = index > 0

        async def produce() -> AuditVerdict:
            verdict = await ask_json(
                self.model,
                prompt,
                AuditVerdict,
                self.stats,
                f"audit_vote_{index}",
                max_tokens=THINKING_MAX_TOKENS if thinking else SCREEN_MAX_TOKENS,
                temperature=THINKING_TEMPERATURE if thinking else SCREEN_TEMPERATURE,
                top_p=THINKING_TOP_P if thinking else None,
                thinking=thinking,
            )
            return verdict or AuditVerdict(answers={})

        key = content_key("audit", AUDIT_VERSION, PROMPT_VERSION, AUDITOR_MODEL, record.id, index, prompt)
        return _valid(record, await self.stage.run(key, AuditVerdict, produce))

    async def audit(self, record: TrainingRecord) -> dict[str, QuestionAudit]:
        references = {qid: reference_option(question) for qid, question in record.questions.items()}
        screen = await self._vote(record, 0)
        flagged = [qid for qid, vote in screen.items() if vote != references[qid]]
        thinking: list[dict[str, str | None]] = []
        if flagged:
            thinking = list(await asyncio.gather(*(self._vote(record, i + 1) for i in range(THINKING_VOTES))))
        return {
            qid: judge(
                question,
                references[qid],
                screen[qid],
                tuple(vote[qid] for vote in thinking) if qid in flagged else (),
            )
            for qid, question in record.questions.items()
        }


def audited_record(record: TrainingRecord, audits: dict[str, QuestionAudit]) -> TrainingRecord:
    questions = {qid: apply_audit(question, audits[qid]) for qid, question in record.questions.items()}
    meta = {
        **record.meta,
        "audit": {
            "model": AUDITOR_MODEL,
            "version": AUDIT_VERSION,
            "questions": {
                qid: {
                    "status": audit.status,
                    "reference": audit.reference,
                    "votes": list(audit.votes),
                    "corrected": audit.corrected,
                }
                for qid, audit in audits.items()
            },
        },
    }
    return record.model_copy(update={"questions": questions, "meta": meta})


async def audit_split(
    auditor: Auditor, records: list[TrainingRecord], sources: tuple[str, ...], workers: int, label: str
) -> tuple[list[TrainingRecord], Counter[str]]:
    gate = asyncio.Semaphore(workers)
    counts: Counter[str] = Counter()
    done = 0

    async def one(record: TrainingRecord) -> TrainingRecord:
        nonlocal done
        if record.source not in sources:
            return record
        async with gate:
            audits = await auditor.audit(record)
        for audit in audits.values():
            counts[audit.status] += 1
        done += 1
        if done % PROGRESS_EVERY == 0:
            print(json.dumps({"split": label, "records": done, **counts}), flush=True)
        return audited_record(record, audits)

    return list(await asyncio.gather(*(one(record) for record in records))), counts


def auditor_settings() -> TeacherSettings:
    env = load_env()
    return TeacherSettings.from_env(
        {**env, tc.ENV_TEACHER_URL: require(env, ENV_AUDITOR_URL), tc.ENV_TEACHER_MODEL: AUDITOR_MODEL}
    )


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"model": AUDITOR_MODEL, "version": AUDIT_VERSION}
    async with TeacherClient(auditor_settings(), max_in_flight=args.workers) as client:
        print(f"auditor healthy after {await client.wait_healthy(COLD_START_TIMEOUT_S):.0f}s", flush=True)
        auditor = Auditor(client, Stage(args.output / "audit_votes.jsonl"))
        for split in args.splits:
            records = load_records(args.input / f"{split}.jsonl")[: args.limit]
            audited, counts = await audit_split(auditor, records, tuple(args.sources), args.workers, split)
            with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                for record in audited:
                    handle.write(record.model_dump_json() + "\n")
            summary[split] = {"records": len(audited), **counts}
            print(json.dumps({split: summary[split]}), flush=True)
        summary["completion_tokens"] = auditor.stats.completion_tokens
        summary["rejected"] = dict(auditor.stats.rejected)
    (args.output / "audit_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and correct record labels with an independent Qwen model.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=list(SYNTHETIC_SPLITS), choices=SYNTHETIC_SPLITS)
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep-auditor", action="store_true", help="leave the Modal container running afterwards")
    args = parser.parse_args()
    try:
        print(json.dumps(asyncio.run(_run(args)), indent=2))
    finally:
        if not args.keep_auditor:
            print("stopped containers:", stop_app_containers(AUDITOR_APP))


if __name__ == "__main__":
    main()
