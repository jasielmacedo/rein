import argparse
import json
import re
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict

from rein.control.backend import DeciderBackend
from rein.control.generator import GeneratorRequest, chat_messages
from rein.control.induction import QwenXmlToolCall
from rein.control.local_generator import Decoding, TransformersGenerator
from rein.control.session import RuleSession
from rein.control.sheet import RuleSheet
from rein.control.visible import TOOL_CALL_MARKER
from rein.e2e.run import BASELINE, DEFAULT_GENERATOR, REIN
from rein.env import load_env, optional
from rein.model.loader import load_decider
from rein.schema import Message, State

TRANSCRIPT_FILE = "transcript.md"
SEED_STRIDE = 1000
TURNS_FILE = "turns.json"
SELF_NAMING = r"\b(i am|i'm|my name is|eu sou|me chamo)\s+{name}\b"


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "narrator"] = "user"
    text: str
    expect: str


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    system: str
    context: str
    tools: list[dict[str, Any]]
    sheet: RuleSheet
    turns: list[Turn]
    tool_result: str
    keep_name: str | None = None
    forbidden_name: str | None = None
    judge_rules: dict[str, str] = {}


def _visible(reply: str) -> str:
    return reply.split(TOOL_CALL_MARKER, 1)[0].strip()


def _plain(generator: TransformersGenerator, state: State, system: str, prefix: str = "") -> str:
    return "".join(generator.stream(GeneratorRequest(chat_messages(state, system), prefix)))


class Arm:
    """One side of the comparison: its own conversation history, which drifts with its own replies."""

    def __init__(
        self, name: str, scenario: Scenario, generator: TransformersGenerator, session: RuleSession | None, seed: int
    ):
        self.name = name
        self.seed = seed
        self.scenario = scenario
        self.generator = generator
        self.session = session
        self.history: list[Message] = []

    def state(self) -> State:
        return State(context=self.scenario.context, conversation=tuple(self.history))

    def step(self, turn: Turn) -> dict[str, Any]:
        torch.manual_seed(self.seed * SEED_STRIDE + len(self.history))
        self.history.append(Message(role=turn.role, text=turn.text))
        audit: dict[str, Any] | None = None
        if self.session is None:
            reply = _plain(self.generator, self.state(), self.scenario.system)
            call = QwenXmlToolCall().parse(reply)
        else:
            result = self.session.run_turn(self.state())
            reply, call, audit = result.reply, result.tool_call, result.audit
        visible = _visible(reply)
        follow_up = ""
        self.history.append(Message(role="assistant", text=visible, tool_calls=(call,) if call else ()))
        if call is not None:
            self.history.append(Message(role="tool", name=call.name, text=self.scenario.tool_result))
            follow_up = _visible(_plain(self.generator, self.state(), self.scenario.system))
            self.history.append(Message(role="assistant", text=follow_up))
        return {"reply": visible, "tool": call.model_dump() if call else None, "follow_up": follow_up, "audit": audit}


def _score(scenario: Scenario, turn: Turn, step: dict[str, Any]) -> dict[str, Any]:
    spoken = f"{step['reply']} {step['follow_up']}".lower()
    tool = step["tool"]["name"] if step["tool"] else None
    tools = {spec["function"]["name"] for spec in scenario.tools}
    score: dict[str, Any] = {"tool": tool, "action_ok": tool == turn.expect if turn.expect in tools else tool is None}
    if scenario.keep_name:
        score["says_own_name"] = scenario.keep_name.lower() in spoken
    if scenario.forbidden_name:
        pattern = SELF_NAMING.format(name=scenario.forbidden_name.lower())
        score["adopts_new_name"] = re.search(pattern, spoken) is not None
    return score


def _transcript(scenario: Scenario, turns: list[dict[str, Any]]) -> str:
    lines = [f"# {scenario.name}", ""]
    for number, row in enumerate(turns, 1):
        lines += [f"## Turn {number} ({row['expect']})", "", f"**{row['role']}:** {row['text']}", ""]
        for arm in (BASELINE, REIN):
            step, score = row[arm], row[arm]["score"]
            call = (
                f" `{step['tool']['name']}({json.dumps(step['tool']['arguments'], ensure_ascii=False)})`"
                if step["tool"]
                else ""
            )
            lines.append(f"**{arm}**{call}: {step['reply'] or '(no text)'}")
            if step["follow_up"]:
                lines.append(f"  *after the tool:* {step['follow_up']}")
            lines.append("  - " + ", ".join(f"{key}: {value}" for key, value in score.items()))
            lines.append("")
    return "\n".join(lines)


def run_session(scenario: Scenario, generator: TransformersGenerator, session: RuleSession, seed: int) -> list[dict]:
    arms = {
        BASELINE: Arm(BASELINE, scenario, generator, None, seed),
        REIN: Arm(REIN, scenario, generator, session, seed),
    }
    turns = []
    for turn in scenario.turns:
        row: dict[str, Any] = {"role": turn.role, "text": turn.text, "expect": turn.expect}
        for name, arm in arms.items():
            step = arm.step(turn)
            row[name] = {**step, "score": _score(scenario, turn, step)}
        turns.append(row)
    return turns


def main() -> None:
    parser = argparse.ArgumentParser(description="One dense, multi-turn session: Qwen alone versus Qwen with Rein.")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--rein", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--generator", default=DEFAULT_GENERATOR)
    parser.add_argument("--seeds", type=int, default=1, help="independent sessions; each one samples instead of greedy")
    args = parser.parse_args()
    scenario = Scenario.model_validate(yaml.safe_load(args.scenario.read_text(encoding="utf-8")))
    token = optional(load_env(), "HF_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoding = Decoding(sample=args.seeds > 1)
    generator = TransformersGenerator(args.generator, device, tools=scenario.tools, token=token, decoding=decoding)
    session = RuleSession(
        sheet=scenario.sheet,
        backend=DeciderBackend(load_decider(args.rein, device, token)),
        generator=generator,
        syntax=QwenXmlToolCall(),
        system=scenario.system,
    )
    for seed in range(args.seeds):
        out = args.out / f"seed-{seed}" if args.seeds > 1 else args.out
        turns = run_session(scenario, generator, session, seed)
        out.mkdir(parents=True, exist_ok=True)
        (out / TURNS_FILE).write_text(json.dumps(turns, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / TRANSCRIPT_FILE).write_text(_transcript(scenario, turns), encoding="utf-8")
        print(f"wrote {out / TRANSCRIPT_FILE}", flush=True)


if __name__ == "__main__":
    main()
