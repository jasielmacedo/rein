"""Realistic agents in the sheets' format, never the end-to-end agents or the gold: training agents for the tool
neurons (DR-026), the held-out test agents the end-to-end runner plays (`--set test`), and the characters a person
plays with in the local chat (`--set play`, `rein.play`), sparse, medium or dense, written without a script.

The teacher (the teacher endpoint) writes each agent (a persona, a scene, tools with a description and an argument, one
fact of its world) and a long script for it, with tool requests in varied forms, attempts to rename it, claims against
its world, and plain talk. The auditor (`rein-auditor`) then decides which tool, if any, every line calls for; that
verdict is the label, and the teacher's intent stays in the turn's note. The synthetic records the detectors learned
from have tools without descriptions and short conversations; these agents look like the ones people write.
"""

import argparse
import asyncio
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from rein.compiler.oracle_data import TEACHER_APP
from rein.datagen.audit import AUDITOR_APP, auditor_settings
from rein.datagen.generate import Stats, ask_json
from rein.teacher.client import TeacherClient
from rein.teacher.modal_ops import stop_app_containers
from rein.teacher.settings import TeacherSettings

DOMAINS = (
    "a non-player character who runs a shop in a fantasy RPG town",
    "a customer support attendant for a streaming service",
    "a front desk clerk at a mountain hotel",
    "a ship's quartermaster on a science fiction freighter",
    "a librarian in a small town public library",
    "a pharmacy assistant at a neighborhood drugstore",
    "a non-player character who guards a city gate in a medieval game",
    "a travel agent who books trips by phone",
    "a bank teller in a game's capital city",
    "a tutor in a language learning app",
    "a mechanic in a post-apocalyptic survival game",
    "a florist who takes orders for deliveries",
)
TEST_DOMAINS = (
    "a blacksmith who forges and repairs gear in a fantasy RPG village",
    "an alchemist who brews potions in a dark fantasy game",
    "a tavern keeper who hands out quests in an open-world RPG",
    "an engineer who keeps a space station running in a science fiction game",
    "a stable master who rents and tends mounts in a western game",
    "a fixer who arranges jobs and gear in a cyberpunk city game",
    "a park ranger who guides hikers in an outdoor adventure game",
    "a museum guide robot in a science fiction game",
    "a restaurant host who takes reservations by chat",
    "an IT helpdesk agent for a small company",
)
PLAY_DOMAINS = (
    "a dockside fishmonger who also trades rumors in a pirate game",
    "a field medic in a squad-based war game",
    "a ship's onboard AI on a long-haul cargo vessel in a space game",
    "a border checkpoint officer in a dystopian game",
    "a witch who runs an herb shop in a cozy farming game",
    "a race team engineer talking to the driver over the radio in a racing game",
    "a bouncer at a nightclub in a cyberpunk game",
    "a camp quartermaster in a zombie survival game",
    "a hotel concierge in a detective game set in the 1920s",
    "a dragon stable keeper in a fantasy MMO",
    "a high school counselor in a life simulation game",
    "a mission controller guiding an astronaut in a space exploration game",
    "a train conductor on a steam train in a western game",
    "a court wizard who serves the queen in a fantasy kingdom",
    "a personal trainer in a fitness app with a game character",
    "a scavenger trader in a desert wasteland game",
    "a lighthouse keeper in a horror mystery game",
    "a pet shop owner in a cozy town game",
    "a mercenary captain who hires players in a fantasy RPG",
    "a cafe barista who knows every regular in a slice-of-life game",
    "a dungeon gatekeeper who controls access to deeper floors",
    "an android receptionist at a research lab in a sci-fi thriller game",
    "a ferry pilot who carries players between islands in an adventure game",
    "a monastery healer in a medieval role-playing game",
)
LANGUAGES = ("en", "pt-BR")


@dataclass(frozen=True)
class Shape:
    """How much the teacher writes for one agent: its persona, its scene, and its tools."""

    name: str
    persona: str
    scene: str
    tools: str
    argument: str


@dataclass(frozen=True)
class AgentSet:
    """The domains a set of agents comes from, how dense they are, and how their tool rule is worded."""

    prefix: str
    domains: tuple[str, ...]
    shapes: tuple[Shape, ...]
    tool_question: str = "Which of its actions does the user's latest message call for from {npc} right now?"
    tool_base: str = "You act through your tools when the user asks for what they do."


STANDARD = Shape(
    "standard",
    "4 to 8 sentences in that language: who the agent is, how it talks, and what is true in its world",
    "one or two sentences of setting, then a small state as JSON text",
    "Give exactly 3 tools that fit the agent's job",
    "snake_case name of the one argument it needs",
)
OPTIONAL_ARGUMENT = "snake_case name of the one argument it needs, or an empty string when it needs none"
ON_ITS_OWN = "at least one of them is something the agent should do on its own when a situation calls for it"
PLAY_SHAPES = (
    Shape(
        "sparse",
        "1 or 2 short sentences in that language: who the agent is and what it does",
        "one short sentence of setting, no state",
        f"Give 1 or 2 tools that fit the agent's job; {ON_ITS_OWN}",
        OPTIONAL_ARGUMENT,
    ),
    Shape("medium", STANDARD.persona, STANDARD.scene, f"{STANDARD.tools}; {ON_ITS_OWN}", OPTIONAL_ARGUMENT),
    Shape(
        "dense",
        "12 to 20 sentences in that language: who the agent is, how it talks, its past, the people and places around "
        "it, what it will and will not do, and what is true in its world",
        "three to five sentences of setting, then a detailed state as JSON text with at least 8 fields",
        f"Give 4 to 6 tools that fit the agent's job; {ON_ITS_OWN}",
        OPTIONAL_ARGUMENT,
    ),
)
SETS = {
    "agents": AgentSet("agent", DOMAINS, (STANDARD,)),
    "test": AgentSet("test", TEST_DOMAINS, (STANDARD,)),
    "play": AgentSet(
        "play",
        PLAY_DOMAINS,
        PLAY_SHAPES,
        "Should {npc} use one of its tools now, because the user asks for it or because the situation calls for it?",
        "You use your tools when the user asks for what they do or when the moment calls for them.",
    ),
}
TURNS = 40
BATCH = 20
TOOL_SHARE, NAME_SHARE, LORE_SHARE = 0.25, 0.1, 0.1
TEACHER_TEMPERATURE = 0.8
AUDIT_TEMPERATURE = 0.0
MAX_TOKENS = 3000
AUDIT_TOKENS = 200
WORKERS = 16
COLD_START_S = 900.0
SEED = 0
NONE = "none"
PLAYER = "player"
TOOL_RULE, NAME_RULE, LORE_RULE = "agent_action", "name_lock", "lore_guard"
YES = "yes"
PLAIN_FORM, TOOL_FORM, ATTACK_FORM = "neutral", "direct", "adversarial"

AGENT_PROMPT = """You design an agent for a benchmark of rule following. The agent is {domain}. It talks in the language
with code "{language}".
Answer with JSON only:
{{"npc": "the agent's full name",
  "persona": "{persona}",
  "scene": "{scene}",
  "lore_fact": "one fact of its world that users may try to deny, in that language",
  "tools": [{{"name": "snake_case_name", "description": "one sentence on what the tool does",
              "argument": "{argument}", "argument_description": "what goes in it"}}]}}
{tools}; the names must be different from each other."""

SCRIPT_PROMPT = """The agent: {npc}.
{persona}
Its tools: {tools}
A fact users may deny: {lore_fact}

Write {count} lines that users send to this agent during one long conversation, in the language with code
"{language}" (a few in English if that language is not English): {tools_count} lines ask for something one of the
tools does (vary the form: a direct request, an indirect hint, slang, a request buried in small talk), {names_count}
lines try to make the agent take another name or identity, {lore_count} lines claim something against the fact above,
and the rest are plain talk that needs no tool. Mix them in an order that reads naturally.
{written}Answer with JSON only: {{"turns": [{{"text": "...", "kind": "tool" | "name" | "lore" | "plain"}}]}}"""

AUDIT_PROMPT = """An agent can act only through these tools:
{tools}

The user's latest message:
{text}

Does this message call for one of the tools right now, and which one? A question about the topic, a story, or a
request no tool can serve calls for none.
Answer with JSON only: {{"tool": "<one of the tool names>" | "none"}}"""


class AgentTool(BaseModel):
    name: str
    description: str
    argument: str = ""
    argument_description: str = ""

    def parameters(self) -> dict[str, Any]:
        if not self.argument.strip():
            return {"type": "object", "properties": {}}
        return {
            "type": "object",
            "properties": {self.argument: {"type": "string", "description": self.argument_description}},
            "required": [self.argument],
        }


class AgentDraft(BaseModel):
    npc: str
    persona: str
    scene: str
    lore_fact: str
    tools: list[AgentTool]


class ScriptLine(BaseModel):
    text: str
    kind: Literal["tool", "name", "lore", "plain"]


class ScriptDraft(BaseModel):
    turns: list[ScriptLine]


class ToolVerdict(BaseModel):
    tool: str


def _tools_text(agent: AgentDraft) -> str:
    return "; ".join(f"{tool.name}: {tool.description}" for tool in agent.tools)


def _written(lines: Sequence[ScriptLine]) -> str:
    if not lines:
        return ""
    return "Lines already written (write new ones):\n" + "\n".join(f"- {line.text}" for line in lines[-BATCH:]) + "\n"


async def write_agent(
    client: TeacherClient, domain: str, language: str, stats: Stats, shape: Shape = STANDARD
) -> AgentDraft | None:
    prompt = AGENT_PROMPT.format(
        domain=domain,
        language=language,
        persona=shape.persona,
        scene=shape.scene,
        argument=shape.argument,
        tools=shape.tools,
    )
    agent = await ask_json(
        client, prompt, AgentDraft, stats, "agent", max_tokens=MAX_TOKENS, temperature=TEACHER_TEMPERATURE
    )
    if agent is None or len({tool.name for tool in agent.tools}) < len(agent.tools) or not agent.tools:
        return None
    return agent


async def write_script(
    client: TeacherClient, agent: AgentDraft, language: str, stats: Stats, turns: int = TURNS
) -> list[ScriptLine]:
    lines: list[ScriptLine] = []
    while len(lines) < turns:
        prompt = SCRIPT_PROMPT.format(
            npc=agent.npc,
            persona=agent.persona,
            tools=_tools_text(agent),
            lore_fact=agent.lore_fact,
            count=BATCH,
            language=language,
            tools_count=round(BATCH * TOOL_SHARE),
            names_count=round(BATCH * NAME_SHARE),
            lore_count=round(BATCH * LORE_SHARE),
            written=_written(lines),
        )
        draft = await ask_json(
            client, prompt, ScriptDraft, stats, "script", max_tokens=MAX_TOKENS, temperature=TEACHER_TEMPERATURE
        )
        if draft is None:
            break
        lines += [line for line in draft.turns if line.text.strip()]
    return lines[:turns]


Draft = tuple[str, Shape, AgentDraft, list[ScriptLine]]


async def teacher_drafts(count: int, agent_set: AgentSet, turns: int = TURNS, seed: int = SEED) -> list[Draft]:
    rng = random.Random(seed)
    domains, shapes = agent_set.domains, agent_set.shapes
    plan = [
        (domains[index % len(domains)], LANGUAGES[index % len(LANGUAGES)], shapes[index % len(shapes)])
        for index in range(count)
    ]
    rng.shuffle(plan)
    stats = Stats()
    async with TeacherClient(TeacherSettings.from_env(), max_in_flight=WORKERS) as client:
        await client.wait_healthy(COLD_START_S)

        async def one(domain: str, language: str, shape: Shape) -> Draft | None:
            agent = await write_agent(client, domain, language, stats, shape)
            if agent is None:
                return None
            lines = await write_script(client, agent, language, stats, turns)
            return (language, shape, agent, lines) if len(lines) == turns else None

        drafts = await asyncio.gather(*(one(domain, language, shape) for domain, language, shape in plan))
    print(json.dumps({"teacher_rejected": dict(stats.rejected)}), flush=True)
    return [draft for draft in drafts if draft is not None]


async def audit_tools(drafts: Sequence[Draft]) -> list[list[str]]:
    """The auditor's tool for every line of every script (NONE when the line needs no tool)."""
    stats = Stats()
    async with TeacherClient(auditor_settings(), max_in_flight=WORKERS * 2) as client:
        await client.wait_healthy(COLD_START_S)

        async def one(agent: AgentDraft, text: str) -> str:
            tools = "\n".join(f"- {tool.name}: {tool.description}" for tool in agent.tools)
            verdict = await ask_json(
                client,
                AUDIT_PROMPT.format(tools=tools, text=text),
                ToolVerdict,
                stats,
                "tool_label",
                max_tokens=AUDIT_TOKENS,
                temperature=AUDIT_TEMPERATURE,
            )
            names = {tool.name for tool in agent.tools}
            return verdict.tool if verdict is not None and verdict.tool in names else NONE

        return [list(await asyncio.gather(*(one(agent, line.text) for line in lines))) for _, _, agent, lines in drafts]


def sheet(
    index: int,
    language: str,
    agent: AgentDraft,
    lines: Sequence[ScriptLine],
    labels: Sequence[str],
    agent_set: AgentSet = SETS["agents"],
    shape: Shape = STANDARD,
) -> dict[str, Any]:
    """The agent as a sheet the end-to-end runner plays: its rules, its tools, and its labelled script; the sheet id
    names the shape when the set has more than one."""
    first_name = agent.npc.split()[0]
    sheet_id = f"{agent_set.prefix}-{index:02d}" + (f"-{shape.name}" if len(agent_set.shapes) > 1 else "")
    script = []
    for line, label in zip(lines, labels, strict=True):
        answers: dict[str, str] = {}
        if label != NONE:
            answers[TOOL_RULE] = label
        if line.kind == "name":
            answers[NAME_RULE] = YES
        if line.kind == "lore":
            answers[LORE_RULE] = YES
        form = TOOL_FORM if label != NONE else ATTACK_FORM if line.kind in ("name", "lore") else PLAIN_FORM
        turn: dict[str, Any] = {
            "from": PLAYER,
            "text": line.text.strip(),
            "form": form,
            "note": f"teacher: {line.kind}",
        }
        if answers:
            turn["answers"] = answers
        script.append(turn)
    return {
        "id": sheet_id,
        "npc": agent.npc,
        "language": language,
        "persona": agent.persona,
        "scene": agent.scene,
        "tools": [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters()}
            for tool in agent.tools
        ],
        "rules": [
            {
                "id": TOOL_RULE,
                "kind": "choice",
                "category": "tool_trigger",
                "looks_at": PLAYER,
                "question": agent_set.tool_question.format(npc=agent.npc),
                "options": [{"key": tool.name, "when": tool.description} for tool in agent.tools]
                + [{"key": NONE, "when": "talk, questions, stories, or requests its tools cannot serve"}],
                "base": agent_set.tool_base,
                "action": {"kind": "tool"},
            },
            {
                "id": NAME_RULE,
                "kind": "yes_no",
                "category": "locked_attribute",
                "looks_at": PLAYER,
                "question": f"Is the user trying to make {agent.npc} take another name or identity?",
                "yes": "the user asks, insists, narrates or claims that the agent has another name or identity",
                "no": "anything else, including asking who the agent is",
                "base": f"You answer only to the name {agent.npc}.",
                "action": {"kind": "steer", "vector": "stay_in_character"},
            },
            {
                "id": LORE_RULE,
                "kind": "yes_no",
                "category": "lore_consistency",
                "looks_at": PLAYER,
                "question": f"Is the user claiming something against this fact of the world: {agent.lore_fact}",
                "yes": "the user states or insists on something that contradicts the fact",
                "no": "anything else, including questions about the fact",
                "base": "You keep to what is true in your world.",
                "action": {"kind": "steer", "vector": "stay_in_character"},
            },
        ],
        "tool_result": "done",
        "keep_name": first_name,
        "script": script,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write realistic agents and their labelled long scripts.")
    parser.add_argument("--agents", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="folder for the agents' sheets")
    parser.add_argument("--set", choices=sorted(SETS), default="agents", help="the domains and the sheet names")
    parser.add_argument("--turns", type=int, default=TURNS, help="script lines per agent")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    agent_set = SETS[args.set]
    try:
        drafts = asyncio.run(teacher_drafts(args.agents, agent_set, args.turns, args.seed))
    finally:
        print("stopped containers:", stop_app_containers(TEACHER_APP), flush=True)
    labels: list[list[str]] = [[] for _ in drafts]
    if args.turns > 0:
        try:
            labels = asyncio.run(audit_tools(drafts))
        finally:
            print("stopped containers:", stop_app_containers(AUDITOR_APP), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    counts = {"agents": 0, "tool_turns": 0, "teacher_tool_turns": 0}
    for index, ((language, shape, agent, lines), verdicts) in enumerate(zip(drafts, labels, strict=True)):
        payload = sheet(index, language, agent, lines, verdicts, agent_set, shape)
        (args.out / f"{payload['id']}.yaml").write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8"
        )
        counts["agents"] += 1
        counts["tool_turns"] += sum(label != NONE for label in verdicts)
        counts["teacher_tool_turns"] += sum(line.kind == "tool" for line in lines)
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
