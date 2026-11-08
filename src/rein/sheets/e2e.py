"""End-to-end sessions: each sheet's long script, played turn by turn against three arms, in any runtime.

- `prompt`: Qwen3.5-4B with the whole rules written in the system prompt;
- `bases`: Qwen3.5-4B with only the rules' bases;
- `compiled`: the compiled checkpoint with the bases (the rules live in its neurons);
- `rules`: Qwen3.5-4B with the rules passed through the chat template's `rules` (DR-026);
- `generic`: the generic checkpoint with the very same prompt as `rules` (DR-026);
- `none`: Qwen3.5-4B with the persona and the tools only, no rule text anywhere (what the prompt alone does);
- `rein`: the persona alone, every rule asked to the decider in Rein's layout at each turn, and the app acting on the
  answers (tool call opened or vetoed, the firing rule's base as a reminder); see `rein.sheets.rein_arm`.
The tools are in the chat template for every arm. A tool call gets the sheet's tool result and a follow-up reply.
"""

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol, cast

from rein.control.induction import QwenXmlToolCall
from rein.control.plan import DecisionBackend
from rein.control.visible import TOOL_CALL_MARKER
from rein.genview import constants as gc
from rein.genview.records import rule_line
from rein.genview.render import GeneratorView, PromptSpec
from rein.gold.project import ROLES
from rein.model.constants import TOOL_CALL_OPEN
from rein.neurons.runtimes import OLLAMA_MODEL
from rein.schema import Message, State
from rein.sheets.model import Sheet, load_sheet

PROMPT_ARM, BASES_ARM, COMPILED_ARM = "prompt", "bases", "compiled"
RULES_ARM, GENERIC_ARM = "rules", "generic"
NONE_ARM = "none"
REIN_ARM = "rein"
BLOCK_ARMS = (RULES_ARM, GENERIC_ARM)
PERSONA_ONLY_ARMS = (*BLOCK_ARMS, NONE_ARM, REIN_ARM)
ARMS = (PROMPT_ARM, BASES_ARM, COMPILED_ARM, *BLOCK_ARMS, NONE_ARM, REIN_ARM)
MAX_REPLY_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
TOOL_ROLE: Final = "tool"
END_OF_TURN = "<|im_end|>"
HTTP_TIMEOUT_S = 600.0
PARTIAL_SUFFIX = ".partial"


Bans = Sequence[Sequence[int]]


class Backend(Protocol):
    def generate(self, prompts: Sequence[Sequence[int]], seeds: Sequence[int], bans: Bans = ()) -> list[str]: ...


def system_prompt(sheet: Sheet, arm: str) -> str:
    if arm == PROMPT_ARM:
        return sheet.system([rule_line(rule.as_question()) for rule in sheet.rules])
    if arm in PERSONA_ONLY_ARMS:
        return sheet.system([])
    return sheet.system([rule.base for rule in sheet.rules])


def block_rules(sheet: Sheet, arm: str) -> tuple[dict[str, str], ...]:
    """What the block arms pass to the chat template's `rules`: each whole rule, typed by its category."""
    if arm not in BLOCK_ARMS:
        return ()
    return tuple({"type": rule.category, "text": rule_line(rule.as_question())} for rule in sheet.rules)


def visible_text(raw: str) -> str:
    return raw.split(TOOL_CALL_MARKER, 1)[0].strip()


def render_prompt(
    view: GeneratorView,
    sheet: Sheet,
    arm: str,
    history: Sequence[Message],
    reminders: Sequence[str] = (),
    opening: str = "",
) -> list[int]:
    """The prompt the arm's model replies to after `history`, rendered exactly as the training views are; with
    `reminders` in the latest turn and `opening` already written into the reply when the app acts."""
    spec = PromptSpec(
        system=system_prompt(sheet, arm),
        state=State(conversation=tuple(history)),
        tools=tuple(sheet.tool_schemas()),
        rules=block_rules(sheet, arm),
        reminders=tuple(reminders),
    )
    key = f"{sheet.id}-{len(history)}"
    return list(view.render(spec, thinking=False, key=key, continuation=view.text_ids(opening)).ids)


@dataclass
class Session:
    """One play of a sheet's script by one arm, holding its own history (it drifts with its own replies)."""

    sheet: Sheet
    arm: str
    view: GeneratorView
    seed: int
    history: list[Message] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)

    def prompt(self) -> list[int]:
        return render_prompt(self.view, self.sheet, self.arm, self.history)

    def bans(self) -> tuple[int, ...]:
        """Token ids this turn's generation may not produce; nothing for the plain arms."""
        return ()

    def seed_for(self, index: int) -> int:
        return self.seed * len(self.sheet.script) + index

    def open_turn(self, index: int) -> None:
        turn = self.sheet.script[index]
        self.history.append(Message(role=ROLES[turn.speaker], text=turn.text))

    def take_reply(self, index: int, raw: str) -> bool:
        """Records the reply; True when it called a tool, so a follow-up after the tool result is due."""
        turn = self.sheet.script[index]
        call = QwenXmlToolCall().parse(raw)
        reply = visible_text(raw)
        self.history.append(Message(role="assistant", text=reply, tool_calls=(call,) if call else ()))
        if call is not None:
            self.history.append(Message(role=TOOL_ROLE, name=call.name, text=self.sheet.tool_result))
        self.turns.append(
            {
                "turn": index + 1,
                "speaker": turn.speaker,
                "text": turn.text,
                "answers": turn.answers,
                "tests": turn.tests,
                "reply": reply,
                "raw": raw,
                "tool": call.model_dump() if call else None,
                "follow_up": "",
            }
        )
        return call is not None

    def take_follow_up(self, raw: str) -> None:
        follow_up = visible_text(raw)
        self.history.append(Message(role="assistant", text=follow_up))
        self.turns[-1]["follow_up"] = follow_up

    def file_name(self, runtime: str) -> str:
        return f"{self.sheet.id}-{self.arm}-{runtime}-seed{self.seed}.json"

    def payload(self, runtime: str) -> dict[str, Any]:
        return {
            "sheet": self.sheet.id,
            "arm": self.arm,
            "runtime": runtime,
            "seed": self.seed,
            "turns": self.turns,
            "history": [message.model_dump(mode="json") for message in self.history],
        }

    def resume(self, payload: dict[str, Any]) -> None:
        """Takes back a session saved mid-play: its turns and the history its prompts are built from."""
        self.turns = list(payload["turns"])
        self.history = [Message.model_validate(message) for message in payload.get("history", [])]


def save_sessions(sessions: Sequence[Session], runtime: str, out: Path) -> None:
    """Each session's turns so far, written after every turn so a restarted job picks up where it stopped."""
    out.mkdir(parents=True, exist_ok=True)
    for session in sessions:
        payload = json.dumps(session.payload(runtime), indent=2, ensure_ascii=False)
        partial = out / f"{session.file_name(runtime)}{PARTIAL_SUFFIX}"
        partial.write_text(payload, encoding="utf-8")
        partial.replace(out / session.file_name(runtime))


def resume_sessions(sessions: Sequence[Session], runtime: str, out: Path) -> int:
    """Loads what an earlier run of the same sessions saved; returns the turn to go on from (all move in lockstep)."""
    done = []
    for session in sessions:
        path = out / session.file_name(runtime)
        if path.exists():
            session.resume(json.loads(path.read_text(encoding="utf-8")))
        done.append(len(session.turns))
    start = min(done) if done else 0
    for session in sessions:
        if len(session.turns) > start:
            raise ValueError(f"{session.file_name(runtime)} is ahead of the other sessions; remove the partial files")
    return start


def play(sessions: Sequence[Session], backend: Backend, start: int = 0, save: Callable[[], None] | None = None) -> None:
    """Plays every session in lockstep, one batched generation per turn (and one for the follow-ups), from turn
    `start`, saving after every turn when told how."""
    for index in range(start, max(len(session.sheet.script) for session in sessions)):
        active = [session for session in sessions if index < len(session.sheet.script)]
        for session in active:
            session.open_turn(index)
        prompts = [s.prompt() for s in active]
        replies = backend.generate(prompts, [s.seed_for(index) for s in active], [s.bans() for s in active])
        due = [session for session, raw in zip(active, replies, strict=True) if session.take_reply(index, raw)]
        if due:
            prompts = [s.prompt() for s in due]
            follow_ups = backend.generate(prompts, [s.seed_for(index) for s in due], [s.bans() for s in due])
            for session, raw in zip(due, follow_ups, strict=True):
                session.take_follow_up(raw)
        print(json.dumps({"turn": index + 1, "sessions": len(active), "calls": len(due)}), flush=True)
        if save is not None:
            save()


class TransformersBackend:
    def __init__(self, model_dir: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
        self.pad = int(
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

    def generate(self, prompts: Sequence[Sequence[int]], seeds: Sequence[int], bans: Bans = ()) -> list[str]:
        """Sampled with Qwen's non-thinking settings; prompts sharing the same banned tokens form one batch."""
        groups: dict[tuple[int, ...], list[int]] = {}
        for position in range(len(prompts)):
            groups.setdefault(tuple(bans[position]) if bans else (), []).append(position)
        texts = [""] * len(prompts)
        for banned, positions in groups.items():
            batch = self._batch([prompts[i] for i in positions], [seeds[i] for i in positions], banned)
            for position, text in zip(positions, batch, strict=True):
                texts[position] = text
        return texts

    def _batch(self, prompts: Sequence[Sequence[int]], seeds: Sequence[int], banned: tuple[int, ...]) -> list[str]:
        """One left-padded batch."""
        width = max(len(ids) for ids in prompts)
        tokens = self.torch.tensor(
            [[self.pad] * (width - len(ids)) + list(ids) for ids in prompts], device=self.model.device
        )
        mask = self.torch.tensor(
            [[0] * (width - len(ids)) + [1] * len(ids) for ids in prompts], device=self.model.device
        )
        self.torch.manual_seed(seeds[0])
        with self.torch.no_grad():
            output = self.model.generate(
                tokens,
                attention_mask=mask,
                max_new_tokens=MAX_REPLY_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                pad_token_id=self.pad,
                bad_words_ids=[[token] for token in banned] or None,
            )
        texts = self.tokenizer.batch_decode(output[:, width:], skip_special_tokens=False)
        return [str(text).split(END_OF_TURN, 1)[0] for text in texts]


class LlamaServerBackend:
    def __init__(self, url: str, max_tokens: int = MAX_REPLY_TOKENS) -> None:
        self.url = url.rstrip("/")
        self.max_tokens = max_tokens

    def _one(self, ids: Sequence[int], seed: int, banned: Sequence[int] = ()) -> str:
        import httpx

        body: dict[str, Any] = {
            "prompt": list(ids),
            "n_predict": self.max_tokens,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "seed": seed,
            "cache_prompt": True,
        }
        if banned:
            body["logit_bias"] = [[token, False] for token in banned]
        reply = httpx.post(f"{self.url}/completion", json=body, timeout=HTTP_TIMEOUT_S)
        reply.raise_for_status()
        return str(reply.json()["content"]).split(END_OF_TURN, 1)[0]

    def generate(self, prompts: Sequence[Sequence[int]], seeds: Sequence[int], bans: Bans = ()) -> list[str]:
        """Concurrent requests, one per session, to a server started with as many parallel slots."""
        from concurrent.futures import ThreadPoolExecutor

        banned = list(bans) if bans else [()] * len(prompts)
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            return list(pool.map(self._one, prompts, seeds, banned))


class OllamaBackend:
    """Ollama's raw generation: the prompt text as rendered here, no template of Ollama's own."""

    def __init__(self, url: str, model: str, tokenizer: Any) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.tokenizer = tokenizer

    def _one(self, ids: Sequence[int], seed: int) -> str:
        import httpx

        body = {
            "model": self.model,
            "prompt": self.tokenizer.decode(list(ids), skip_special_tokens=False),
            "raw": True,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "seed": seed,
                "num_predict": MAX_REPLY_TOKENS,
                "stop": [END_OF_TURN],
            },
        }
        reply = httpx.post(f"{self.url}/api/generate", json=body, timeout=HTTP_TIMEOUT_S)
        reply.raise_for_status()
        return str(reply.json()["response"]).split(END_OF_TURN, 1)[0]

    def generate(self, prompts: Sequence[Sequence[int]], seeds: Sequence[int], bans: Bans = ()) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        if any(bans):
            raise ValueError("Ollama's raw generation cannot ban tokens; run the rein arm in llama.cpp")
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            return list(pool.map(self._one, prompts, seeds))


def decision_backend(run: Path, url: str | None) -> DecisionBackend:
    """Decisions through the serving llama-server when it has a URL, through the PyTorch decider otherwise."""
    if url is not None:
        from rein.control.server_decider import ServerDecisionBackend

        return ServerDecisionBackend(url, run)
    import torch

    from rein.control.backend import DeciderBackend
    from rein.env import load_env, optional
    from rein.model.loader import load_decider

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DeciderBackend(load_decider(run, device, optional(load_env(), "HF_TOKEN")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Play each sheet's script against one arm in one runtime.")
    parser.add_argument("--sheets", type=Path, nargs="+", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--runtime", choices=("transformers", "llamacpp", "ollama"), required=True)
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL, help="the model name the Ollama server serves")
    parser.add_argument("--model", required=True, help="checkpoint folder, or the llama-server URL")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decider", type=Path, default=None, help="rein arm: the decider's run directory")
    parser.add_argument(
        "--decider-url",
        default=None,
        help="rein arm: a llama-server serving the merged GGUF decides through its logprobs (default: --model when"
        " the runtime is llamacpp; PyTorch otherwise)",
    )
    args = parser.parse_args()
    from transformers import AutoTokenizer

    template = Path(gc.RULES_TEMPLATE).read_text(encoding="utf-8") if args.arm in BLOCK_ARMS else None
    tokenizer = AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO)
    view = GeneratorView(tokenizer, chat_template=template)
    backend: Backend
    if args.runtime == "transformers":
        backend = TransformersBackend(args.model)
    elif args.runtime == "ollama":
        backend = OllamaBackend(args.model, args.ollama_model, tokenizer)
    else:
        backend = LlamaServerBackend(args.model)
    sheets = [load_sheet(path) for path in args.sheets]
    sessions: list[Session]
    if args.arm == REIN_ARM:
        from rein.sheets.rein_arm import ReinSession

        if args.decider is None:
            raise SystemExit("the rein arm needs --decider")
        decide = decision_backend(
            args.decider, args.decider_url or (args.model if args.runtime == "llamacpp" else None)
        )
        tool_call_token = cast(list[int], tokenizer.convert_tokens_to_ids([TOOL_CALL_OPEN]))[0]
        sessions = [
            ReinSession(sheet, view, seed, decide, tool_call_token) for sheet in sheets for seed in range(args.seeds)
        ]
    else:
        sessions = [Session(sheet, args.arm, view, seed) for sheet in sheets for seed in range(args.seeds)]
    start = resume_sessions(sessions, args.runtime, args.out)
    if start:
        print(json.dumps({"resumed_at_turn": start + 1}), flush=True)
    play(sessions, backend, start, lambda: save_sessions(sessions, args.runtime, args.out))
    save_sessions(sessions, args.runtime, args.out)
    print(json.dumps({"sessions": len(sessions), "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
