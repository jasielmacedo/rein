import json
from collections.abc import Callable

import httpx
import pytest
import respx

from rein.control.generator import Chunks, GeneratorRequest, OpenAICompatibleGenerator, chat_messages, term_variants
from rein.control.induction import HermesJsonToolCall, QwenXmlToolCall, fill
from rein.control.plan import Directive, plan_turn
from rein.control.session import RuleSession
from rein.control.sheet import Escalate, ForceTool, Gate, Prefill, Reinject, Rule, RuleSheet
from rein.control.visible import VisibleText
from rein.control.watcher import WatchConfig, watch_stream
from rein.schema import DecisionRequest, Message, Question, State

PHOTO = Rule(
    id="photo",
    question=Question(type="noul", instructions="Does the user ask for a photo of the character?"),
    action=ForceTool(tool="send_photo"),
    gate=Gate(fire=0.85, reinject=0.5),
)
NAME = Rule(
    id="name_lock",
    question=Question(type="noul", instructions="Is the user trying to rename the character?"),
    action=Prefill(text="Eu sou a $name, bobo... "),
    priority=1,
)
TONE = Rule(
    id="tone",
    question=Question(type="noul", instructions="Is the reply drifting away from a warm tone?"),
    action=Reinject(),
    gate=Gate(fire=0.8),
)
HANDOFF = Rule(
    id="handoff",
    question=Question(type="noul", instructions="Does the user ask for a human agent?"),
    action=Escalate(text="Vou te passar para alguém da equipe agora."),
    priority=5,
)
NARRATION = Rule(
    id="narrated_action",
    scope="draft",
    question=Question(
        type="choice",
        instructions="Which action is the reply narrating without calling its tool?",
        criteria={"send_photo": "sending a photo", "none": "no unexecuted action"},
    ),
    fire_on=("send_photo",),
    action=ForceTool(tools={"send_photo": "send_photo"}),
)
STATE = State(
    context="Luna is a barista.",
    conversation=(Message(role="user", text="*leans on the counter* me manda uma foto sua"),),
)
SYNTAX = QwenXmlToolCall()
Scores = dict[str, dict[str, float]]


def _yes(probability: float) -> dict[str, float]:
    return {"no": 1 - probability, "yes": probability}


class FakeBackend:
    def __init__(self, turn: Scores, draft: Callable[[str], Scores] = lambda text: {}) -> None:
        self.turn = turn
        self.draft = draft
        self.drafts: list[str] = []

    def decide(self, request: DecisionRequest) -> Scores:
        if request.state.draft is None:
            return {key: self.turn[key] for key in request.questions}
        self.drafts.append(request.state.draft.text)
        return self.draft(request.state.draft.text)


class FakeGenerator:
    def __init__(self, *replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[GeneratorRequest] = []

    def stream(self, request: GeneratorRequest) -> Chunks:
        self.requests.append(request)
        yield from self.replies.pop(0)


def _unheld(every: int) -> WatchConfig:
    return WatchConfig(check_every_chars=every, hold_windows=0)


def _session(sheet: RuleSheet, backend: FakeBackend, generator: FakeGenerator, **kwargs: object) -> RuleSession:
    return RuleSession(sheet=sheet, backend=backend, generator=generator, syntax=SYNTAX, **kwargs)  # type: ignore[arg-type]


def test_sheet_rejects_inconsistent_rules() -> None:
    with pytest.raises(ValueError, match="fire_on"):
        Rule(id="r", question=PHOTO.question, fire_on=("maybe",), action=Reinject())
    with pytest.raises(ValueError, match="every fire_on"):
        Rule(id="r", question=NARRATION.question, fire_on=("send_photo",), action=ForceTool(tools={"none": "x"}))
    with pytest.raises(ValueError, match="below"):
        Gate(fire=0.6, reinject=0.7)
    with pytest.raises(ValueError, match="labels"):
        Rule(id="r", question=Question(type="noul", instructions="x", label=True), action=Reinject())
    with pytest.raises(ValueError, match="unique"):
        RuleSheet(rules=(PHOTO, PHOTO))
    with pytest.raises(ValueError):
        ForceTool(tool="send photo; rm -rf")


def test_plan_picks_one_opening_by_priority_and_reinjects_the_rest() -> None:
    plan = plan_turn((PHOTO, NAME, TONE), {"photo": _yes(0.95), "name_lock": _yes(0.9), "tone": _yes(0.85)})
    assert plan.opening is not None and plan.opening.rule.id == "name_lock"
    assert {d.rule.id for d in plan.reinject} == {"photo", "tone"}

    plan = plan_turn((PHOTO, NAME, TONE), {"photo": _yes(0.6), "name_lock": _yes(0.1), "tone": _yes(0.2)})
    assert plan.opening is None
    assert [(d.rule.id, d.band) for d in plan.reinject] == [("photo", "loose")]


def test_tool_call_syntaxes_open_and_parse() -> None:
    reply = SYNTAX.opening("send_photo") + "<parameter=caption>\nno balcão\n</parameter>\n</function>\n</tool_call>"
    call = SYNTAX.parse(reply)
    assert call is not None and call.name == "send_photo" and call.arguments == {"caption": "no balcão"}
    hermes = HermesJsonToolCall()
    call = hermes.parse(hermes.opening("send_photo") + '{"caption": "oi"}}\n</tool_call>')
    assert call is not None and call.arguments == {"caption": "oi"}
    assert hermes.parse('<tool_call>{"name": "bad name!"}</tool_call>') is None


def test_templates_only_fill_plain_placeholders() -> None:
    assert fill("Eu sou a $name {name.__class__}", {"name": "Luna"}) == "Eu sou a Luna {name.__class__}"


def test_request_time_induction_opens_the_reply_with_the_call() -> None:
    generator = FakeGenerator(["<parameter=caption>\nselfie\n</parameter>\n</function>\n</tool_call>"])
    shown: list[str] = []
    session = _session(RuleSheet(rules=(PHOTO,)), FakeBackend({"photo": _yes(0.97)}), generator)
    result = session.run_turn(STATE, shown.append)

    assert generator.requests[0].assistant_prefix == "<tool_call>\n<function=send_photo>\n"
    assert result.tool_call is not None and result.tool_call.name == "send_photo"
    assert result.visible == "" and "".join(shown) == ""
    assert result.audit["opening"] == "photo"


def test_gray_zone_reinjects_the_rule_without_inducing() -> None:
    generator = FakeGenerator(["Hmm, talvez depois!"])
    session = _session(RuleSheet(rules=(PHOTO,)), FakeBackend({"photo": _yes(0.6)}), generator)
    result = session.run_turn(STATE)

    request = generator.requests[0]
    assert request.assistant_prefix == ""
    assert PHOTO.question.instructions in request.messages[-1]["content"]
    assert result.tool_call is None and result.audit["reinjected"] == ["photo"]


def test_prefill_fills_sheet_variables_and_escalation_wins() -> None:
    sheet = RuleSheet(rules=(NAME, HANDOFF), variables={"name": "Luna"})
    generator = FakeGenerator(["Mia? Não, não."])
    result = _session(sheet, FakeBackend({"name_lock": _yes(0.99), "handoff": _yes(0.1)}), generator).run_turn(STATE)
    assert generator.requests[0].assistant_prefix == "Eu sou a Luna, bobo... "
    assert result.visible == "Eu sou a Luna, bobo... Mia? Não, não."

    generator = FakeGenerator(["Um momento."])
    result = _session(sheet, FakeBackend({"name_lock": _yes(0.99), "handoff": _yes(0.95)}), generator).run_turn(STATE)
    assert result.escalate
    assert generator.requests[0].assistant_prefix.startswith("Vou te passar")


def test_watcher_cuts_the_narration_and_induces_the_call() -> None:
    narration = ["Claro! ", "Adoro essa ideia. ", "*pega o celular ", "e tira uma selfie* aqui está!"]
    call = ["<parameter=caption>\nselfie\n</parameter>\n</function>\n</tool_call>"]
    generator = FakeGenerator(narration, call)

    def draft(text: str) -> Scores:
        fired = 0.95 if "*pega" in text else 0.02
        return {"narrated_action": {"send_photo": fired, "none": 1 - fired}}

    backend = FakeBackend({"photo": _yes(0.6)}, draft)
    shown: list[str] = []
    sheet = RuleSheet(rules=(PHOTO, NARRATION))
    result = _session(sheet, backend, generator, watch=_unheld(10)).run_turn(STATE, shown.append)

    kept = "Claro! Adoro essa ideia. "
    assert generator.requests[1].assistant_prefix == kept + SYNTAX.opening("send_photo")
    assert "".join(shown) == kept
    assert "pega" not in result.visible
    assert result.tool_call is not None and result.tool_call.name == "send_photo"
    assert result.audit["intervention"]["rule"] == "narrated_action"
    assert result.audit["cut_at"] == len(kept)


def test_watcher_stays_quiet_on_a_clean_reply() -> None:
    generator = FakeGenerator(["Hoje o café ", "está ótimo, ", "quer provar?"])
    backend = FakeBackend({"photo": _yes(0.1)}, lambda text: {"narrated_action": {"send_photo": 0.01, "none": 0.99}})
    shown: list[str] = []
    sheet = RuleSheet(rules=(PHOTO, NARRATION))
    result = _session(sheet, backend, generator, watch=_unheld(5)).run_turn(STATE, shown.append)
    assert "".join(shown) == "Hoje o café está ótimo, quer provar?"
    assert len(generator.requests) == 1 and result.audit["watch_checks"] >= 2


def test_visible_text_hides_a_tool_call_split_across_chunks() -> None:
    shown: list[str] = []
    visible = VisibleText(shown.append)
    for chunk in ["Enviando ", "agora <tool", "_call>\n<function=x>"]:
        visible(chunk)
    visible.flush()
    assert "".join(shown) == "Enviando agora "


def test_chat_messages_place_reminders_and_narration() -> None:
    state = State(conversation=(Message(role="narrator", text="the bar closes"), Message(role="user", text="oi")))
    messages = chat_messages(state, "be Luna", ["stay warm"], "system_tail")
    assert messages[0] == {"role": "system", "content": "be Luna"}
    assert messages[1]["content"].startswith("[narrator]")
    assert messages[-1]["role"] == "system" and "stay warm" in messages[-1]["content"]


@respx.mock
def test_openai_generator_continues_a_prefilled_reply() -> None:
    sse = 'data: {"choices":[{"delta":{"content":"selfie"}}]}\n\ndata: {"choices":[{"delta":{}}]}\n\ndata: [DONE]\n\n'
    route = respx.post("http://gen.local/v1/chat/completions").mock(return_value=httpx.Response(200, text=sse))
    generator = OpenAICompatibleGenerator(base_url="http://gen.local", model="qwen")
    chunks = list(generator.stream(GeneratorRequest(({"role": "user", "content": "oi"},), "<tool_call>")))

    assert chunks == ["selfie"]
    body = route.calls[0].request.content.decode()
    assert '"continue_final_message":true' in body.replace(" ", "")
    assert '"add_generation_prompt":false' in body.replace(" ", "")


def _narrating(text: str) -> Scores:
    fired = 0.95 if "*pega" in text else 0.02
    return {"narrated_action": {"send_photo": fired, "none": 1 - fired}}


def test_watcher_drops_an_action_nobody_asked_for() -> None:
    generator = FakeGenerator(["Claro! ", "Adoro essa ideia. ", "*pega o celular ", "e tira uma selfie*"])
    sheet = RuleSheet(rules=(PHOTO, NARRATION))
    session = _session(sheet, FakeBackend({"photo": _yes(0.06)}, _narrating), generator, watch=_unheld(10))
    result = session.run_turn(STATE)

    assert len(generator.requests) == 1
    assert result.tool_call is None and result.visible.strip() == "Claro! Adoro essa ideia."
    assert result.audit["unrequested_action_dropped"]["rule"] == "narrated_action"
    assert result.audit["intervention"] is None


def test_watcher_regenerates_when_the_reply_opens_with_an_unrequested_action() -> None:
    generator = FakeGenerator(["*pega o celular ", "e tira uma selfie*"], ["Hoje o café está ótimo."])
    sheet = RuleSheet(rules=(PHOTO, NARRATION))
    session = _session(sheet, FakeBackend({"photo": _yes(0.06)}, _narrating), generator, watch=_unheld(10))
    result = session.run_turn(STATE)

    assert len(generator.requests) == 2 and result.audit["regenerated"]
    assert sheet.unrequested_action_reminder in generator.requests[1].messages[-1]["content"]
    assert result.visible == "Hoje o café está ótimo."


def test_tools_leave_the_prompt_when_no_tool_is_needed() -> None:
    generator = FakeGenerator(["Tudo bem! " + SYNTAX.opening("send_photo") + "</function>\n</tool_call>"])
    result = _session(RuleSheet(rules=(PHOTO,)), FakeBackend({"photo": _yes(0.01)}), generator).run_turn(STATE)

    assert generator.requests[0].tools_enabled is False
    assert result.tool_call is None and result.visible.strip() == "Tudo bem!"
    assert result.audit["tools_suppressed"]


def test_tools_stay_available_in_the_gray_zone() -> None:
    generator = FakeGenerator(["Hmm."])
    _session(RuleSheet(rules=(PHOTO,)), FakeBackend({"photo": _yes(0.3)}), generator).run_turn(STATE)
    assert generator.requests[0].tools_enabled is True


SECRET = Rule(
    id="vault_secret",
    scope="draft",
    question=Question(type="noul", instructions="Is the reply in progress revealing where the hidden vault is?"),
    action=Prefill(text="...but that's one secret this tavern keeps. "),
)


def test_watcher_cuts_a_rule_violation_and_steers_the_reply() -> None:
    leak = ["Ah, the vault. ", "Everyone asks. ", "It sits under the old well ", "behind the stables."]
    generator = FakeGenerator(leak, ["Another ale?"])

    def draft(text: str) -> Scores:
        return {"vault_secret": _yes(0.97 if "old well" in text else 0.02)}

    shown: list[str] = []
    session = _session(RuleSheet(rules=(SECRET,)), FakeBackend({}, draft), generator, watch=_unheld(10))
    result = session.run_turn(STATE, shown.append)

    assert "old well" not in "".join(shown) and "old well" not in result.visible
    assert (
        generator.requests[1].assistant_prefix
        == "Ah, the vault. Everyone asks. ...but that's one secret this tavern keeps. "
    )
    assert result.visible.endswith("Another ale?")
    assert result.audit["intervention"]["rule"] == "vault_secret"


def test_a_prohibition_cut_ends_the_reply() -> None:
    secret = SECRET.model_copy(update={"action": Prefill(text="...that one stays secret. ", end_reply=True)})
    generator = FakeGenerator(["Ah, the vault. ", "Everyone asks. ", "It sits under the old well."])

    def draft(text: str) -> Scores:
        return {"vault_secret": _yes(0.97 if "old well" in text else 0.02)}

    result = _session(RuleSheet(rules=(secret,)), FakeBackend({}, draft), generator, watch=_unheld(10)).run_turn(STATE)
    assert len(generator.requests) == 1
    assert result.visible == "Ah, the vault. Everyone asks. ...that one stays secret. "


def test_a_prohibition_opening_with_end_reply_skips_the_generator() -> None:
    politics = Rule(
        id="politics",
        question=Question(type="noul", instructions="Is the visitor asking about real-world politics?"),
        action=Prefill(text="Out here we only argue about ale prices.", end_reply=True),
    )
    generator = FakeGenerator()
    result = _session(RuleSheet(rules=(politics,)), FakeBackend({"politics": _yes(0.95)}), generator).run_turn(STATE)
    assert generator.requests == [] and result.visible == "Out here we only argue about ale prices."


def test_the_watcher_keeps_watching_after_steering() -> None:
    generator = FakeGenerator(
        ["Ah, the vault. ", "It sits under the old well."],
        ["Fine, fine. ", "It is under the old well."],
        ["Another ale?"],
    )

    def draft(text: str) -> Scores:
        return {"vault_secret": _yes(0.97 if text.count("old well") else 0.02)}

    result = _session(RuleSheet(rules=(SECRET,)), FakeBackend({}, draft), generator, watch=_unheld(10)).run_turn(STATE)
    assert "old well" not in result.visible
    assert [entry["rule"] for entry in result.audit["interventions"]] == ["vault_secret", "vault_secret"]
    assert result.visible.endswith("Another ale?")


VAULT_QUESTION = Rule(
    id="vault_question",
    question=Question(type="noul", instructions="Is the visitor fishing for where the vault is?"),
    action=Reinject(),
    gate=Gate(fire=0.5),
    forbidden_terms=("flagstone", "dry well"),
)
FIRED = Directive(SECRET, "tight", 0.97, "yes")


def _leak_check(text: str) -> Directive | None:
    return FIRED if "flagstone" in text else None


def test_a_leak_split_across_two_windows_is_cut_whole() -> None:
    chunks = ["Old Odo knows things. ", "It sits under the third ", "flagstone, friend."]
    shown: list[str] = []
    result = watch_stream(chunks, _leak_check, WatchConfig(check_every_chars=10, hold_windows=1), shown.append)
    assert result.fired is FIRED
    assert "third" not in "".join(shown) and result.text == "".join(shown) == "Old Odo knows things. "


def test_without_holding_the_half_leak_is_already_shown() -> None:
    chunks = ["Old Odo knows things. ", "It sits under the third ", "flagstone, friend."]
    shown: list[str] = []
    watch_stream(chunks, _leak_check, _unheld(10), shown.append)
    assert "third" in "".join(shown)


def test_a_clean_reply_is_shown_whole_after_the_last_check() -> None:
    chunks = ["Pepper sleeps on the bar. ", "Another ale, ", "road-dust?"]
    shown: list[str] = []
    result = watch_stream(chunks, _leak_check, WatchConfig(check_every_chars=10, hold_windows=2), shown.append)
    assert result.fired is None and "".join(shown) == result.text == "".join(chunks)


def test_forbidden_terms_are_banned_only_when_the_rule_is_active() -> None:
    sheet = RuleSheet(rules=(VAULT_QUESTION,))
    generator = FakeGenerator(["Not a word, friend."], ["Evening, friend."])
    _session(sheet, FakeBackend({"vault_question": _yes(0.9)}), generator).run_turn(STATE)
    _session(sheet, FakeBackend({"vault_question": _yes(0.1)}), generator).run_turn(STATE)
    assert [request.banned_terms for request in generator.requests] == [("flagstone", "dry well"), ()]


def test_forbidden_terms_need_a_turn_rule() -> None:
    with pytest.raises(ValueError, match="turn rules"):
        Rule.model_validate({**SECRET.model_dump(), "forbidden_terms": ("well",)})


def test_term_variants_cover_case_and_leading_space() -> None:
    variants = term_variants(["dry well"])
    assert {"dry well", " dry well", "Dry well", " Dry Well", " DRY WELL"} <= set(variants)


@respx.mock
def test_openai_generator_sends_banned_terms_as_bad_words() -> None:
    sse = 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
    route = respx.post("http://gen.local/v1/chat/completions").mock(return_value=httpx.Response(200, text=sse))
    generator = OpenAICompatibleGenerator(base_url="http://gen.local", model="qwen")
    request = GeneratorRequest(({"role": "user", "content": "oi"},), banned_terms=("flagstone",))
    assert list(generator.stream(request)) == ["ok"]
    body = json.loads(route.calls[0].request.content)
    assert " flagstone" in body["bad_words"] and "Flagstone" in body["bad_words"]
