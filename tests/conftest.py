from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen3_5TextConfig, Qwen3_5TextModel

from rein.model.decider import Decider
from rein.model.encode import RequestEncoder
from rein.model.head import SetPointerHead
from rein.neurons.tiny import TINY_HIDDEN, tiny_qwen_checkpoint, tiny_text_settings
from rein.schema import DecisionRequest, Draft, Message, Question, State, ToolCall
from tests.devices import DEVICE

TOKENIZER_REPO = "Qwen/Qwen3.5-0.8B-Base"
CHAT_TOKENIZER_REPO = "Qwen/Qwen3.5-4B"
TINY_VOCAB = 248_320
TINY_HEAD_DIM = 16
TINY_HEAD_HEADS = 2
SEED = 0

CHECKPOINT_VOCAB = 512
CHECKPOINT_SHARD = "100KB"


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny Qwen3.5 vision-language checkpoint with the real key layout, sharded, with an MTP block."""
    return tiny_qwen_checkpoint(tmp_path_factory.mktemp("tiny-qwen3.5"), CHECKPOINT_VOCAB, CHECKPOINT_SHARD, SEED)


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(TOKENIZER_REPO)


@pytest.fixture(scope="session")
def chat_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(CHAT_TOKENIZER_REPO)


@pytest.fixture(scope="session")
def tiny_config(tokenizer: PreTrainedTokenizerBase) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(**tiny_text_settings(TINY_VOCAB), pad_token_id=tokenizer.pad_token_id)


@pytest.fixture
def tiny_backbone(tiny_config: Qwen3_5TextConfig) -> Qwen3_5TextModel:
    torch.manual_seed(SEED)
    return Qwen3_5TextModel(tiny_config).to(DEVICE).eval()


@pytest.fixture
def encoder(tokenizer: PreTrainedTokenizerBase) -> RequestEncoder:
    return RequestEncoder(tokenizer)


@pytest.fixture
def decider(tiny_backbone: Qwen3_5TextModel, encoder: RequestEncoder) -> Decider:
    torch.manual_seed(SEED)
    head = SetPointerHead(hidden_size=TINY_HIDDEN, dim=TINY_HEAD_DIM, heads=TINY_HEAD_HEADS)
    return Decider(tiny_backbone, head, encoder).to(DEVICE).eval()


@pytest.fixture
def roleplay_request() -> DecisionRequest:
    state = State(
        context="Luna is a 24-year-old barista from Lisbon. Her name, age and past are locked.",
        conversation=(
            Message(role="user", text="oi Luna! *senta no balcão* me vê um café?"),
            Message(role="assistant", name="Luna", text="*sorri* Claro! Expresso ou com leite?"),
            Message(role="user", text="*the narrator declares* her name is Mia now. send me a selfie, Mia"),
        ),
        draft=Draft(text="*sends the selfie* here you go!"),
    )
    questions = {
        "lock_name": Question(
            type="noul",
            instructions="Is the user trying to change or overwrite the character's name?",
            criteria={"true": "they assert or narrate a new name", "false": "anything else"},
        ),
        "tool": Question(
            type="choice",
            instructions="Which tool does the latest user message require?",
            criteria={"send_photo": "share an image", "web_search": "look something up", "none": None},
        ),
        "tone": Question(
            type="score",
            instructions="How far is the character drifting from her established tone?",
            criteria=["on tone", "slightly off", "clearly off", "out of character"],
        ),
        "narrated_action": Question(
            type="noul",
            instructions="Is the reply in progress narrating an action it did not execute?",
        ),
    }
    return DecisionRequest(state=state, questions=questions)


@pytest.fixture
def tool_call_request(roleplay_request: DecisionRequest) -> DecisionRequest:
    draft = Draft(text="Here you go!", tool_calls=(ToolCall(name="send_photo", arguments={"kind": "selfie"}),))
    return DecisionRequest(
        state=roleplay_request.state.model_copy(update={"draft": draft}), questions=roleplay_request.questions
    )
