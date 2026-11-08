from pathlib import Path

from rein.play.engine import history
from rein.play.server import is_local_client
from rein.play.store import ASSISTANT, TOOL, USER, Store


def _store(tmp_path: Path) -> Store:
    return Store(tmp_path / "play.sqlite")


def _reply_turns() -> list[dict[str, object]]:
    return [
        {"role": USER, "text": "my leg is bleeding"},
        {
            "role": ASSISTANT,
            "text": "",
            "tool_call": {"name": "bandage", "arguments": {}},
            "prompt_ids": [1, 2, 3],
            "raw": "<tool_call>",
            "seed": 7,
        },
        {"role": TOOL, "text": "done", "tool_name": "bandage"},
        {"role": ASSISTANT, "text": "Hold still.", "prompt_ids": [1, 2, 3, 4], "raw": "Hold still.", "seed": 8},
    ]


def test_messages_come_back_in_order_with_their_tool_call(tmp_path: Path) -> None:
    store = _store(tmp_path)
    conversation = store.create_conversation("play-00-sparse", {"id": "play-00-sparse"}, {"model": "m.gguf"})
    stored = store.add_messages(conversation, _reply_turns())
    assert [message["role"] for message in stored] == [USER, ASSISTANT, TOOL, ASSISTANT]
    assert stored[1]["tool_call"] == {"name": "bandage", "arguments": {}}
    assert store.messages(conversation) == stored
    assert store.conversations("play-00-sparse")[0]["first_text"] == "my leg is bleeding"


def test_labels_only_land_on_model_replies(tmp_path: Path) -> None:
    store = _store(tmp_path)
    conversation = store.create_conversation("play-00-sparse", {}, {})
    user, reply, *_ = store.add_messages(conversation, _reply_turns())
    assert not store.set_label(user["id"], "wrong")
    assert store.set_label(reply["id"], "  should have bandaged  ")
    assert store.messages(conversation)[1]["label"] == "should have bandaged"


def test_deleting_a_conversation_deletes_its_messages(tmp_path: Path) -> None:
    store = _store(tmp_path)
    conversation = store.create_conversation("play-00-sparse", {}, {})
    store.add_messages(conversation, _reply_turns())
    assert store.delete_conversation(conversation)
    assert store.messages(conversation) == []
    assert store.conversation(conversation) is None
    assert not store.delete_conversation(conversation)


def test_history_rebuilds_calls_and_tool_results(tmp_path: Path) -> None:
    store = _store(tmp_path)
    conversation = store.create_conversation("play-00-sparse", {}, {})
    rebuilt = history(store.add_messages(conversation, _reply_turns()))
    assert [message.role for message in rebuilt] == [USER, ASSISTANT, TOOL, ASSISTANT]
    assert rebuilt[1].tool_calls[0].name == "bandage"
    assert rebuilt[2].name == "bandage"


def test_only_loopback_and_private_clients_get_in() -> None:
    assert is_local_client("127.0.0.1")
    assert is_local_client("192.168.0.12")
    assert not is_local_client("8.8.8.8")
    assert not is_local_client(None)
    assert not is_local_client("not an address")
