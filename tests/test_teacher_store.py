from pathlib import Path

from rein.teacher.store import JsonlStore, content_key


def test_append_is_idempotent(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path / "rows.jsonl")
    assert store.append("a", {"value": 1})
    assert not store.append("a", {"value": 2})
    assert "a" in store
    assert len(store) == 1


def test_reopen_resumes_keys(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    first = JsonlStore(path)
    first.append("a", {"value": 1})
    first.append("b", {"value": 2})
    reopened = JsonlStore(path)
    assert len(reopened) == 2
    assert not reopened.append("b", {"value": 3})
    assert [row["value"] for row in reopened.rows()] == [1, 2]


def test_partial_trailing_line_is_isolated(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"key": "a", "value": 1}\n{"key": "b", "val', encoding="utf-8")
    store = JsonlStore(path)
    assert store.append("c", {"value": 3})
    assert [row["key"] for row in store.rows()] == ["a", "c"]


def test_content_key_ignores_dict_order() -> None:
    assert content_key({"x": 1, "y": 2}, "p") == content_key({"y": 2, "x": 1}, "p")
    assert content_key({"x": 1}) != content_key({"x": 2})
