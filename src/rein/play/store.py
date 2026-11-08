"""The local chat's memory, one SQLite file: the conversations a person had with the model, every message, the exact
prompt and raw output behind each model reply (so the training view can be rebuilt as the model saw it), and the
person's own labels on replies. Each conversation keeps a copy of its character's sheet as it was when it started."""

import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

USER: Final = "user"
ASSISTANT: Final = "assistant"
TOOL: Final = "tool"
PREVIEW_CHARS = 80

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    character TEXT NOT NULL,
    sheet TEXT NOT NULL,
    settings TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    text TEXT NOT NULL,
    tool_call TEXT,
    tool_name TEXT,
    prompt_ids TEXT,
    raw TEXT,
    seed INTEGER,
    label TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    decisions TEXT
);
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages (conversation_id, id);
"""
DECISIONS = "decisions"
MESSAGE_COLUMNS = f"id, conversation_id, role, text, tool_call, tool_name, label, created_at, {DECISIONS}"
JSON_COLUMNS = ("tool_call", DECISIONS)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _message(row: sqlite3.Row) -> dict[str, Any]:
    message = dict(row)
    for column in JSON_COLUMNS:
        message[column] = json.loads(message[column]) if message[column] else None
    return message


def _json(value: Any) -> str | None:
    return json.dumps(value, ensure_ascii=False) if value else None


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(messages)")}
            if DECISIONS not in columns:
                connection.execute(f"ALTER TABLE messages ADD COLUMN {DECISIONS} TEXT")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection

    def create_conversation(self, character: str, sheet: Mapping[str, Any], settings: Mapping[str, Any]) -> int:
        now = _now()
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT INTO conversations (character, sheet, settings, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (character, json.dumps(sheet, ensure_ascii=False), json.dumps(settings), now, now),
            )
            return int(cursor.lastrowid or 0)

    def conversations(self, character: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT c.id, c.created_at, c.updated_at, COUNT(m.id) AS messages,
                          (SELECT text FROM messages WHERE conversation_id = c.id AND role = 'user'
                           ORDER BY id LIMIT 1) AS first_text
                   FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
                   WHERE c.character = ? GROUP BY c.id ORDER BY c.updated_at DESC""",
                (character,),
            ).fetchall()
        return [{**dict(row), "first_text": (row["first_text"] or "")[:PREVIEW_CHARS]} for row in rows]

    def conversation_counts(self) -> dict[str, int]:
        with self._connection() as connection:
            rows = connection.execute("SELECT character, COUNT(*) FROM conversations GROUP BY character").fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def conversation(self, conversation_id: int) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, character, sheet, created_at FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        if row is None:
            return None
        return {**dict(row), "sheet": json.loads(row["sheet"])}

    def messages(self, conversation_id: int) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT {MESSAGE_COLUMNS} FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
            ).fetchall()
        return [_message(row) for row in rows]

    def add_messages(self, conversation_id: int, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Stores the messages in order in one transaction and returns them as `messages` lists them."""
        now = _now()
        ids = []
        with self._connection() as connection:
            for message in messages:
                prompt_ids = message.get("prompt_ids")
                cursor = connection.execute(
                    f"""INSERT INTO messages (conversation_id, role, text, tool_call, tool_name, prompt_ids, raw, seed,
                                              created_at, {DECISIONS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        conversation_id,
                        message["role"],
                        message["text"],
                        _json(message.get("tool_call")),
                        message.get("tool_name"),
                        json.dumps(list(prompt_ids)) if prompt_ids else None,
                        message.get("raw"),
                        message.get("seed"),
                        now,
                        _json(message.get(DECISIONS)),
                    ),
                )
                ids.append(int(cursor.lastrowid or 0))
            connection.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
            marks = ", ".join("?" for _ in ids)
            rows = connection.execute(
                f"SELECT {MESSAGE_COLUMNS} FROM messages WHERE id IN ({marks}) ORDER BY id", ids
            ).fetchall()
        return [_message(row) for row in rows]

    def delete_conversation(self, conversation_id: int) -> bool:
        with self._connection() as connection:
            return connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount > 0

    def set_label(self, message_id: int, label: str) -> bool:
        """The person's note on one model reply; an empty label clears it."""
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE messages SET label = ? WHERE id = ? AND role = ?", (label.strip(), message_id, ASSISTANT)
            )
            return cursor.rowcount > 0
