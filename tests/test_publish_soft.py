from pathlib import Path

import pytest

from rein.export.publish_soft import (
    MODEL_CARD,
    gguf_uploads,
    parse_extras,
    publish_card,
    publish_checkpoint,
    publish_extras,
    publish_gguf,
)


class FakeApi:
    def __init__(self) -> None:
        self.repos: list[tuple[str, bool]] = []
        self.folders: list[tuple[str, str, list[str]]] = []
        self.files: list[tuple[str, str, str]] = []

    def create_repo(self, repo: str, private: bool, exist_ok: bool) -> None:
        self.repos.append((repo, private))

    def upload_folder(self, repo_id: str, folder_path: str, commit_message: str, ignore_patterns: list[str]) -> None:
        self.folders.append((repo_id, folder_path, ignore_patterns))

    def upload_file(self, path_or_fileobj: str, path_in_repo: str, repo_id: str, commit_message: str) -> None:
        self.files.append((repo_id, path_or_fileobj, path_in_repo))


def _gguf_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "gguf"
    folder.mkdir()
    for name in ("soft-Q8_0.gguf", "soft-Q4_K_M.gguf", "soft-bf16.gguf", "notes.txt", "rein_config.json"):
        (folder / name).write_bytes(b"x")
    return folder


def test_gguf_files_are_renamed_after_the_model(tmp_path: Path) -> None:
    folder = _gguf_folder(tmp_path)
    assert [target for _, target in gguf_uploads(folder, "Rein-NPC-4B")] == [
        "Rein-NPC-4B-bf16.gguf",
        "Rein-NPC-4B-Q4_K_M.gguf",
        "Rein-NPC-4B-Q8_0.gguf",
    ]


def test_publishing_creates_private_repos_and_replaces_the_base_readme(tmp_path: Path) -> None:
    api = FakeApi()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    card = tmp_path / "card.md"
    card.write_text("# card", encoding="utf-8")
    publish_checkpoint(api, "jmc0/Rein-NPC-4B", checkpoint, card, private=True)  # type: ignore[arg-type]
    assert api.repos == [("jmc0/Rein-NPC-4B", True)]
    assert api.folders[0][0] == "jmc0/Rein-NPC-4B" and MODEL_CARD in api.folders[0][2]
    assert api.files == [("jmc0/Rein-NPC-4B", str(card), MODEL_CARD)]

    published = publish_gguf(api, "jmc0/Rein-NPC-4B-GGUF", _gguf_folder(tmp_path), "Rein-NPC-4B", card, private=True)  # type: ignore[arg-type]
    assert published == [
        "Rein-NPC-4B-bf16.gguf",
        "Rein-NPC-4B-Q4_K_M.gguf",
        "Rein-NPC-4B-Q8_0.gguf",
        "rein_config.json",
    ]
    assert api.repos[-1] == ("jmc0/Rein-NPC-4B-GGUF", True)
    assert [target for repo, _, target in api.files if repo == "jmc0/Rein-NPC-4B-GGUF"] == [*published, MODEL_CARD]


def test_card_only_sends_the_card_and_the_decision_config(tmp_path: Path) -> None:
    api = FakeApi()
    card = tmp_path / "card.md"
    card.write_text("# card", encoding="utf-8")
    assert publish_card(api, "jmc0/Rein-NPC-4B-GGUF", card, _gguf_folder(tmp_path)) == [MODEL_CARD, "rein_config.json"]  # type: ignore[arg-type]
    assert publish_card(api, "jmc0/Rein-NPC-4B", card, tmp_path) == [MODEL_CARD]  # type: ignore[arg-type]
    assert api.repos == []


def test_extra_files_are_parsed_and_shipped_under_their_repo_path(tmp_path: Path) -> None:
    sheet = tmp_path / "shop-keeper.yaml"
    sheet.write_text("npc: Mira", encoding="utf-8")
    extras = parse_extras([f"{sheet}:examples/shop-keeper.yaml"])
    assert extras == [(sheet, "examples/shop-keeper.yaml")]
    api = FakeApi()
    assert publish_extras(api, "jmc0/Rein-NPC-4B-GGUF", extras) == ["examples/shop-keeper.yaml"]  # type: ignore[arg-type]
    assert api.files == [("jmc0/Rein-NPC-4B-GGUF", str(sheet), "examples/shop-keeper.yaml")]
    with pytest.raises(ValueError):
        parse_extras(["no-separator"])
