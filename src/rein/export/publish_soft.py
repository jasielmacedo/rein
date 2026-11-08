"""Puts a soft-token model on the Hub: the merged checkpoint with its card as one model repo, and its GGUF files,
renamed after the model, with their card as a second repo. Both repos are created private, for a look before they
open."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from rein.env import load_env, optional
from rein.model.run_meta import CONFIG_FILE

MODEL_CARD = "README.md"
GGUF_SUFFIX = ".gguf"
EXPORT_LABEL = "soft-"
CHECKPOINT_IGNORE = (MODEL_CARD, ".gitattributes")


def gguf_uploads(folder: Path, name: str) -> list[tuple[Path, str]]:
    """The folder's GGUF files and their names in the repo: `soft-Q4_K_M.gguf` becomes `<name>-Q4_K_M.gguf`."""
    uploads = []
    for path in sorted(folder.glob(f"*{GGUF_SUFFIX}"), key=lambda path: path.name.lower()):
        quant = path.stem.removeprefix(EXPORT_LABEL)
        uploads.append((path, f"{name}-{quant}{GGUF_SUFFIX}"))
    return uploads


def publish_checkpoint(api: HfApi, repo: str, folder: Path, card: Path, private: bool) -> None:
    """Every file of the merged checkpoint but the base's README, then the card as the README."""
    api.create_repo(repo, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo,
        folder_path=str(folder),
        commit_message="Rein-NPC weights",
        ignore_patterns=list(CHECKPOINT_IGNORE),
    )
    api.upload_file(path_or_fileobj=str(card), path_in_repo=MODEL_CARD, repo_id=repo, commit_message="model card")


def publish_gguf(api: HfApi, repo: str, folder: Path, name: str, card: Path, private: bool) -> list[str]:
    """The GGUF files under the model's name, the decision config beside them when the folder has it, and the card."""
    api.create_repo(repo, private=private, exist_ok=True)
    published = []
    uploads = gguf_uploads(folder, name)
    if (folder / CONFIG_FILE).exists():
        uploads.append((folder / CONFIG_FILE, CONFIG_FILE))
    for path, target in uploads:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=target, repo_id=repo, commit_message=target)
        published.append(target)
    api.upload_file(path_or_fileobj=str(card), path_in_repo=MODEL_CARD, repo_id=repo, commit_message="model card")
    return published


def publish_card(api: HfApi, repo: str, card: Path, folder: Path) -> list[str]:
    """Only the small files: the card, and the decision config when the folder holds one."""
    uploads = [(card, MODEL_CARD)]
    if (folder / CONFIG_FILE).exists():
        uploads.append((folder / CONFIG_FILE, CONFIG_FILE))
    for path, target in uploads:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=target, repo_id=repo, commit_message=target)
    return [target for _, target in uploads]


def parse_extras(values: list[str]) -> list[tuple[Path, str]]:
    """`local/path:repo/path` pairs of extra files to ship in every repo, such as an example sheet."""
    extras = []
    for value in values:
        local, separator, remote = value.rpartition(":")
        if not separator or not local or not remote:
            raise ValueError(f"extra files are given as local/path:repo/path, not {value!r}")
        extras.append((Path(local), remote))
    return extras


def publish_extras(api: HfApi, repo: str, extras: list[tuple[Path, str]]) -> list[str]:
    for path, target in extras:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=target, repo_id=repo, commit_message=target)
    return [target for _, target in extras]


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a soft-token model and its GGUF files to the Hub.")
    parser.add_argument("--name", required=True, help="the model's name, used in the GGUF file names")
    parser.add_argument("--checkpoint", type=Path, help="the merged safetensors checkpoint")
    parser.add_argument("--card", type=Path, help="the checkpoint's model card")
    parser.add_argument("--repo", help="repo id for the checkpoint")
    parser.add_argument("--gguf-dir", type=Path, help="folder with soft-<quant>.gguf files")
    parser.add_argument("--gguf-card", type=Path, help="the GGUF repo's card")
    parser.add_argument("--gguf-repo", help="repo id for the GGUF files")
    parser.add_argument("--public", action="store_true", help="create the repos public (default: private)")
    parser.add_argument("--card-only", action="store_true", help="upload the cards and the decision config only")
    parser.add_argument(
        "--extra", nargs="*", default=[], help="local/path:repo/path files to ship in every repo (an example sheet)"
    )
    args = parser.parse_args()
    api = HfApi(token=optional(load_env(), "HF_TOKEN"))
    private = not args.public
    visibility = "public" if args.public else "private"
    extras = parse_extras(args.extra)
    if args.checkpoint is not None:
        if args.card is None or args.repo is None:
            raise SystemExit("--checkpoint needs --card and --repo")
        if args.card_only:
            print(f"pushed {publish_card(api, args.repo, args.card, args.checkpoint)} to {args.repo}")
        else:
            publish_checkpoint(api, args.repo, args.checkpoint, args.card, private)
            print(f"pushed {args.checkpoint} to {args.repo} ({visibility})")
        if extras:
            print(f"pushed {publish_extras(api, args.repo, extras)} to {args.repo}")
    if args.gguf_dir is not None:
        if args.gguf_card is None or args.gguf_repo is None:
            raise SystemExit("--gguf-dir needs --gguf-card and --gguf-repo")
        if args.card_only:
            print(f"pushed {publish_card(api, args.gguf_repo, args.gguf_card, args.gguf_dir)} to {args.gguf_repo}")
        else:
            files = publish_gguf(api, args.gguf_repo, args.gguf_dir, args.name, args.gguf_card, private)
            print(f"pushed {files} to {args.gguf_repo} ({visibility})")
        if extras:
            print(f"pushed {publish_extras(api, args.gguf_repo, extras)} to {args.gguf_repo}")


if __name__ == "__main__":
    main()
