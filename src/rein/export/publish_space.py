"""Publishes a static page folder as a Hugging Face Space: the playground, or any folder with an `index.html` and a
README whose front matter says `sdk: static`. The Space is created private unless `--public` is given."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from rein.env import load_env, optional

SPACE_SDK = "static"
DEFAULT_FOLDER = Path("tools/playground")
DEFAULT_REPO = "jmc0/rein-npc-playground"


def publish_space(api: HfApi, repo: str, folder: Path, public: bool, message: str) -> str:
    api.create_repo(repo, repo_type="space", space_sdk=SPACE_SDK, private=not public, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="space", folder_path=str(folder), commit_message=message)
    if public:
        api.update_repo_settings(repo_id=repo, repo_type="space", private=False)
    return f"https://huggingface.co/spaces/{repo}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a static page folder as a Hugging Face Space.")
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--public", action="store_true", help="open the Space (default: private)")
    parser.add_argument("--message", default="playground update")
    args = parser.parse_args()
    api = HfApi(token=optional(load_env(), "HF_TOKEN"))
    url = publish_space(api, args.repo, args.folder, args.public, args.message)
    print(f"pushed {args.folder} to {url} ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
