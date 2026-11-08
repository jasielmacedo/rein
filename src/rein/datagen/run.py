import argparse
import asyncio
import json
from pathlib import Path

import yaml

from rein.datagen.pipeline import DatagenConfig, Pipeline
from rein.datagen.taxonomy import load_taxonomy
from rein.teacher.client import TeacherClient
from rein.teacher.modal_ops import TEACHER_APP, stop_app_containers
from rein.teacher.settings import TeacherSettings


async def _run(configs: list[DatagenConfig]) -> dict[str, object]:
    settings = TeacherSettings.from_env()
    summary: dict[str, object] = {}
    async with TeacherClient(settings, max_in_flight=max(config.max_in_flight for config in configs)) as client:
        cold_start = await client.wait_healthy()
        print(f"teacher healthy after {cold_start:.0f}s")
        for config in configs:
            stats = await Pipeline(config, client, load_taxonomy()).run()
            summary[config.run_name] = {
                "accepted": dict(stats.accepted),
                "rejected": dict(stats.rejected),
                "tokens": stats.completion_tokens,
            }
            print(json.dumps({config.run_name: dict(stats.accepted)}))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Rein records with the Modal teacher.")
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--keep-teacher", action="store_true", help="leave the Modal container running afterwards")
    args = parser.parse_args()
    configs = [DatagenConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))) for path in args.config]
    try:
        summary = asyncio.run(_run(configs))
        print(json.dumps(summary, indent=2))
    finally:
        if not args.keep_teacher:
            print("stopped containers:", stop_app_containers(TEACHER_APP))


if __name__ == "__main__":
    main()
