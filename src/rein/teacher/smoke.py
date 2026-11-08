import argparse
import asyncio
import json
import statistics
import time
from typing import Any

from rein.env import PROJECT_ROOT
from rein.teacher.client import TeacherClient, parse_json_object
from rein.teacher.settings import TeacherSettings

DEFAULT_LEVELS = (8, 16, 32)
DEFAULT_COMPLETION_TOKENS = 384
JSON_CHECK_MAX_TOKENS = 128
JSON_CHECK_KEYS = ("name", "mood")
THINK_MARKERS = ("<think>", "</think>")
OUT_DIR = PROJECT_ROOT / "runs" / "teacher-smoke"

JSON_CHECK_MESSAGES = [
    {"role": "user", "content": "Reply with a JSON object holding the keys name and mood for a tavern keeper NPC."}
]


def bench_messages(index: int) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                f"Scene {index}. Write 8 short roleplay turns between a user and a fantasy innkeeper. "
                "Use *asterisks* for actions."
            ),
        }
    ]


async def run_smoke(levels: tuple[int, ...], completion_tokens: int) -> dict[str, Any]:
    settings = TeacherSettings.from_env()
    async with TeacherClient(settings, max_in_flight=max(levels)) as client:
        cold_start_s = await client.wait_healthy()
        check = await client.chat(JSON_CHECK_MESSAGES, temperature=0.0, top_p=1.0, max_tokens=JSON_CHECK_MAX_TOKENS)
        parsed = parse_json_object(check.content)
        json_check = {
            "keys_present": all(key in parsed for key in JSON_CHECK_KEYS),
            "think_leak": any(marker in check.content for marker in THINK_MARKERS),
            "latency_s": round(check.latency_s, 2),
        }
        bench = []
        for level in levels:
            started = time.monotonic()
            results = await asyncio.gather(
                *(
                    client.chat(
                        bench_messages(i),
                        max_tokens=completion_tokens,
                        json_mode=False,
                        extra={"min_tokens": completion_tokens},
                    )
                    for i in range(level)
                )
            )
            elapsed = time.monotonic() - started
            tokens = sum(result.completion_tokens for result in results)
            bench.append(
                {
                    "in_flight": level,
                    "elapsed_s": round(elapsed, 2),
                    "completion_tokens": tokens,
                    "tokens_per_s": round(tokens / elapsed, 1),
                    "mean_latency_s": round(statistics.mean(result.latency_s for result in results), 2),
                }
            )
    return {"cold_start_s": round(cold_start_s, 1), "json_check": json_check, "bench": bench}


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm up the teacher endpoint and measure throughput.")
    parser.add_argument("--levels", default=",".join(str(level) for level in DEFAULT_LEVELS))
    parser.add_argument("--completion-tokens", type=int, default=DEFAULT_COMPLETION_TOKENS)
    args = parser.parse_args()
    levels = tuple(int(part) for part in args.levels.split(",") if part.strip())
    report = asyncio.run(run_smoke(levels, args.completion_tokens))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"smoke-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
