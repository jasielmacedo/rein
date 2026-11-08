"""Qwen3.5-4B writes a sheet's example cases in one vLLM batch, with no teacher: each prompt is the case writer's own
prompt, and an answer that fails the case checks gets one more try at a lower temperature."""

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rein.datagen.generate import CASE_MAX_TOKENS, GENERATION_TEMPERATURE, REPAIR_TEMPERATURE, check_case
from rein.datagen.models import ConversationDraft
from rein.datagen.plan import CaseSpec
from rein.teacher.client import TeacherError, parse_json_object

TEMPERATURES = (GENERATION_TEMPERATURE, REPAIR_TEMPERATURE)
MAX_MODEL_LEN = 16384
NO_IMAGES = {"image": 0, "video": 0}
SEED = 0


def written_case(content: str, spec: CaseSpec) -> bool:
    """Whether Qwen's answer is a case the pipeline would accept from the teacher."""
    try:
        conversation = ConversationDraft.model_validate(parse_json_object(content))
    except (TeacherError, ValidationError, ValueError):
        return False
    return check_case(conversation, spec) is None


def write(model: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, max_model_len=MAX_MODEL_LEN, limit_mm_per_prompt=NO_IMAGES, seed=SEED)
    tokenizer = llm.get_tokenizer()
    written: dict[str, str] = {}
    pending = rows
    for temperature in TEMPERATURES:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in pending
        ]
        outputs = llm.generate(prompts, SamplingParams(temperature=temperature, max_tokens=CASE_MAX_TOKENS, seed=SEED))
        failed = []
        for row, output in zip(pending, outputs, strict=True):
            content = output.outputs[0].text
            if written_case(content, CaseSpec(**row["spec"])):
                written[row["case_id"]] = content
            else:
                failed.append(row)
        print(json.dumps({"temperature": temperature, "written": len(written), "failed": len(failed)}), flush=True)
        pending = failed
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5-4B writes the example cases of a sheet.")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    written = write(args.model, rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(
            json.dumps(
                {"case_id": row["case_id"], "prompt": row["prompt"], "content": written.get(row["case_id"])},
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    print(json.dumps({"prompts": len(rows), "written": len(written)}), flush=True)


if __name__ == "__main__":
    main()
