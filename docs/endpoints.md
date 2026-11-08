# The teacher and the auditor

Rein's data comes from two open-weights models behind OpenAI-compatible endpoints. No closed model writes, labels or
judges anything in this project.

## Teacher: writes the synthetic data

- Model: `QuantTrio/Qwen3.5-35B-A3B-AWQ`, the official Qwen3.5-35B-A3B weights in AWQ 4-bit (Apache-2.0).
- Served in the project with vLLM 0.27.1 on one A100-80GB, context 32,768, up to 32 requests in flight.
- Any server with `/v1/chat/completions` and `/health` works. Point `REIN_TEACHER_URL` at it (see `.env.example`).
  For an endpoint behind Modal's proxy auth, set `MODAL_PROXY_KEY` and `MODAL_PROXY_SECRET`; for a Modal app,
  `REIN_TEACHER_APP` lets a job stop its containers when it ends.
- Thinking is disabled per request (`chat_template_kwargs.enable_thinking = false`).
- `python -m rein.teacher.smoke` checks the endpoint and measures its throughput.

The teacher writes scenarios with rule sheets (game NPCs, roleplay, tool agents, support), conversations per target
rule in many case forms (direct, paraphrase, slang, indirect, narration attack, intent spread over turns, stale
trigger, early reference, co-triggered rules, adversarial bypass), hard negatives with the same words and another
intent, drafts of replies in progress for the watcher, and long conversations with distractions. Prompts are in
`configs/datagen/`; the pipeline is `rein.datagen`.

## Auditor: re-checks every label and judges replies

- Model: `Qwen/Qwen3.8-27B-FP8`, served by `modal_apps/auditor_app.py` (vLLM 0.27.1, one H100, Modal proxy auth).
  Point `REIN_AUDITOR_URL` at it.
- Label-blind: the auditor answers each rule on its own and never sees the existing label.
- A non-thinking screen vote runs first. When it disagrees with the label, two thinking votes run, and the label
  changes only if both agree. Score questions change only when the gap is two levels or more; smaller
  disagreements become soft targets. About 3% of questions were corrected and 1.5% became soft.
- Every audited record keeps `meta.audit` with the status, the votes and the original label.
- The same endpoint is the blind judge of the end-to-end sessions (`rein.e2e.rule_judge`, `rein.sheets.e2e_judge`).
