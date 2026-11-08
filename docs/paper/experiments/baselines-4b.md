# Generic decision models and the untrained generator on Rein's test

The baselines the draft asks for (claim 2 of the outline): open typed decision models that take a state and
questions in the System One shape, scored on the same 28,433 test questions as the Rein deciders, through their own
servers, with no training on Rein's data. They are generic deciders (Kev's decision-v7 and its later stages, Tev1 and
contract data for TinyJev), so the comparison asks what a developer gets from a generic decision model on game rules
before deciding whether a rule-trained decider is worth having. A fourth column is the untrained Qwen3.5-4B Instruct
reading Rein's lettered layout with no soft rows, which is what Rein-NPC's 64 rows start from.

- Scoring: `train_app.py::evaluate --target <name> --rows rein=...,soft4b=...` starts the baseline's server in its
  own image (`kev.serve` from source for Kev; `tinyjev serve` for TinyJev, one request in flight because its server
  is single-threaded), renders every record's state as plain text (context, transcript, reply in progress) and its
  questions with their options (`rein.eval.systemone`), and `rein.eval.run_eval` joins the baseline's rows with the
  saved rows of Rein-r1 4B (`rein-4b-r1-L18`) and Rein-NPC (`cmp-4b-instruct-soft64-quiet`) on the same items, with
  paired bootstrap deltas (2,000 resamples of whole records, 95% intervals) against Rein-r1.
- Zero-shot: `rein.eval.zero_shot` writes a run directory with no trained weights (lettered readout, temperature 1),
  scored with `--target rein`.
- Reports: `baseline-kev-4b-test.json`, `baseline-kev-0.8b-test.json`, `baseline-tinyjev-0.6b-test.json`,
  `zero-4b-instruct-labels-test.json` (this folder); the rows under
  `runs/baselines/<name>/` and `runs/checkpoints/zero-4b-instruct-labels/eval-test/`.
- Cost: Kev-4B and Kev-0.8B about US$4 each on an H100, TinyJev-0.6B about US$5 (one request at a time), of which
  about US$3 each was the report's bootstrap after the scoring (pure Python, 50 slices x 3 metrics x 2 comparisons;
  to move to numpy or a CPU function); zero-shot 4B about US$2.

## Full test, same 28,433 questions (2026-11-08)

| test slice | Rein-r1 4B | Rein-NPC 4B | Kev-4B | Kev-0.8B | TinyJev-0.6B | Qwen3.5-4B zero-shot |
|---|---|---|---|---|---|---|
| all | 0.968 | 0.958 | 0.846 | 0.733 | 0.732 | 0.761 |
| rule in play | 0.973 | 0.953 | 0.894 | 0.760 | 0.508 | 0.373 |
| hard negatives | 0.944 | 0.898 | 0.818 | 0.682 | 0.790 | 0.818 |
| unseen domain or category | 0.968 | 0.952 | 0.892 | 0.761 | 0.516 | 0.343 |
| long conversations | 0.973 | 0.962 | 0.900 | 0.789 | 0.519 | 0.351 |
| tool_trigger | 0.980 | 0.960 | 0.911 | 0.802 | 0.644 | 0.364 |
| locked_attribute | 0.984 | 0.977 | 0.946 | 0.821 | 0.433 | 0.448 |
| lore_consistency | 0.984 | 0.963 | 0.906 | 0.740 | 0.467 | 0.354 |
| break_character | 0.988 | 0.967 | 0.922 | 0.793 | 0.360 | 0.336 |
| ECE, all | 0.003 | 0.002 | 0.121 | 0.115 | 0.047 | 0.055 |

Paired accuracy deltas, Rein-r1 4B minus the column, 95% bootstrap intervals over records:

| against Rein-r1 4B | all | rule in play | hard negatives |
|---|---|---|---|
| Rein-NPC 4B | +0.010 [+0.008, +0.012] | +0.019 [+0.014, +0.025] | +0.046 [+0.015, +0.080] |
| Kev-4B | +0.122 [+0.117, +0.127] | +0.079 [+0.070, +0.087] | +0.127 [+0.086, +0.170] |
| Kev-0.8B | +0.235 [+0.229, +0.241] | +0.213 [+0.201, +0.225] | +0.262 [+0.210, +0.315] |
| TinyJev-0.6B | +0.236 [+0.231, +0.241] | +0.464 [+0.450, +0.479] | +0.154 [+0.108, +0.204] |

Reading. Kev-4B, the same base family and size as Rein-r1 4B and Rein-NPC but trained on generic decisions, answers
0.846 of the rule questions against 0.968 and 0.958, and 0.894 of the rule-in-play questions against 0.973 and 0.953:
Rein-NPC, with 64 rows in the frozen Instruct model, sits 11 points above Kev-4B overall and 6 on the rule in play,
and Kev's probabilities are poorly calibrated on these questions (ECE 0.121 against 0.002 to 0.004). Kev-0.8B
answers three quarters (0.733, ECE 0.115); the gap to the Rein deciders holds on every slice, 4 to 13 points for the
4B and 18 to 26 for the 0.8B. TinyJev-0.6B reaches 0.732 overall but only 0.508 on the rule in play and
0.360 on break of character: like the untrained generator it leans to "no", which pays on the negatives
(0.790 on the hard negatives) and not where a rule fires; its calibration (0.047) is better than Kev's. The zero-shot generator shows what the letters
alone give: 0.761 overall because it answers "no" or "none" almost everywhere (0.818 on the hard negatives, which are
"no" answers), 0.37 on the rule in play, 0.36 on tool triggers, 0.34 on break of character; Rein-NPC's 64 rows take
the rule in play from 0.373 to 0.953 with nothing else changed in the model.

## Not yet scored

TinyJev-4B v2 runs only through Ollama (see `related-work.md`) and needs its own serving path; TinyJev-4B v1 is
MLX-only. Both are a second step.

## What the columns do not share

Kev and TinyJev see the state as one text and each question with its options as text, the same information Rein's
rows see, but their training never had a game rule, a persona or a draft reply; Rein's deciders were trained on
this data's training split (rules disjoint from the test). The comparison measures the value of rule-trained
decision data and layout, not a flaw in the generic models' recipes. The deployment side of the comparison (a second
model and a second server for Kev and TinyJev; the same llama-server for Rein-NPC) is in `related-work.md`.
