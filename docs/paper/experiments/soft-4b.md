# soft-4b: the decider as 64 soft tokens inside the Qwen3.5-4B Instruct checkpoint

The line opened by DR-030 and DR-032: no second model, no change to Qwen's weights, no fine-tuning. The decider is
64 new special tokens (`<|rein_0|>` to `<|rein_63|>`, ids 248077 to 248140) whose embedding rows, spare rows of the
Instruct model's own table, are the only trained parameters (64 x 2,560 numbers). A decision is one forward pass of
the unchanged generator over the conversation plus a row in Rein's layout: 48 soft tokens, the question, the lettered
options, 16 soft tokens, and the answer is read from the letters' next-token logits. The same GGUF talks and decides.

Runs: `cmp-0.8b-instruct-soft-labels` (24 tokens, 2 epochs), `cmp-0.8b-instruct-soft64-labels` (64 tokens, 4 epochs),
`cmp-0.8b-instruct-soft64-quiet` (64 tokens, silence term), `cmp-4b-instruct-soft64-quiet` (the 4B; this card).
Eval reports: `cmp-4b-instruct-soft64-quiet-test.json`, `cmp-4b-instruct-soft64-quiet-dev.json`.

## Recipe of the 4B run

- Base `Qwen/Qwen3.5-4B` (the post-trained model), frozen, full 32 layers (llama.cpp computes them all, so the
  readout sits at the last layer; the 0.8B runs read at the last layer too).
- 64 soft rows, lr 1e-3 (AdamW, one-cycle), CE + 0.25 Brier on the lettered readout, 4 epochs, 5,724 steps,
  batch 32,768 tokens, seed 0. Temperature fitted on the calibration split: 0.812.
- Silence term (DR-032): a bank of 131k positions from 1,024 training conversations rendered through the chat template,
  a hinge on the worst 2% of positions (margin 20 nats between the real vocabulary's logsumexp and the soft rows'),
  weight 10, the rows projected off the bank's top 32 output directions. The bank's worst gap held at +10.6 nats.
- Modal H100: 11,212 s of training (3 h 07), about US$15 with the bank, the dev passes and the test; US$4 more on a
  first attempt (mean hinge, weight 1) stopped when the bank's gap turned negative.
- Labels: mix-v1-audited, the same data and splits as every Rein run.

## Test, rules never seen (28,433 questions, audited labels)

| Slice | n | Accuracy | ECE |
|---|---|---|---|
| All questions | 28,433 | 0.958 | 0.002 |
| Target rule | 5,179 | 0.953 | 0.005 |
| hard_negative | 324 | 0.898 | 0.036 |
| narrated_action (watcher) | 525 | 0.909 | 0.031 |
| Held-out domain | 1,491 | 0.952 | 0.012 |
| Held-out category (mandatory_phrase) | 308 | 0.942 | 0.032 |
| Long conversations | 1,276 | 0.962 | 0.012 |
| tool_trigger | 404 | 0.960 | 0.022 |
| locked_attribute | 386 | 0.977 | 0.020 |
| lore_consistency | 381 | 0.963 | 0.016 |
| break_character | 333 | 0.967 | 0.018 |
| prompt_injection | 363 | 0.989 | 0.012 |
| tone_drift (score) | 94 | 0.745 | 0.135 |

Dev by epoch: 0.934, 0.952, 0.953, 0.956 (still rising at the end).

## Against the other deciders on the same test

| test slice | Rein-r1 0.8B | A aLoRA 0.8B | B 64 0.8B | B 64 quiet 0.8B | Rein-r1 4B | B 64 quiet 4B |
|---|---|---|---|---|---|---|
| all | 0.955 | 0.952 | 0.940 | 0.940 | 0.968 | 0.958 |
| rule in play | 0.955 | 0.944 | 0.922 | 0.921 | 0.973 | 0.953 |
| hard negatives | 0.895 | 0.858 | 0.833 | 0.840 | 0.944 | 0.898 |
| unseen domain or category | 0.942 | 0.932 | 0.904 | 0.908 | 0.968 | 0.952 |
| long conversations | 0.954 | 0.949 | 0.930 | 0.934 | 0.973 | 0.962 |
| tool_trigger | 0.958 | 0.955 | 0.938 | 0.943 | 0.980 | 0.960 |
| locked_attribute | 0.979 | 0.964 | 0.959 | 0.943 | 0.984 | 0.977 |
| lore_consistency | 0.963 | 0.950 | 0.929 | 0.934 | 0.984 | 0.963 |
| break_character | 0.982 | 0.970 | 0.949 | 0.946 | 0.988 | 0.967 |
| ECE | 0.004 | 0.004 | 0.003 | 0.005 | 0.003 | 0.002 |
| training | 1 h 19 | 0 h 46 | 1 h 10 | 1 h 08 | 1 h 54 | 3 h 07 |

Rein-r1 is a LoRA (r16) through the truncated Base model with a pointer head: a second model next to the generator
(0.8B at block 24, 4B at block 18). A aLoRA is an activated LoRA on the 0.8B Instruct (rows only when the row
starts), the alternative to soft tokens that also needs no second model but does need runtime support. B is the soft
tokens. On the 4B the soft tokens match the 0.8B Rein-r1 on every slice and stay 1.0 point under the 4B Rein-r1
overall, 2.0 on the rule in play and 4.6 on the hard negatives, with nothing added to the model but 64 rows.
Whether the silence constraint cost accuracy on the 4B is open (it did not on the 0.8B: 0.921 against 0.922).

## The tied head, the silence term, and the untied export

Qwen3.5 ties its input embeddings to its output head, so trained soft rows become output tokens: the 24-token rows
took 0.39 of the next-token mass on average over chat positions, the 64-token rows 0.49 (`rein.eval.soft_silence`,
6 play sessions, 12,288 positions). Two fixes, both in this run's exports:

- The silence term keeps the rows out of the chat on the tied copy. Over 2,000 held-out test conversations (984,371
  positions): mean soft mass 1.2e-6, maximum 0.0010, worst gap +6.9 nats, best rank 14, 802 positions (0.08%) with a
  soft token inside the top 100. The 0.8B's mean-hinge version leaked (worst gap −7.2, maximum mass 0.999, 3.3% in the
  top 100); the tail hinge of the 4B does not.
- `rein.export.soft_merge --untie` writes the base's table a second time as `lm_head.weight`, so the chat's logits are
  exactly Qwen's whatever the context: mean 6.5e-8, maximum 1.5e-5, worst gap +11.1, no soft token in the top 100, the
  base's own spare rows. Cost: the second table, +358 MB at Q4_K_M (Q6_K head).

Both merges change no tensor but the soft rows (`--verify`). The merged checkpoints convert and quantize with the
stock `convert_hf_to_gguf.py` and `llama-quantize`:

| file | tied | untied |
|---|---|---|
| safetensors bf16 | 8.0 GB | 10.6 GB |
| GGUF bf16 | 8.42 GB | 9.70 GB |
| GGUF Q8_0 | 4.48 GB | 5.16 GB |
| GGUF Q4_K_M | 2.71 GB | 3.07 GB |

Decisions through llama-server on the Q4_K_M files against the run in PyTorch (`rein.eval.soft_agreement`, 20 held-out
test records, 104 questions): untied 101 of 104 the same option (mean largest probability gap 0.028, worst 0.21), tied
102 of 104 (0.030, 0.39); every disagreement is a question PyTorch itself leaves under 0.65, and on the 90 questions
where PyTorch puts 0.8 or more on an option both files agree 90 times. The 0.8B tied files gave 97 to 99 of 104 at
Q4_K_M and 104 of 104 at f16: the quantization of the backbone, not of the rows.

## Cost of the validation

Builds (merge on CPU, convert and quantize on L4): about US$0.05 each. Silence passes on L4: about US$0.40 each.
Agreement passes on L4: US$0.58 and US$0.63 (the PyTorch 4B without the fast kernels is slow on an L4). End to end
and judge: see `e2e-soft-4b.md`. Whole validation of the 4B under US$7.

## The user's hand-written gold set

`data/phase2/gold-block/gold.jsonl`: the 25 agents and conversations the user wrote by hand (`tools/gold-project.json`,
turned into decision records by `rein.gold.project`), 428 records and 1,352 questions, 92 of them with a reply in
progress; the user's own marks are the labels, unmarked turns take the null option. Scored with the same evaluator
as the test (`cmp-4b-instruct-soft64-quiet-gold.json`), nothing of Rein-NPC tuned on it. The set was inspected during
the phase 2 neurons' diagnosis, so it is development material for that line, not for this one.

| Slice | n | Accuracy | ECE |
|---|---|---|---|
| All questions | 1352 | 0.924 | 0.025 |
| Target rule | 155 | 0.942 | 0.061 |
| Long conversations | 29 | 0.966 | 0.054 |
| tool_trigger | 78 | 0.910 | 0.108 |
| locked_attribute | 8 | 0.875 | 0.139 |
| mutable_attribute | 13 | 1.000 | 0.032 |
| lore_consistency | 9 | 1.000 | 0.075 |
| business_limit | 11 | 1.000 | 0.047 |
| npc_secret | 15 | 1.000 | 0.045 |
| pii | 5 | 1.000 | 0.002 |

The user's style costs 3.4 points against the teacher-written test overall (0.924 against 0.958) and
1.1 on the rule in play (0.942 against 0.953); the 78 tool openings land at 0.910. The
small slices (5 to 15 questions) are read as signs, not measurements. npc_secret and pii are out of this line's scope
(DR-031) and reported for completeness. Rein-r1, the two-model reader, reached 0.986 on the gold rule in play
(research log, 2026-11-06). Cost: about US$0.50 on an H100.

## What this run settled and what it left open

Settled: the decider fits in the Instruct checkpoint's spare rows with no change to the generator, reaches the 0.8B
Rein-r1's accuracy on the 4B, and ships as one standard GGUF whose chat is Qwen's (untied) or quiet to 0.1% (tied).
Open: the gap to the 4B Rein-r1 (more epochs, the unconstrained run `cmp-4b-instruct-soft64-free.yaml` with the untied
export), tone_drift (0.745 on 94 questions), and the situation-triggered tool calls seen in the human play material
(`e2e-soft-4b.md`).
