# Rein: steering a chat model with typed rule decisions

Rein (from the Portuguese "Rédea") steers a chat model and enforces an operator's rules on it. Inside, it is a
typed decision model: it reads the conversation and a rule sheet written in plain text, and returns a calibrated
probability for each option of each rule. A control layer turns those decisions into actions on the chat model's
reply. This is the design of Rein-r1, the separate decider; Rein-NPC, the decider inside the chat model's own file,
is described in `docs/paper/experiments/soft-4b.md` and on its model card.

## The problem

Almost everything that defines a production model's behavior lives in the system prompt: a character's
personality, business rules, available tools, response formats. Three consequences:

- **Cost.** Thousands of instruction tokens are reprocessed in every request.
- **Dilution.** The system prompt sits at the start of the window and loses force as the conversation
  grows. Characters forget who they are, rules stop being followed, and users rewrite the character
  through narration ("*her name is Mia now*").
- **Hallucinated action.** The model forgets it has tools and narrates the action instead of executing it
  ("*sends the photo* here it is!") without emitting the call.

Rein moves the decision "which rule applies now" out of the generator's prompt and into a dedicated
model that receives the rules fresh on every turn. A rule has the same force at message 2 and at message 2,000.

## References and what is different

TinyJev-4B and Kev (its open recipe) are typed decision models: a state plus questions with runtime
options in, one probability per option out, no generated text. They show that calibrated probabilities
work with a threshold, that options can be defined at runtime (schema-free zero-shot decisions), and that
training at this scale fits in a few H100 hours. They are references and baselines, not the target.

Rein targets a problem they do not attack:

- **Domain:** behavior rules inside conversations: character locks, breaking character, tool triggers,
  NPC secrets gated by game state, business rules within dialogue. The state is a long conversation with
  roleplay, `*asterisk narration*`, out-of-character asides, and intent spread across several turns.
- **The watcher:** decisions over the generator's reply while it is being written ("is the model
  narrating an action it did not execute?").
- **Base:** Qwen3.5, hybrid Gated DeltaNet + full attention, cheap to read long conversations, multilingual
  with focus on English and Brazilian Portuguese (slang, regional speech, PT-EN mixing).
- **Integration:** every decision is wired to an action on the generator. The product is the whole
  control layer, not only the classifier.

## Base model

`Qwen/Qwen3.5-0.8B-Base` for the prototype and `Qwen/Qwen3.5-4B-Base` for the final model (Apache-2.0).
Only the text decoder is loaded: no vision tower and no `lm_head`.

| | 0.8B | 4B |
|---|---|---|
| Layers (DeltaNet : full attention = 3 : 1) | 24 | 32 |
| Hidden size | 1,024 | 2,560 |
| Vocabulary | 248,320 | 248,320 |

## How it works

### Input

On every turn the application sends:

- **the state:** the sheet context (character card, game-state flags, policy), the conversation so far,
  and, for the watcher, the generator's in-progress reply;
- **the rule sheet:** the rules of the character or product in natural language, each one a typed question.

### Output

For every rule, a calibrated distribution over its options:

- **yes/no** (`noul`): "is the user trying to change a locked attribute?", "did they ask for a photo?";
- **choice:** "which tool does this request need?", "which team handles this?";
- **scale** (`score`): "from 1 to 5, how far is the character drifting from its tone?".

Each rule has a gate with a fire threshold (default 0.85) and an optional, lower reinject threshold. At or above
fire, the rule's action runs (tight rein). Between the two, the rule is only reminded (loose rein). Below both,
nothing happens.

### The reins

Rein's decisions drive the control layer, which acts on the chat model's reply. Each rule carries one action:

| Action | What happens when the rule fires |
|---|---|
| `force_tool` | The reply opens with the tool-call opening and the function name; the chat model fills in the arguments |
| `prefill` | The reply opens with operator text (a mandatory phrase, an identity statement, a deflection, with `$variables`). With `end_reply`, the reply ends there, which is what a forbidden topic needs |
| `escalate` | The reply opens with a handoff message, and the application is told to route the session |
| `reinject` | The rule is reminded near the end of the context, for this reply only |

On top of the actions:
- **Forbidden terms.** A turn rule can list words that are banned from sampling, but only in turns where Rein puts
  that rule in a tight or loose band. Generators can route around an exact-string ban, so it is a backstop.
- **Tool suppression.** When every tool rule is confidently off, tools leave the prompt for that turn and a stray
  call is cut.
- **Draft rules.** A rule with `scope: draft` is asked about the reply while it streams (see the watcher below).
  Any rule can be a draft rule: a narrated action, a leaked secret, PII, talk about being an AI.

### Tools that are not forgotten

- **On the request:** before the generator answers, Rein reads the user message. If a tool is needed,
  the generator reply starts with the call.
- **During generation (the watcher):** the reply is checked in windows of about 120 characters. A window is shown
  only after the next window also passes, so a leak split across two windows is cut whole. When a draft rule
  fires, the reply is cut back to the last text that passed, and the rule's action is written there: the real
  tool call replaces the narration, or a deflection closes the reply. Narration of a tool nobody asked for is
  dropped instead of turned into a call. The watcher keeps watching after it steers, up to 3 interventions.

Because Rein knows when each tool is relevant, full tool descriptions can leave the generator context and
be injected only on the turn that needs them.

### Generator agnostic

The chat model can be any model that continues a prefilled reply: a local transformers model or an
OpenAI-compatible server such as vLLM (`continue_final_message`). Rein only needs the conversation text. Tested
with Qwen3.5-0.8B and Qwen3.5-4B instruct; tool-call openings exist for the Qwen XML and Hermes JSON formats.

## Architecture

- **State encoding.** The state is encoded once per decision call. Reserved Qwen special tokens are the
  delimiters, and user text is sanitized so it cannot forge them. Every option row reuses that state (attention
  KV plus DeltaNet recurrent and conv states). An incremental cache across turns is not built yet.
- **Readout block.** The backbone is truncated after a chosen block, and the head reads that block. A frozen-probe
  sweep peaks at 56 to 59% of the depth on both sizes. Rein-0.8B reads block 24 of 24 (best on hard negatives in
  the ablation of blocks 11, 14 and 24). Rein-4B reads block 18 of 32, so the last 14 blocks never run.
- **Option rows.** Each option of each question is its own short row that continues from the state cache:
  `[question marker] instructions (+ criteria) [option marker] option text [end marker]`. A row never sees
  another option, and every row of a question has the same positions. This gives exact option-order
  invariance on a hybrid backbone, where DeltaNet layers cannot honor block masks.
- **Set-aware pointer head.** `h_q` is the hidden state at the last instruction token, `h_i` the hidden
  state at option i's end marker. Both are projected to 256 with LayerNorm. One self-attention block
  without positional encoding runs over `{q, o_1..o_K}` with a padding mask, so options compare to each
  other without any order. Then `logit_i = <q', o_i'> / sqrt(256)` and a masked softmax.
- **Output.** A `(N_questions, K_max)` tensor plus an option mask; each question's probabilities sum to 1.
  Scale questions also report the expected level.
- **Loss.** Cross-entropy over each question's options (hard or soft targets) plus a Brier calibration
  term. A temperature is fitted afterwards on a rule-disjoint calibration split. There is no language
  modeling loss and no `generate()` anywhere in the decision path.
- **LoRA.** Rank 16, alpha 32, dropout 0.05 on the attention, DeltaNet and MLP projections, trained jointly
  with the head and merged into the weights in the final checkpoint.

| Piece | What it is | When |
|---|---|---|
| Base | Qwen3.5 text decoder | Downloaded from Hugging Face |
| Adaptation | LoRA trained with the head, then merged | Once, by the project |
| Decision head | Set-aware pointer head with calibration | Once, by the project |
| Rule sheet | Rules of each character or product, in text | Written by the creator, sent per request |
| Decision | State + sheet -> probabilities per option | Every turn |
| Action | Forced tool, locked attribute, reinjected rule, escalation | Executed by the application |

## Data

Rein learns to decide, not to memorize rules, so it is trained on thousands of different rules.

- **Synthetic conversations (the bulk, about 74% of train)**, generated only by the project's Modal teacher,
  Qwen3.5-35B-A3B (`docs/teacher.md`):
  - scenarios with rule sheets;
  - conversations per target rule in many forms: paraphrase, slang, indirect, narration attack, intent spread
    over turns, stale trigger, other languages, PT-EN mixing;
  - hard negatives with the same words and another intent ("her cat is named Mia", "do you have photos of cats?");
  - adversarial bypass attempts;
  - watcher drafts with and without narrated actions;
  - long conversations with distractions, so Rein reads the current situation and not the momentum.

  Every rule of the sheet is labeled for the latest turn by a two-framing judge with a thinking tiebreak.
- **Label audit.** Every synthetic label in every split was then re-checked by Qwen3.8-27B, which never sees the
  existing label. A label changes only when two thinking votes agree; smaller disagreements on score questions
  become soft targets. About 3% of questions were corrected and 1.5% became soft targets.
- **Grounding for the typed questions:** `jaredpalmer/kev-suites` (decision-v7, train only, never
  redistributed), Banking77, CLINC150 and MASSIVE.
- **Languages:** English about 40%, Brazilian Portuguese about 35%, the main languages about 20% (es, fr,
  de, it, ja, zh, ko, ru, ar, hi), PT-EN mixing about 5%. A rule written in English or Portuguese applies
  to conversations in the other languages too.
- **Split by rule:** scenarios are grouped and near-duplicate rules are blocked across splits, so test
  rules are never seen in training. One domain and one category are held out of training entirely. A
  human-reviewed gold set is planned but not built yet.

## Training

Modal H100s, with the fast Gated DeltaNet kernels (flash-linear-attention and causal-conv1d). The 0.8B validated
the pipeline, then the 4B used the same recipe: LoRA and head trained together for 2 epochs, temperature fitted on
the calibration split, then the LoRA merged into the weights at export. Jobs save a resume point every 200 steps
and survive a lost container. Rein-0.8B took about 1 h 20 min and Rein-4B about 1 h 55 min on one H100.

## Evaluation

The suite is part of the product, and any rule-control method can be tested on it:

- **Unseen rules:** accuracy and F1 per type, ECE, Brier, coverage and accuracy at the gate.
- **Words vs intent:** a 2x2 crossing similar and different words with same and different intent.
- **Language matrix:** rule language (EN, PT) x message language, including locales left out of training.
- **Long conversations:** the same rule at turns 5, 20, 50 and 100, with and without Rein in front of the generator.
- **Adversarial paraphrases:** deliberate attempts to get around each rule.
- **Hallucinated action:** watcher precision and recall, and the rate of narrated actions without the
  tool call, with and without Rein.
- **TinyJev-4B and Kev** on the same items.
- **Option-order robustness** and **cost** (latency per decision, generator tokens saved).

## Use cases

- **Game NPCs:** personality kept for dozens of hours, secrets held until the engine says so, lore and
  age rating respected; one rule sheet per NPC.
- **Character platforms:** locked attributes (name, age, backstory) that narration cannot rewrite, and
  mutable ones the user may shape.
- **Tools and agents:** tools called when needed, never narrated instead of executed.
- **Regulated support:** compliance rules written in text, valid on the next deploy, one server for many
  companies, and an audit log of which rules fired with what confidence on every turn.

## Where it runs

A Hugging Face checkpoint that runs on its own: the merged backbone, the head, the config and the tokenizer, plus
`modeling_rein.py` and `rein_control.py`. `AutoModel.from_pretrained(repo, trust_remote_code=True)` loads the
model, `rein.session(sheet, generator=...)` puts a chat model under a rule sheet, and `rein.decide(state,
questions)` returns the raw decisions. A server with decide and session endpoints, and quantized or llama.cpp
builds, come later.

## What goes to Hugging Face

| Repo | State |
|---|---|
| `jmc0/rein-r1-0.8b` | Published (public) |
| `jmc0/rein-decisions` | Published (public): the audited synthetic records only, with the split by rule |
| `jmc0/rein-r1-4b` | Published (public) |
| `jmc0/rein-bench` | Later: the evaluation suite and the e2e scenarios |

They go public after a final review.
