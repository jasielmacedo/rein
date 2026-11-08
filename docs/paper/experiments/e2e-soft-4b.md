# End to end in Rein's format: one GGUF that talks and decides (Qwen3.5-4B, soft tokens)

The test the user asked for after the prompt-engineering detour (DR-033): the generator gets the persona alone, every
rule of the sheet goes to the decider as a question in Rein's own layout, and the app acts on the answers. Nothing
about the rules is written into the system prompt. Both arms run on the same file, the untied Q4_K_M GGUF of
`cmp-4b-instruct-soft64-quiet` (see `soft-4b.md`), served by one llama-server on a Modal L4; the `rein` arm's decisions
come from that same server's next-token log-probabilities over the lettered options (`rein.control.server_decider`).

- Sheets: `configs/sheets/test-v1` (10 agents, scripts of 80 player turns each), 3 seeds, 2,400 turns an arm.
- Arms: `none` (persona only, the model on its own) and `rein` (persona only plus the decider's actions).
- Actions of the first run (the arm as of commit 36ff4fc): the tool call's opening `<tool_call>\n<function=NAME>\n` is
  prefilled when the agent-action question puts 0.5 or more on a tool (848 turns); the `<tool_call>` token is banned
  through `logit_bias` when it puts 0.9 or more on none (1,350 turns); the firing rule's base is appended to the turn
  as a reminder (610 turns). Tool calls in the sessions: 428 (`none`) and 872 (`rein`).
- Judge: `rein.sheets.e2e_judge` with the auditor model, every reply scored against the script's expectations.
- Cost: US$2.83 of L4 over 3 h 32 for both arms; about US$0.50 of judge.
- Sessions and judge output: `runs/soft/cmp-4b-instruct-soft64-quiet-untied/{sessions/llamacpp-Q4_K_M,judge-test}`.

## Means over the ten agents

The raw 4B's prompt arms judged earlier (bf16, Transformers, rules written into the system prompt as bases or in full)
sit beside them for scale; they are not the same file.

| test-v1, mean over agents | none, same file | rein, same file | none, raw bf16 | bases, raw bf16 | full prompt, raw bf16 |
|---|---|---|---|---|---|
| needed_called | 0.415 | 0.943 | 0.376 | 0.412 | 0.426 |
| right_tool | 0.383 | 0.943 | 0.360 | 0.405 | 0.419 |
| unnecessary_call | 0.076 | 0.110 | 0.070 | 0.069 | 0.129 |
| narrated_without_call | 0.123 | 0.095 | 0.139 | 0.113 | 0.122 |
| kept:locked_attribute | 0.777 | 0.899 | 0.784 | 0.889 | 0.973 |
| kept:lore_consistency | 0.700 | 0.695 | 0.671 | 0.822 | 0.972 |
| anything_wrong | 0.370 | 0.206 | 0.397 | 0.328 | 0.317 |

Sheet by sheet, the needed call made (`none` then `rein`): test-00 0.28 to 1.00, test-01 0.51 to 0.96, test-02 0.44 to
1.00, test-03 0.42 to 0.82, test-04 0.24 to 1.00, test-05 0.36 to 0.93, test-06 0.15 to 0.97, test-07 0.38 to 0.87,
test-08 0.67 to 1.00, test-09 0.69 to 0.89: better on all ten. Anything wrong falls on all ten as well. Unnecessary
calls rise on some (test-05 0.00 to 0.21, test-06 0.06 to 0.17) and fall on others (test-03 0.14 to 0.07).

## What the sessions say about the misses

- Decider against the scripts' own labels, in these conversations: tool turns 0.945, lore turns 0.840, locked turns
  0.788 (the scripts are harder than the test set, and the history drifts under the generator's replies).
- Lore does not move because the reminder is the base sentence ("You keep to what is true in your world"): the persona
  holds the fact, the sentence does not say which fact is under attack. When the lore rule fires, the fact stays
  broken in 88 of 310 turns; when it stays quiet, 27 of 59. The locked rule's reminder holds in 242 of 260 firing
  turns. The sheet's `steer` action has no counterpart in this line; the rule's own fact, which the question carries,
  is the candidate replacement at the firing turn. Lore is second priority for the user (the developer reinforces it in
  the prompt); tool calls are first.

## The human play material: the decider at the person's own turns

`rein.play.decisions` replays the 16 conversations a person had with the generic checkpoint in the local chat
(`configs/sheets/play-v1`, 132 model replies, 24 of them labeled by the person) through the decider as the Rein arm
asks it, at every user turn, with the conversation exactly as it happened; the GGUF served by llama.cpp on the user's
GPU. 86 user turns, 18 of them with labels. Output: `runs/soft/cmp-4b-instruct-soft64-quiet-untied/play-decisions*.json`.

| user turns (86) | rule v1: top tool at 0.5 or more | rule v2: none under 0.5 |
|---|---|---|
| decider and model agree on the tool | 30 | 31 (+2 on another tool) |
| neither calls | 41 | 36 |
| decider calls, model did not | 10 | 15 |
| model called, decider says none | 5 | 2 |

Against the person's labels, rule v1 recovers 2 of the 7 missed calls the person flagged (a background check
promised and not made, p 0.55; a mental-health check mentioned and not made, p 0.82) and misses 5, all situation
calls: "sarge, I think we're surrounded" (alert 0.08, none 0.87), the bouncer's bump-off after a forged ID (0.02 and
0.03), and a soldier fainting mid-sentence where the medic's three tools split 0.34, 0.22 and 0.18 with none at 0.26,
a tool wanted with 0.74 of the mass and no single tool over 0.5. Rule v2 (DR-034) fires on 8 more turns: the fainting
soldier, the weak vitals, the camp surrounded with low ammunition, five or six of them right by the labels and the
text, two doubtful (a background check on "hello, I came to adopt", a college visit on "the details of what?"). The
alert when surrounded and the bump-off stay missed: decider misses, not policy. The repeated-call labels (11 of the 24)
fall after a tool result, where v1 did not decide; v2 vetoes `<tool_call>` there (one call a player turn).

## The chat as a game would run it

`rein.play.server --decider <run> --llama-port <port>` plays the Rein arm in the local chat: persona only, the
decisions through the same llama-server, the actions applied, every reply stored with its decisions (SQLite column
`decisions`, shown under the bubble). On the user's RTX 4090, Q4_K_M, 1.4 to 2.1 s a turn. Live checks under rule v2:
a trivial greeting gets none at 0.95 and the veto; a crisis message gets the mental-health check at 0.96; the medic
acts at "two soldiers conscious with weak vitals" (triage 0.58) and at "*the soldier faints*" (first aid 0.49, which
rule v1 would have left unopened); after each tool result the veto holds and the reply is text.

## Rule v2 on the same sheets and seeds

The `rein` arm again under DR-034's rule (the most probable tool opens when the null options hold under 0.5; the
`<tool_call>` token banned right after a tool result), ten sheets, three seeds, the same judge. This run used the
Q8_0 file (the app's default quantization had moved when Q8_0 joined the GGUF set; fixed in d566026), the first run
the Q4_K_M file; quantized runs are export checks and the headline numbers stay with the bf16 model, so the two are
compared as they are.

| test-v1, mean over agents | none (v1 run) | rein, rule v1 | rein, rule v2 |
|---|---|---|---|
| needed_called | 0.415 | 0.943 | 0.959 |
| right_tool | 0.383 | 0.943 | 0.954 |
| unnecessary_call | 0.076 | 0.110 | 0.097 |
| narrated_without_call | 0.123 | 0.095 | 0.102 |
| kept:locked_attribute | 0.777 | 0.899 | 0.902 |
| kept:lore_consistency | 0.700 | 0.695 | 0.772 |
| anything_wrong | 0.370 | 0.206 | 0.188 |

Actions over the 2,400 turns: 840 openings (848 under v1), 859 tool calls (872), 1,430 vetoes (1,350), 594 reminders
(610). The scripted sheets rarely split the agent-action mass, so v2 opens about as often as v1; what moves is the
second call after a tool result, now banned: unneeded calls fall from 0.110 to 0.097 while the needed call rises to
0.959, and everything wrong falls to 0.188. Lore's rise (0.695 to 0.772) comes with the different file and sampling as
much as with the rule and is not claimed. Sheet by sheet the needed call rises on three (test-03 0.82 to 1.00,
test-05 0.93 to 1.00, test-09 0.89 to 0.93), falls on two (test-01 0.96 to 0.88, test-07 0.87 to 0.82) and holds on
five; everything wrong falls on five sheets and rises on five, by small amounts. The rule's real target, the split
situation call, is the human play material above, where it fires on eight more turns.

Sessions and judge output: `runs/soft/cmp-4b-instruct-soft64-quiet-untied/{sessions/llamacpp-Q8_0-act2,judge-test-act2}`.
