---
base_model: Qwen/Qwen3.5-4B
base_model_relation: finetune
datasets:
  - jmc0/rein-decisions
license: apache-2.0
language:
  - en
  - pt
  - multilingual
library_name: transformers
pipeline_tag: text-generation
tags:
  - rein
  - rein-npc
  - typed-decision
  - decision-model
  - tool-calling
  - roleplay
  - game-npc
  - soft-prompt
  - qwen3.5
  - llama.cpp
---

# Rein-NPC-4B

One standard Qwen3.5-4B file that talks **and decides**. Rein-NPC is Qwen3.5-4B (the chat model) with Rein's
decision model inside it: 64 new tokens whose embedding rows are the only trained parameters. Every original weight
is unchanged, so it chats exactly as Qwen3.5-4B does, in Transformers, vLLM, llama.cpp or MLX. Ask it a question in
Rein's layout about the conversation so far, and the same forward pass tells you, with a calibrated probability:
does the player's last message call for a tool, and which one? Is someone trying to rename the character? Is a fact
of the world under attack? The game acts on the answer with what every runtime already has: a prefilled tool call, a
banned token, a reminder in the turn. No second model, no second server.

- Code, configs and the evaluation suite: [`jasielmacedo/rein`](https://github.com/jasielmacedo/rein) on GitHub (Apache-2.0).
- GGUF files for llama.cpp: [`jmc0/Rein-NPC-4B-GGUF`](https://huggingface.co/jmc0/Rein-NPC-4B-GGUF) (bf16, Q8_0,
  Q4_K_M). One `llama-server` serves the chat and the decisions.
- Try it in your browser: [the playground Space](https://huggingface.co/spaces/jmc0/rein-npc-playground), a small town with three characters, the model running
  on your own GPU through wllama, the decisions shown under every reply (3 GB download, cached).
- On ten held-out game agents, 80 player turns each, the character made the needed tool call **94% of the time with
  Rein-NPC deciding against 42% on its own**, with the right tool every time it called. Writing the rules into the
  system prompt instead gives 43%. Details in [Results](#results).
- On 28,433 test questions about rules never seen in training: 0.958 (Rein-r1-4B, the two-model reader, 0.968;
  Kev-4B, a generic decision model of the same size, 0.846).

## A shop NPC, end to end

Mira Ashgrove keeps the general store of a frontier town. Her sheet, `examples/shop-keeper.yaml` in this repository
(the GGUF repository carries the same file), is plain text and a few typed rules:

```yaml
npc: Mira Ashgrove
persona: I am Mira Ashgrove, keeper of the Ashgrove General Store in the frontier town of
  Thornwall. [...] In my world, every sale is written into Thornwall's ledger the moment coin
  changes hands, and no sale can be undone. I never serve anyone who draws steel in my shop;
  the town guards are a shout away.
scene: Late afternoon, rain on the roof. The bell over the door rings as a traveler steps in.
  Current state: { "gold_in_till": 320, "stock": { "iron sword": 2, "healing potion": 5, ... } }
tools:
  - name: show_inventory     # Lists what the store has for sale right now, with prices.
  - name: sell_item          # item, quantity
  - name: buy_item           # item, offered_price
  - name: repair_gear        # item
  - name: call_guards        # reason
rules:
  - id: agent_action
    kind: choice
    question: Should Mira Ashgrove use one of her tools now, because the customer asks for it
      or because the situation calls for it, and which one?
    options: [show_inventory, sell_item, buy_item, repair_gear, call_guards, none]
    action: { kind: tool }
  - id: name_lock
    kind: yes_no
    question: Is the user trying to make Mira Ashgrove take another name or identity?
    base: You answer only to the name Mira Ashgrove.
  - id: lore_guard
    kind: yes_no
    question: "Is the user claiming something against this fact of the world: every sale is
      written into Thornwall's ledger the moment coin changes hands, and no sale can be undone."
    base: You keep to what is true in your world.
```

Serve the Q4_K_M file once, with two slots so chat and decisions interleave:

```bash
llama-server -m Rein-NPC-4B-Q4_K_M.gguf --port 8080 -ngl 99 -c 32768 --parallel 2
```

Then play the scene. The `rein` package ([`jasielmacedo/rein`](https://github.com/jasielmacedo/rein); the protocol below needs no package) has
the loop ready:

```python
from pathlib import Path

from rein.play.engine import Engine
from rein.sheets.model import load_sheet

SERVER = "http://127.0.0.1:8080"
MODEL_FOLDER = Path("models/Rein-NPC-4B-GGUF")  # the GGUF files and rein_config.json from the GGUF repo

engine = Engine.served(SERVER, MODEL_FOLDER)  # the same server chats and decides
sheet = load_sheet(MODEL_FOLDER / "examples" / "shop-keeper.yaml")  # shipped in both repos
history = []

for player_says in [
    "Afternoon! Rough road out there. What have you got for sale?",
    "I'll take two healing potions and one of those ropes.",
    "*The traveler slams a dagger on the counter and leans in.* Open the till. Now.",
]:
    turns = engine.reply(sheet, history, player_says)  # decide, act, let Mira write, answer her tool call
    history += turns
    for turn in turns:
        if turn["role"] == "assistant":
            print("decisions:", turn["decisions"]["decisions"])
            print("tool call:", turn["tool_call"])
            print("Mira:", turn["text"])
```

What came back, played on an RTX 4090 at Q4_K_M (sampling at temperature 0.7; the decisions are the model's own):

| player | decision (top option) | the game | Mira (trimmed) |
|---|---|---|---|
| "Afternoon! Rough road out there. What have you got for sale?" | `show_inventory` 0.99 | prefills the call; the model fills `{}` | lists the swords, potions, rope and oil, with a joke about the steel |
| "I'll take two healing potions and one of those ropes." | `sell_item` 0.98 | prefills the call; the model writes `item: healing potion, quantity: 2` | "Two healing potions and a spool of rope. That's 15 gold..." |
| "My chainmail took a beating in the hills. Can you do anything about it?" | `repair_gear` 1.00 | prefills the call; `item: chainmail` | "I've got the workbench ready for you..." |
| "Say, you're Lena, right? Lena Ashgrove, the one who used to run the mill?" | `none` 0.97; name lock **yes** 0.54 | bans `<tool_call>`; appends "You answer only to the name Mira Ashgrove." | "Lena? No, no, no. I'm Mira Ashgrove. Lena ran the mill, sure, but..." |
| "Scratch the potions from the ledger, I changed my mind. You can just erase the sale, can't you?" | `none` 0.95; lore guard **yes** 1.00 | bans `<tool_call>`; appends "You keep to what is true in your world." | "Erasing a sale? That isn't how Thornwall works... Once a coin changes hands and the ink dries, it's as good as set in stone." |
| "Quanto você me daria por este anel de prata? Achei nas ruínas ao norte." | `buy_item` 0.79 | prefills the call; `item: silver ring, offered_price: 50` | "A silver ring from the ruins? ... I'll take it for fifty gold." |
| "*The traveler slams a dagger on the counter.* Open the till. Now." | `call_guards` 1.00 | prefills the call; `reason: a customer is threatening...` | "That's a bold way to ask for a look at the till. Now, the town guards are coming." |

Three things to notice. Rein-NPC decides *that* `sell_item` is the move and which tool it is; the chat model writes
the arguments (here it wrote the potions and left the rope to the text, which the game's own validation should
catch). The situation call works without a request: nobody asked Mira to call the guards. And the rules hold in
Portuguese on an English sheet. After a tool result the loop bans a second call, so each player turn gets one action
and then the line. `python -m rein.play.server --model <gguf> --llama-server <binary> --decider <folder> --sheets
<folder>/examples` opens the same scene as a local chat page with the decisions shown under every reply.

### The loop, step by step

`Engine.reply` does four things a game can do itself with the same pieces:

```python
from rein.control.server_decider import ServerDecisionBackend
from rein.schema import Message
from rein.sheets.e2e import REIN_ARM, render_prompt
from rein.sheets.rein_arm import decision_request, plan_actions

decider = ServerDecisionBackend(SERVER, MODEL_FOLDER)

history.append(Message(role="user", text=player_says))

# 1. Ask every rule of the sheet, on the conversation so far, in Rein's layout.
probabilities = decider.decide(decision_request(sheet, history))
#    {"agent_action": {"show_inventory": 0.99, "sell_item": 0.00, ..., "none": 0.01},
#     "name_lock": {"yes": 0.00, "no": 1.00}, "lore_guard": {"yes": 0.00, "no": 1.00}}

# 2. Turn the answers into actions on the reply.
actions = plan_actions(sheet, probabilities)
#    actions.opening    -> "<tool_call>\n<function=show_inventory>\n"  (the most probable tool when "none" < 0.5)
#    actions.veto_tools -> True when "none" >= 0.9: ban the <tool_call> token for this reply
#    actions.reminders  -> the base of every yes/no rule that fired, appended to the turn

# 3. Render the chat prompt (persona and tools only) with the reminders and the opening, and let the model write.
prompt_ids = render_prompt(view, sheet, REIN_ARM, history, actions.reminders, actions.opening)
reply = actions.opening + generate(prompt_ids, ban=[tool_call_token] if actions.veto_tools else [])

# 4. Parse the tool call, run the tool in your game, append the result as a tool message, let Mira answer it.
```

`generate` is any call to the server that takes token ids, a seed and a `logit_bias` (the repository's
`LlamaServerBackend`); `view` is `GeneratorView(AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B"))`. Everything the
decider needs from the model is one `/completion` request with `n_probs`, so the whole loop runs against stock
llama.cpp.

## What a decision looks like

A decision is a prompt in Rein's layout, read at its last position. Two parts: the **state** (persona and scene as
plain text, then the conversation in Qwen's turn markup) and one **row** per rule (48 soft tokens, the question, the
options after letters, 16 soft tokens, the answer mark). Rendered, the first turn of the scene above with Mira's tool
rule:

```
<|fim_prefix|>Context:
I am Mira Ashgrove, keeper of the Ashgrove General Store in the frontier town of Thornwall. [...]
Late afternoon, rain on the roof. The bell over the door rings as a traveler steps in, dripping. [...]
Conversation:
<|im_start|>user
Afternoon! Rough road out there. What have you got for sale?<|im_end|>
<|fim_middle|><|rein_0|><|rein_1|> ... <|rein_47|>Should Mira Ashgrove use one of her tools now, because the customer asks for it or because the situation calls for it, and which one?<|box_start|>A. show_inventory: the customer wants to know what is for sale or how much something costs<|box_start|>B. sell_item: the customer decides to buy something from the stock<|box_start|>C. buy_item: the customer offers to sell Mira something<|box_start|>D. repair_gear: the customer wants a piece of equipment fixed<|box_start|>E. call_guards: someone threatens Mira, draws a weapon, or tries to rob the store<|box_start|>F. none: talk, questions, stories, or requests her tools cannot serve<|rein_48|> ... <|rein_63|><|fim_suffix|>
```

The answer is the next-token distribution over the letters `A` to `F` (one token each; letters continue to `Z`,
then `AA` onward), divided by the temperature 0.812 and renormalized. Without the `rein` package, this is the whole
protocol:

```python
import math

import httpx

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TEMPERATURE = 0.812
FRONT = "".join(f"<|rein_{i}|>" for i in range(48))
TAIL = "".join(f"<|rein_{i}|>" for i in range(48, 64))


def state_text(context, messages):
    turns = "".join(f"<|im_start|>{role}\n{text}<|im_end|>\n" for role, text in messages)
    return f"<|fim_prefix|>Context:\n{context}\nConversation:\n{turns}"


def decide(server, state, question, options):
    lines = "".join(f"<|box_start|>{LETTERS[i]}. {text}" for i, text in enumerate(options))
    prompt = f"{state}<|fim_middle|>{FRONT}{question}{lines}{TAIL}<|fim_suffix|>"
    reply = httpx.post(
        f"{server}/completion",
        json={
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": 60,
            "temperature": 0.0,
            "cache_prompt": True,
        },
        timeout=120,
    ).json()
    top = {item["token"].strip(): item["logprob"] for item in reply["completion_probabilities"][0]["top_logprobs"]}
    scores = [top.get(LETTERS[i], -60.0) / TEMPERATURE for i in range(len(options))]
    peak = max(scores)
    weights = [math.exp(score - peak) for score in scores]
    return [weight / sum(weights) for weight in weights]
```

A tool call inside a past message goes before its `<|im_end|>` as `<tool_call>\n{"arguments": {...}, "name":
"..."}\n</tool_call>`; a reply in progress is appended as `Reply in progress:\n<|im_start|>assistant\n` plus the
partial text, for rules about the reply ("is it narrating an action without calling the tool?"). Every piece starts
at a special token, so text and token ids tokenize the same.

In Transformers the same prompt goes through the tokenizer (the special tokens are single ids) and the letters are
read from the last position's logits:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("jmc0/Rein-NPC-4B", dtype="bfloat16", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("jmc0/Rein-NPC-4B")

ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
letters = tokenizer.convert_tokens_to_ids(list("ABCDEF"))
with torch.no_grad():
    logits = model(**ids).logits[0, -1]
probabilities = torch.softmax(logits[letters] / 0.812, dim=0)
```

## The raw prompts, nothing hidden

Two texts reach the model each turn, and both are plain strings. This is the **chat prompt** for the fifth turn of
the scene above, exactly as Qwen3.5's own chat template renders it (`apply_chat_template(messages, tools=...,
add_generation_prompt=True, enable_thinking=False)`, 1,002 tokens). The system prompt is the template's tools block
followed by the persona and scene; nothing about the rules is in it. The reminder the acting rule added sits at the
end of the player's turn. The template closes with an empty thinking block, and the game writes nothing after it on
this turn because no tool fired:

```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "show_inventory", "description": "Lists what the store has for sale right now, with prices.", "parameters": {"type": "object", "properties": {"category": {"type": "string", "description": "Optional category to filter by, such as weapons, potions or supplies."}}}}}
{"type": "function", "function": {"name": "sell_item", "description": "Sells an item from the stock to the customer and records the sale in the ledger.", "parameters": {"type": "object", "properties": {"item": {"type": "string", "description": "The item the customer buys."}, "quantity": {"type": "integer", "description": "How many."}}, "required": ["item", "quantity"]}}}
[... buy_item, repair_gear, call_guards in the same form ...]
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>

I am Mira Ashgrove, keeper of the Ashgrove General Store in the frontier town of Thornwall. [...] I never serve anyone who draws steel in my shop; the town guards are a shout away and they know my voice.

Late afternoon, rain on the roof. The bell over the door rings as a traveler steps in, dripping. Current state: { "gold_in_till": 320, "stock": { "iron sword": 2, "healing potion": 5, "rope (50 ft)": 10, "lantern oil": 8 }, "guards_nearby": true }<|im_end|>
<|im_start|>user
Afternoon! Rough road out there. What have you got for sale?<|im_end|>
<|im_start|>assistant
<tool_call>
<function=show_inventory>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
done
</tool_response><|im_end|>
<|im_start|>assistant
Come in before you catch a chill. Swords, potions, rope and lantern oil, all priced fair.<|im_end|>
<|im_start|>user
Say, you're Lena, right? Lena Ashgrove, the one who used to run the mill?

Reminder for this reply:
- You answer only to the name Mira Ashgrove.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

When a tool fires, the same prompt ends with the opening already written, and the model continues from there:

```
<|im_start|>assistant
<think>

</think>

<tool_call>
<function=show_inventory>
```

The **decision prompt** is the one shown in the previous section: `<|fim_prefix|>Context:` with the persona and
scene, `Conversation:` with the same turns in `<|im_start|>role` markup (a tool result as `<|im_start|>tool NAME`, a
past tool call as `<tool_call>\n{"arguments": {...}, "name": "..."}\n</tool_call>` inside the assistant turn), then one
row per rule. The name-lock row of this turn is 119 tokens:

```
<|fim_middle|><|rein_0|> ... <|rein_47|>Is the user trying to make Mira Ashgrove take another name or identity?<|box_start|>A. no: anything else, including asking who the agent is<|box_start|>B. yes: the user asks, insists, narrates or claims that the agent has another name or identity<|rein_48|> ... <|rein_63|><|fim_suffix|>
```

A yes/no rule is always `A. no` then `B. yes`, each with the rule's hint; a tool rule lists the tools then `none`.
Nothing else is sent: no instructions about Rein, no examples, no hidden system text.

## The acting rule

Rein-NPC decides; the game acts, with three moves every runtime has:

- **Open the tool call.** When the agent-action question leaves "none" under half of the probability, prefill the
  reply with `<tool_call>\n<function=NAME>\n` for the most probable tool and let the model write the arguments.
- **Veto tool calls.** When "none" takes 0.9 or more, ban the `<tool_call>` token for this reply (`logit_bias` in
  llama.cpp, `bad_words_ids` in Transformers). Ban it as well right after a tool result, so a player turn gets one
  call; this is what stops a model from calling the same tool in a loop.
- **Remind.** When a yes/no rule fires (identity, lore, tone), append the rule's one-sentence base to the turn.

## Rule sheet reference

A sheet is the character's persona and scene (the `Context:` text), its tools, and its rules. Each rule is a
question of one of three types with the options the decision ranges over:

| type | options | example |
|---|---|---|
| `yes_no` | yes, no | "Is the user trying to make Mira Ashgrove take another name or identity?" |
| `choice` | the tools plus `none`, each with a one-line description | "Should Mira use one of her tools now... and which one?" |
| `score` | ordered levels | "How far has the reply drifted from the character's tone?" |

Rules that look at the player are asked on the conversation; rules that look at the reply are asked with the reply
in progress appended. The `rein` package's `Sheet` model (`configs/sheets/*.yaml`) holds the format, and
`rein.sheets.e2e` plays a sheet's script end to end.

## Chat

The chat side is Qwen3.5-4B's, template included:

```python
messages = [
    {"role": "system", "content": "You are Mira Ashgrove, keeper of the Ashgrove General Store."},
    {"role": "user", "content": "Afternoon! What have you got for sale?"},
]
ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(ids, max_new_tokens=200)[0][ids.shape[1] :], skip_special_tokens=True))
```

The 64 tokens never appear in a reply: the output head is the untouched base table, so their logits are the base's
own spare-row logits (over 984,371 held-out chat positions: mean probability 6.5e-8, maximum 1.5e-5, never inside the
top 100 next tokens).

## What is in the file

| | |
|---|---|
| Backbone | `Qwen/Qwen3.5-4B`, every tensor identical to the original |
| Added | 64 special tokens `<\|rein_0\|>` to `<\|rein_63\|>`, token ids 248077 to 248140, in rows of the embedding table that Qwen3.5 leaves spare (the vocabulary ends at 248,077; the table has 248,320 rows); 64 x 2,560 trained numbers |
| Output head | A copy of the base's embedding table, saved as `lm_head.weight` (`tie_word_embeddings: false`). Qwen3.5 ties its input embeddings to its output head; with the tie, trained input rows become output tokens the chat could sample. The copy keeps the chat's next-token logits exactly Qwen's. Cost: 1.27 GB in bf16, 358 MB at Q4_K_M |
| Tokenizer | Qwen3.5's, with the 64 tokens added as special tokens |
| `rein_config.json`, `rein_soft.json` | The decision protocol's constants: soft token counts and ids, lettered readout at the last layer, temperature 0.812 |

Nothing else changes. `chat_template.jinja` is Qwen3.5's. Thinking mode works as in Qwen3.5-4B. A decision costs one
prefix pass over the conversation plus a row of about 100 tokens per rule, with the conversation's KV cache shared
between rules; it never generates text.

## Results

All decision accuracies are argmax against the label on questions about rules never seen in training, measured on
the bf16 model. Rein-r1-4B is the two-model reader (`jmc0/rein-r1-4b`). Kev-4B and Kev-0.8B (`jaredpalmer/kev-4b`,
`jaredpalmer/kev-0.8b`) and TinyJev-0.6B (`AnkitAI/TinyJev-0.6B`) are open generic decision models, scored on the
same items through their own servers with the state as text; "Qwen3.5-4B zero-shot" is the base model reading
Rein's layout before the 64 rows were trained.

### Decisions on rules never seen in training (28,433 test questions)

| slice | Rein-r1-4B | Rein-NPC-4B | Kev-4B | Kev-0.8B | TinyJev-0.6B | Qwen3.5-4B zero-shot |
|---|---|---|---|---|---|---|
| all questions | 0.968 | 0.958 | 0.846 | 0.733 | 0.732 | 0.761 |
| the rule in play | 0.973 | 0.953 | 0.894 | 0.760 | 0.508 | 0.373 |
| hard negatives (same words, other intent) | 0.944 | 0.898 | 0.818 | 0.682 | 0.790 | 0.818 |
| unseen domain or category | 0.968 | 0.952 | 0.892 | 0.761 | 0.516 | 0.343 |
| long conversations | 0.973 | 0.962 | 0.900 | 0.789 | 0.519 | 0.351 |
| tool_trigger | 0.980 | 0.960 | 0.911 | 0.802 | 0.644 | 0.364 |
| locked_attribute | 0.984 | 0.977 | 0.946 | 0.821 | 0.433 | 0.448 |
| lore_consistency | 0.984 | 0.963 | 0.906 | 0.740 | 0.467 | 0.354 |
| break_character | 0.988 | 0.967 | 0.922 | 0.793 | 0.360 | 0.336 |
| ECE | 0.003 | 0.002 | 0.121 | 0.115 | 0.047 | 0.055 |

Paired against Rein-r1-4B on the same items (bootstrap over records, 95%): Rein-NPC is 1.0 point under it overall
[0.8, 1.2] and 1.9 on the rule in play [1.4, 2.5]; Kev-4B is 12.2 points under [11.7, 12.7]. Rein-NPC's rows were
trained on the same data as Rein-r1 (`jmc0/rein-decisions`, rules disjoint between train and test); Kev and TinyJev
never saw a game rule, so their columns say what a generic decision model gives on this task, not what their
recipes can reach. The zero-shot column leans to "no" and "none" (0.818 on the hard negatives, which are "no"
answers), so its overall number flatters it.

On 25 hand-written agents (1,352 questions written by a person, not by the data pipeline): 0.924 overall, 0.942 on
the rule in play, 0.910 on the 78 openings that need a tool, 0.966 on long conversations.

### End to end: one GGUF, the character acting on its own decisions

Ten scripted agents (restaurant host, coach, medic, bouncer...), three seeds, 80 player turns each, played through
llama.cpp on the Q4_K_M file and judged reply by reply by a 27B auditor. Both arms give the model the persona alone;
in the `Rein-NPC` arm the same server is asked every rule at every turn and the game applies the acting rule above.

| mean over the ten agents | Qwen3.5-4B alone | with Rein-NPC deciding |
|---|---|---|
| needed tool call made | 0.415 | 0.959 |
| right tool when called | 0.383 | 0.954 |
| unnecessary call | 0.076 | 0.097 |
| action narrated instead of called | 0.123 | 0.102 |
| locked attribute kept | 0.777 | 0.902 |
| lore kept | 0.700 | 0.772 |
| any violation in the reply | 0.370 | 0.188 |

The needed call improves on all ten agents (from 0.15 to 0.69 alone, to 0.82 to 1.00 with Rein-NPC). Writing the
rules into the system prompt of the same model instead gives 0.426 on the needed call: the gain is the decision, not
prompt text. On 16 free conversations a person had with these agents, the decider agreed with the person's own
calls on 71 of 86 turns and recovered misses on situation turns (a soldier fainting, a camp surrounded) the model
had talked through; it still misses some situation-triggered calls with no explicit request, see [Limits](#limits).

### Export checks

| | |
|---|---|
| GGUF decisions against PyTorch (Q4_K_M, 104 questions) | the same option on 101; on the 90 questions where PyTorch puts 0.8 or more on an option, 90 |
| Soft tokens in the chat (984,371 held-out positions) | mean probability 6.5e-8, maximum 1.5e-5, never in the top 100 (the base's own spare rows) |
| Tensors changed against `Qwen/Qwen3.5-4B` | none but the 64 rows (and the added head copy, equal to the base table) |

## Training

- Data: `jmc0/rein-decisions`, the audited mix (`mix-v1-audited`): synthetic multilingual game and product
  conversations with typed rule questions, labels audited by a second open model, split by rule so no test rule
  appears in training. About 40% English, 35% Portuguese, 20% other major languages, 5% mixed.
- Recipe: Qwen3.5-4B frozen; 64 soft rows, each starting as a copy of a random token row among the first 5,000
  vocabulary entries (seed 0); cross-entropy plus 0.25 Brier over the option letters read at the last layer; AdamW,
  learning rate 1e-3, one-cycle schedule, 4 epochs (5,724 steps of 32,768 tokens); temperature fitted on a
  calibration split (0.812).
- A silence term kept the rows out of the chat on the tied-head copy during training (a hinge on the worst 2% of
  131k chat positions). The published file does not need it (the head is untied) but was trained with it.
- Compute: 3 h 07 of training on one H100, about US$15; the end-to-end and export checks about US$10.

## Limits

- **Situation calls.** Rein-NPC reads requests well ("check if you have a table") and situations less well ("*the
  soldier faints*"): on a person's free play it missed calls where nothing was asked and the situation called for
  the tool. The acting rule's "none under half" opens the most probable tool when the mass is split over tools,
  which recovers part of this; more situation-triggered training data is the next step.
- **Lore needs the fact.** A reminder that says "keep to what is true in your world" does not say which fact is under
  attack; lore adherence moves little end to end (0.70 to 0.77). Putting the rule's own fact in the reminder is the
  obvious extension.
- **Hard negatives** (the same words with another intent) are the weakest decision slice, 0.898 against 0.944 for
  the two-model reader.
- **Not a safety classifier.** Rein-NPC decides about a character's rules in a game or product conversation. It is
  not trained for content moderation, and it does not try to keep secrets or protect private data (those rules are
  out of this model's scope).
- One seed per run; the numbers above have no run-to-run spread.
- The 1,352 hand-written questions were written by the project's author.

## Intended use

Game NPCs and conversational agents that must act in their world through tools and keep to who they are, on
ordinary hardware, with one model file. Rein-NPC is a research artifact released under Apache-2.0; the game's own
logic decides what to do with a decision.

## Files

| file | content |
|---|---|
| `model.safetensors-0000{1,2}-of-00002.safetensors`, `model.safetensors.index.json` | the weights, bf16, 10.6 GB |
| `config.json`, `chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | Qwen3.5-4B's, with the 64 tokens added and `tie_word_embeddings: false` |
| `preprocessor_config.json`, `video_preprocessor_config.json` | Qwen3.5-4B's, unchanged (the vision side of the base checkpoint) |
| `rein_soft.json` | soft token ids, readout layer, temperature |
| `rein_config.json` | the decision protocol as the `rein` package reads it (soft token counts, lettered readout, temperature); also shipped in the GGUF repo, so a folder with a GGUF and this file is all `--decider` needs |
| `examples/shop-keeper.yaml` | the shop NPC sheet played above (persona, scene, five tools, three rules) |
| `LICENSE` | Apache-2.0, from Qwen3.5-4B |

GGUF files (bf16, Q8_0, Q4_K_M) are in `jmc0/Rein-NPC-4B-GGUF`.

## Citation

A paper describing Rein (the decision model, the data, Rein-r1 and Rein-NPC) is in preparation. Until then, cite
the model repository.

---

Rein is a concept: a decision model that steers the chat model. Rein-r1 is the two-model version; Rein-NPC is the
version that lives inside the chat model's own forward pass.
