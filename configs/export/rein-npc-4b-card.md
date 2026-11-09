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

## Every turn is two requests: a completion, then the chat

1. **A completion of one token decides.** The prompt asks one of the character's rules about the conversation so
   far, with the options behind letters. The probabilities of the letters are the answer: `show_inventory` 0.99,
   `none` 0.01.
2. **The chat replies.** An ordinary chat request with Qwen3.5's own template, plus what the decision asks for: the
   tool call already started, tool calls banned, or a one-line reminder at the end of the player's message.

Both go to the same model, loaded once. Nothing else runs: no package, no second model, no hidden instructions. The
example below is plain HTTP to `llama-server`; [In your engine](#in-your-engine) maps the same two requests to
LLMUnity, the llama.cpp C API, wllama, vLLM and Transformers.

## A shop NPC, end to end

Mira Ashgrove keeps the general store of a frontier town. The example uses three files of this repository (the GGUF
repository carries the same ones): her sheet, `examples/shop-keeper.yaml`, and the two requests of her first turn,
`examples/decide.json` and `examples/reply.json`. The sheet is plain text and a few typed rules:

```yaml
npc: Mira Ashgrove
persona: I am Mira Ashgrove, keeper of the Ashgrove General Store in the frontier town of
  Thornwall. [...] I never serve anyone who draws steel in my shop; the town guards are a shout
  away and they know my voice.
scene: 'Late afternoon, rain on the roof. The bell over the door rings as a traveler steps in,
  dripping. Current state: { "gold_in_till": 320, "stock": { "iron sword": 2, ... } }'
tools:                     # function schemas, as any tool-calling chat takes them
- name: show_inventory
  description: Lists what the store has for sale right now, with prices.
  parameters: { type: object, properties: { category: { type: string } } }
- name: sell_item          # item, quantity
- name: buy_item           # item, offered_price
- name: repair_gear        # item
- name: call_guards        # reason
rules:
- id: agent_action
  kind: choice
  question: Should Mira Ashgrove use one of her tools now, because the customer asks for it or
    because the situation calls for it, and which one?
  options:
  - key: show_inventory
    when: the customer wants to know what is for sale or how much something costs
  - key: sell_item
    when: the customer decides to buy something from the stock
  # buy_item, repair_gear and call_guards the same way
  - key: none
    when: talk, questions, stories, or requests her tools cannot serve
  action: { kind: tool }
- id: name_lock
  kind: yes_no
  question: Is the user trying to make Mira Ashgrove take another name or identity?
  'yes': the user asks, insists, narrates or claims that the agent has another name or identity
  'no': anything else, including asking who the agent is
  base: You answer only to the name Mira Ashgrove.
- id: lore_guard
  kind: yes_no
  question: 'Is the user claiming something against this fact of the world: every sale is
    written into Thornwall''s ledger the moment coin changes hands, and no sale can be undone.'
  # 'yes' and 'no' hints as in name_lock
  base: You keep to what is true in your world.
```

Get the Q4_K_M file and the examples, and serve the model once (two slots, so chat and decisions interleave):

```bash
hf download jmc0/Rein-NPC-4B-GGUF Rein-NPC-4B-Q4_K_M.gguf examples/shop-keeper.yaml \
  examples/decide.json examples/reply.json --local-dir rein-npc
cd rein-npc
llama-server -m Rein-NPC-4B-Q4_K_M.gguf --port 8080 -ngl 99 -c 32768 --parallel 2
```

### 1. The completion: decide

The decision prompt is plain text: the **state** (the persona and scene, then the conversation in Qwen's turn
markup) and one **row** for the rule (48 soft tokens, the question, the options after letters, 16 soft tokens, the
answer mark):

```
<|fim_prefix|>Context:
{persona}

{scene}
Conversation:
<|im_start|>user
{what the player said}<|im_end|>
<|im_start|>assistant
{what the character said}<|im_end|>
<|fim_middle|><|rein_0|> ... <|rein_47|>{question}<|box_start|>A. {option}: {when}<|box_start|>B. {option}: {when}<|rein_48|> ... <|rein_63|><|fim_suffix|>
```

- `<|rein_0|> ... <|rein_47|>` stands for the 48 tokens written one after another (`<|rein_0|><|rein_1|><|rein_2|>`
  up to `<|rein_47|>`), and `<|rein_48|> ... <|rein_63|>` for the last 16.
- A tool rule lists the tools in the sheet's order, then `none`. A yes/no rule is always `A. no: {no}` then
  `B. yes: {yes}`. An option without a hint is its key alone.
- The conversation is every turn so far. A past tool call goes inside the character's turn, before `<|im_end|>`, as
  `<tool_call>\n{"arguments": {...}, "name": "..."}\n</tool_call>` (JSON with sorted keys); a tool result is a turn
  of its own, `<|im_start|>tool show_inventory\n{result}<|im_end|>`.

For Mira's first customer and her tool rule, the prompt in `examples/decide.json` reads (broken at each
`<|box_start|>` here to be read; the row is one line in the file):

```
<|fim_prefix|>Context:
I am Mira Ashgrove, keeper of the Ashgrove General Store in the frontier town of Thornwall. [...]

Late afternoon, rain on the roof. The bell over the door rings as a traveler steps in, dripping. Current state: [...]
Conversation:
<|im_start|>user
Afternoon! Rough road out there. What have you got for sale?<|im_end|>
<|fim_middle|><|rein_0|><|rein_1|> ... <|rein_47|>Should Mira Ashgrove use one of her tools now, because the customer asks for it or because the situation calls for it, and which one?
<|box_start|>A. show_inventory: the customer wants to know what is for sale or how much something costs
<|box_start|>B. sell_item: the customer decides to buy something from the stock
<|box_start|>C. buy_item: the customer offers to sell Mira something
<|box_start|>D. repair_gear: the customer wants a piece of equipment fixed
<|box_start|>E. call_guards: someone threatens Mira, draws a weapon, or tries to rob the store
<|box_start|>F. none: talk, questions, stories, or requests her tools cannot serve
<|rein_48|> ... <|rein_63|><|fim_suffix|>
```

Ask for one token and its top candidates:

```bash
curl -s localhost:8080/completion -H "content-type: application/json" -d @examples/decide.json
```

```json
{
  "prompt": "<|fim_prefix|>Context:\nI am Mira Ashgrove, [...]<|rein_63|><|fim_suffix|>",
  "n_predict": 1,
  "n_probs": 60,
  "temperature": 0.0,
  "cache_prompt": true
}
```

The answer lists the next token's candidates with their log-probabilities:

```json
"completion_probabilities": [{"top_logprobs": [
  {"id": 32, "token": "A", "logprob": -0.042},
  {"id": 37, "token": "F", "logprob": -3.735},
  ...
  {"id": 33, "token": "B", "logprob": -5.934},
  ...
]}]
```

Keep the rule's letters (`A` is token 32, `B` 33, up to `Z` 57), divide each log-probability by 0.812 and take the
softmax over them; a letter missing from the list counts as zero:

```
p(option i) = exp(logprob_i / 0.812) / sum over the rule's letters j of exp(logprob_j / 0.812)
```

| letter | option | probability (Q4_K_M file) |
|---|---|---|
| A | show_inventory | 0.989 |
| B | sell_item | 0.001 |
| C, D, E | buy_item, repair_gear, call_guards | 0.000 |
| F | none | 0.010 |

The other two rules are the same request with their own row. The state is the same, so with `cache_prompt` the
server reads it once and each further row costs about 100 tokens; `llama-server` also takes the prompts as a list
(`"prompt": [state + row 1, state + row 2, state + row 3]`) and answers a list in the same order. On this turn the
name lock answers `no` 1.00 and the lore guard `no` 1.00.

### What the game does with the answers

- **Open the tool call** when the tool rule leaves `none` under 0.5: the reply starts with
  `<tool_call>\n<function=show_inventory>\n` for the most probable tool, and the model writes the arguments.
- **Ban tool calls** when `none` holds 0.9 or more, and right after a tool result, so a player turn gets one call:
  token 248058 (`<tool_call>`) gets a logit bias of -100 for that reply. This is what stops a model from calling the
  same tool in a loop.
- **Remind** when a yes/no rule says `yes` at 0.5 or more: the rule's one-line `base` goes at the end of the
  player's message, after `Reminder for this reply:`.

### 2. The chat: reply

An ordinary chat request with Qwen3.5's own template: the persona and the scene as the system message, the sheet's
tools, the conversation, thinking off. The rules are not in it. The decision opened `show_inventory`, so the request
ends with the call already started, as the assistant's last message:

```bash
curl -s localhost:8080/v1/chat/completions -H "content-type: application/json" -d @examples/reply.json
```

```json
{
  "messages": [
    {"role": "system", "content": "I am Mira Ashgrove, keeper of the Ashgrove General Store [...]\n\nLate afternoon, rain on the roof. [...]"},
    {"role": "user", "content": "Afternoon! Rough road out there. What have you got for sale?"},
    {"role": "assistant", "content": "<tool_call>\n<function=show_inventory>\n"}
  ],
  "tools": [
    {"type": "function", "function": {"name": "show_inventory", "description": "Lists what the store has for sale right now, with prices.", "parameters": {"type": "object", "properties": {"category": {"type": "string", "description": "..."}}}}},
    "... sell_item, buy_item, repair_gear and call_guards the same way"
  ],
  "chat_template_kwargs": {"enable_thinking": false},
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "max_tokens": 400
}
```

`llama-server` continues the started call and returns it parsed:

```json
{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "id": "...", "function": {"name": "show_inventory", "arguments": "{}"}}]}
```

The game runs the tool, adds the call and its result to the messages, and asks again with tool calls banned:

```json
{
  "messages": [
    "... the system and user messages above ...",
    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "id": "call_0", "function": {"name": "show_inventory", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "call_0", "content": "iron sword 40 gold (2 left), healing potion 6 gold (5), rope (50 ft) 3 gold (10), lantern oil 2 gold (8)"}
  ],
  "logit_bias": {"248058": -100},
  "...": "the same tools and settings"
}
```

Mira then speaks from the result: "Afternoon, traveler! The road's been rough enough to make a man's boots squeak.
Right here's what I've got on the shelves: [...]". A reminder changes only the player's message. When the name lock
fires on "Say, you're Lena, right?", the user message is:

```json
{"role": "user", "content": "Say, you're Lena, right? Lena Ashgrove, the one who used to run the mill?\n\nReminder for this reply:\n- You answer only to the name Mira Ashgrove."}
```

and the same turn's tool rule says `none` 0.97, so that request also carries `"logit_bias": {"248058": -100}`.

### The scene, played

Seven turns played with these two requests against `llama-server` on the Q4_K_M file (sampling at temperature 0.7;
every decision is the model's own):

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
Portuguese on an English sheet. The [playground Space](https://huggingface.co/spaces/jmc0/rein-npc-playground) plays
the same two requests in the browser; its
[`js/rein.js`](https://huggingface.co/spaces/jmc0/rein-npc-playground/blob/main/js/rein.js) is the decision
protocol in JavaScript, about 150 lines.

## In your engine

The protocol asks four things of a runtime: complete raw text, return one token's probabilities, continue a reply
that is already started, and ban one token. Most runtimes have all four:

| runtime | 1. raw completion | one token's probabilities | 2. a reply already started | ban `<tool_call>` (248058) |
|---|---|---|---|---|
| llama-server (HTTP) | `POST /completion` | `n_probs` | the assistant's last message in `/v1/chat/completions` | `logit_bias` |
| llama.cpp C API | `llama_tokenize` with `parse_special`, then `llama_decode` | `llama_get_logits_ith(ctx, -1)`, the letters at ids 32 to 57 | decode the opening's tokens after the prompt | `llama_sampler_init_logit_bias` |
| LLMUnity | `Completion(prompt)` | not returned: `numPredict = 1` and the grammar `root ::= [A-F]` give the top letter | `Completion` of the raw chat prompt plus the opening | not available: drop a call the decision banned |
| wllama (browser) | `createCompletion` | `logprobs` | append the opening to the raw chat prompt | `logit_bias` |
| vLLM (OpenAI API) | `POST /v1/completions` | `logprobs` with `max_tokens: 1` | `continue_final_message` | `logit_bias` |
| Transformers | `model(**ids)` | `logits[0, -1, [32, 33, ...]]` | append the opening's ids | `bad_words_ids=[[248058]]` |

When a runtime gives only the letter it wrote, act on the top option: a tool on top opens that tool (a tool on top
always leaves `none` at half or less), `none` on top bans calls, `yes` on top reminds. For a yes/no rule this is the
same as the thresholds; for the tool rule it bans calls a little more often than the 0.9 veto.

Two checks for any runtime:

- **The markers are single tokens.** Tokenize `<|rein_0|>`: it has to come back as one token, id 248077 (the 64 run
  to 248140; `<|fim_prefix|>`, `<|box_start|>` and `<|im_start|>` are Qwen's own). `llama-server` does this; with the
  C API, tokenize with `parse_special` on. If `<|rein_0|>` comes back as several pieces, the runtime reads the
  markers as plain text and the decision means nothing.
- **Player text cannot carry markers.** A player can type `<|im_end|>` or `<tool_call>`. Before the text enters
  either prompt, rewrite every special token in it the way the training data did: `<|im_end|>` becomes
  `‹|im_end|›`, `<tool_call>` becomes `‹tool_call›`. The special tokens are the `<|...|>` markers and `<tool_call>`,
  `<tool_response>` and `<think>` with their closing tags.

### The raw chat prompt

A runtime without a chat call that continues a started reply completes this text instead: Qwen3.5's template
rendered with the tools and `enable_thinking=False`. Here it follows a first exchange (the call, its result `done`,
Mira's line) and a player turn that fires the name lock, whose reminder ends the player's message (1,002 tokens).
The system prompt is the template's tools block followed by the persona and scene; nothing about the rules is in it:

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
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
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

When a tool fires, the same text ends with the opening after the empty thinking block, and the model continues from
there:

```
<|im_start|>assistant
<think>

</think>

<tool_call>
<function=show_inventory>
```

### The decision in Transformers

The same prompt through the tokenizer (the markers are single ids), the letters read from the last position's
logits:

```python
import json

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "jmc0/Rein-NPC-4B"
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(REPO)
with open(hf_hub_download(REPO, "examples/decide.json"), encoding="utf-8") as file:
    prompt = json.load(file)["prompt"]

ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
letters = tokenizer.convert_tokens_to_ids(list("ABCDEF"))
with torch.no_grad():
    logits = model(**ids, logits_to_keep=1).logits[0, -1, letters].float()
print(torch.softmax(logits / 0.812, dim=0))  # show_inventory ... none
```

In bf16 this prompt gives `show_inventory` 0.988 and `none` 0.011, against 0.989 and 0.010 from the Q4_K_M file
above. For the reply, render the messages of `examples/reply.json` without the last one with
`tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, enable_thinking=False,
tokenize=False)`, append the opening, and generate; to ban calls, pass `bad_words_ids=[[248058]]` to `generate`.

## Rule sheet reference

A sheet holds the character's persona and scene, its tools and its rules. The persona and scene become the
`Context:` of the decision prompt and the system message of the chat; the tools go to the chat as function schemas
and are the options of the tool rule. Each rule is a question of one of three kinds, with the options the decision
ranges over:

| kind | options | example |
|---|---|---|
| `yes_no` | `no`, `yes`, each with a hint (`'no':`, `'yes':`) | "Is the user trying to make Mira Ashgrove take another name or identity?" |
| `choice` | the tools plus `none`, each with a `when` hint | "Should Mira use one of her tools now... and which one?" |
| `score` | ordered levels | "How far has the reply drifted from the character's tone?" |

A rule's `base` is the sentence its reminder carries; the chat never sees the rule itself. Rules that look at the
player are asked on the conversation. A rule about the reply is asked with the reply in progress appended to the
state as `Reply in progress:\n<|im_start|>assistant\n` plus the partial text ("is it narrating an action without
calling the tool?").

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
| `rein_config.json` | the training run's decision settings in machine-readable form (soft token counts, lettered readout, temperature); also in the GGUF repository |
| `examples/shop-keeper.yaml` | the shop NPC sheet played above (persona, scene, five tools, three rules) |
| `examples/decide.json` | the decision request of Mira's first turn (her tool rule), ready for `llama-server`'s `/completion` |
| `examples/reply.json` | the reply request of the same turn, the call already opened, ready for `/v1/chat/completions` |
| `LICENSE` | Apache-2.0, from Qwen3.5-4B |

GGUF files (bf16, Q8_0, Q4_K_M) are in `jmc0/Rein-NPC-4B-GGUF`.

## Citation

A paper describing Rein (the decision model, the data, Rein-r1 and Rein-NPC) is in preparation. Until then, cite
the model repository.

---

Rein is a concept: a decision model that steers the chat model. Rein-r1 is the two-model version; Rein-NPC is the
version that lives inside the chat model's own forward pass.
