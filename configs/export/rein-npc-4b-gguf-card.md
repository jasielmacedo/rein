---
base_model: jmc0/Rein-NPC-4B
base_model_relation: quantized
license: apache-2.0
language:
  - en
  - pt
  - multilingual
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - rein
  - rein-npc
  - gguf
  - llama.cpp
  - typed-decision
  - tool-calling
  - game-npc
  - qwen3.5
---

# Rein-NPC-4B, GGUF

llama.cpp files of [`jmc0/Rein-NPC-4B`](https://huggingface.co/jmc0/Rein-NPC-4B): Qwen3.5-4B with Rein's decision
model inside, as 64 trained rows in the embedding table and the base's own table as a separate output head. One
`llama-server` process chats **and decides**. Every player turn is two requests to it: a completion of one token
decides (`/completion` with `n_probs`), then the chat replies (`/v1/chat/completions`, as with any Qwen3.5-4B). The
main card walks a shop NPC through both, with the raw prompts, the acting rule and a table for other runtimes
(LLMUnity, the llama.cpp C API, wllama, vLLM); this card covers the files and the server. Code:
[`jasielmacedo/rein`](https://github.com/jasielmacedo/rein).

## Files

| file | size | what for |
|---|---|---|
| `Rein-NPC-4B-Q4_K_M.gguf` | 3.07 GB | the usual choice for games on consumer GPUs and laptops; the end-to-end results on the main card ran on this file |
| `Rein-NPC-4B-Q8_0.gguf` | 5.16 GB | near-lossless; 8 GB GPUs and up |
| `Rein-NPC-4B-bf16.gguf` | 9.70 GB | the reference, no quantization loss; 24 GB GPUs |
| `rein_config.json` | 1 KB | the training run's decision settings in machine-readable form (soft token counts, lettered readout, temperature) |
| `examples/shop-keeper.yaml` | 4 KB | the shop NPC sheet of the main card's example: persona, scene, five tools, three rules |
| `examples/decide.json` | 3 KB | the decision request of the sheet's first turn (its tool rule), ready for `/completion` |
| `examples/reply.json` | 4 KB | the reply request of the same turn, the tool call already opened, ready for `/v1/chat/completions` |
| `web/Rein-NPC-4B-Q4_K_M-0000N-of-00004.gguf` | 4 files, 3.07 GB | the Q4_K_M file split for browsers (one file under 2 GB each, `llama-gguf-split`); wllama loads the first shard and fetches the rest |

## In the browser

[The playground Space](https://huggingface.co/spaces/jmc0/rein-npc-playground) runs this file in the visitor's browser through wllama (llama.cpp in WebAssembly, WebGPU when the browser has it): three characters in a small town, the decisions under every reply. It downloads the four `web/` shards once into the browser's cache. On a desktop GPU a turn takes a few seconds for the three decisions plus the reply; without WebGPU the CPU build runs, slowly. Source: [`tools/playground/`](https://github.com/jasielmacedo/rein/tree/main/tools/playground).

## Serve

```bash
hf download jmc0/Rein-NPC-4B-GGUF Rein-NPC-4B-Q4_K_M.gguf examples/decide.json examples/reply.json \
  --local-dir rein-npc
cd rein-npc
llama-server -m Rein-NPC-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 32768 --parallel 2
```

Two or more slots let the chat and the decisions interleave. Qwen3.5-4B has 8 full-attention layers among its 32,
about 32 KB of KV cache per token, so a 32k context costs about 1 GB per slot on top of the weights; the Q4_K_M file
with two 16k slots fits in 6 GB.

## One turn, two requests

**1. Decide.** The prompt is plain text: the state (persona and scene, then the conversation in Qwen's turn markup)
and one row for the rule (48 soft tokens, the question, the options after letters, 16 soft tokens, the answer mark).
Ask for one token with its top candidates:

```bash
curl -s localhost:8080/completion -H "content-type: application/json" -d @examples/decide.json
```

Keep the rule's letters in `completion_probabilities[0].top_logprobs` (`A` is token 32, `B` 33, and so on), divide
each log-probability by 0.812 and take the softmax over them: on this turn `A` (`show_inventory`) carries 0.99 and
`F` (`none`) 0.01. Each rule of the sheet is one such request on the same state; `cache_prompt` keeps the state's
cache between them.

**2. Reply.** The decision opened `show_inventory`, so the chat request ends with the call already started as the
assistant's last message. `llama-server` continues it and returns the call parsed:

```bash
curl -s localhost:8080/v1/chat/completions -H "content-type: application/json" -d @examples/reply.json
```

```json
{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "show_inventory", "arguments": "{}"}}]}
```

When the decision says no tool (`none` at 0.9 or more, or right after a tool result), the chat request carries
`"logit_bias": {"248058": -100}`, which bans `<tool_call>` for that reply; when a yes/no rule fires, its one-line base
goes at the end of the player's message. Both files hold the prompts in full: the soft tokens are `<|rein_0|>` to
`<|rein_63|>`, ids 248077 to 248140, each one token to llama.cpp.

## What quantization does to the decisions

Decisions through llama-server on the quantized file against the same model in PyTorch (bf16), 104 questions from
20 held-out test records:

| file | same option as bf16 | mean largest probability gap | worst |
|---|---|---|---|
| Q4_K_M | 101 of 104 | 0.028 | 0.21 |

Every disagreement is a question the bf16 model itself leaves open (its top option at 0.41, 0.62 and 0.38); on the
90 questions where bf16 puts 0.8 or more on an option, the Q4_K_M file picks the same option 90 times. The chat's
own quality is Qwen3.5-4B's at the chosen quantization. The 64 tokens never appear in the chat: the output head is
the base's untouched table (over 984,371 held-out chat positions their mean probability is 6.5e-8 and they never
enter the top 100).

Made with llama.cpp's stock `convert_hf_to_gguf.py` (bf16) and `llama-quantize` from the safetensors checkpoint. The
embedding table and the output head quantize as `llama-quantize` quantizes any Qwen3.5 (Q6_K at Q4_K_M). The 64
decision tokens are ordinary vocabulary entries to llama.cpp: `<|rein_0|>` to `<|rein_63|>`, ids 248077 to 248140,
each one token.

## License

Apache-2.0, as Qwen3.5-4B. See the main card for training data, results, limits and intended use.
