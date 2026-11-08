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
`llama-server` process chats **and decides**: the chat through `/v1/chat/completions` as with any Qwen3.5-4B, a
decision through one `/completion` request with `n_probs`. The main card has the shop-NPC example played end to
end, the decision protocol and the results; this card covers the files and the server. Code:
[`jasielmacedo/rein`](https://github.com/jasielmacedo/rein).

## Files

| file | size | what for |
|---|---|---|
| `Rein-NPC-4B-Q4_K_M.gguf` | 3.07 GB | the usual choice for games on consumer GPUs and laptops; the end-to-end results on the main card ran on this file |
| `Rein-NPC-4B-Q8_0.gguf` | 5.16 GB | near-lossless; 8 GB GPUs and up |
| `Rein-NPC-4B-bf16.gguf` | 9.70 GB | the reference, no quantization loss; 24 GB GPUs |
| `rein_config.json` | 1 KB | the decision protocol for the `rein` package (soft token counts, lettered readout, temperature): a folder with a GGUF and this file is all `rein.play.server --decider` and `Engine.served` need |
| `examples/shop-keeper.yaml` | 4 KB | the shop NPC sheet of the main card's example: persona, scene, five tools, three rules |
| `web/Rein-NPC-4B-Q4_K_M-0000N-of-00004.gguf` | 4 files, 3.07 GB | the Q4_K_M file split for browsers (one file under 2 GB each, `llama-gguf-split`); wllama loads the first shard and fetches the rest |

## In the browser

[The playground Space](https://huggingface.co/spaces/jmc0/rein-npc-playground) runs this file in the visitor's browser through wllama (llama.cpp in WebAssembly, WebGPU when the browser has it): three characters in a small town, the decisions under every reply. It downloads the four `web/` shards once into the browser's cache. On a desktop GPU a turn takes a few seconds for the three decisions plus the reply; without WebGPU the CPU build runs, slowly. Source: [`tools/playground/`](https://github.com/jasielmacedo/rein/tree/main/tools/playground).

## Serve

```bash
llama-server -m Rein-NPC-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 32768 --parallel 2
```

Two or more slots let the chat and the decisions interleave. Qwen3.5-4B has 8 full-attention layers among its 32,
about 32 KB of KV cache per token, so a 32k context costs about 1 GB per slot on top of the weights; the Q4_K_M file
with two 16k slots fits in 6 GB.

## Ask a decision

The prompt is the state (persona and scene, then the conversation in Qwen's turn markup) and one row per rule: 48
soft tokens, the question, the options after letters, 16 soft tokens, the answer mark. Read the letters in
`top_logprobs`, divide by 0.812, renormalize. Mira, the shop keeper of the main card, on her first customer:

```bash
curl -s localhost:8080/completion -H "content-type: application/json" -d @- <<'JSON'
{
  "prompt": "<|fim_prefix|>Context:\nI am Mira Ashgrove, keeper of the Ashgrove General Store in the frontier town of Thornwall. [...]\nConversation:\n<|im_start|>user\nAfternoon! Rough road out there. What have you got for sale?<|im_end|>\n<|fim_middle|><|rein_0|><|rein_1|> ... <|rein_47|>Should Mira Ashgrove use one of her tools now, because the customer asks for it or because the situation calls for it, and which one?<|box_start|>A. show_inventory: the customer wants to know what is for sale or how much something costs<|box_start|>B. sell_item: the customer decides to buy something from the stock<|box_start|>C. buy_item: the customer offers to sell Mira something<|box_start|>D. repair_gear: the customer wants a piece of equipment fixed<|box_start|>E. call_guards: someone threatens Mira, draws a weapon, or tries to rob the store<|box_start|>F. none: talk, questions, stories, or requests her tools cannot serve<|rein_48|> ... <|rein_63|><|fim_suffix|>",
  "n_predict": 1,
  "n_probs": 60,
  "temperature": 0,
  "cache_prompt": true
}
JSON
```

Write the 64 soft tokens out in full (`...` stands for the ones left out here). The answer is in
`completion_probabilities[0].top_logprobs`: on this turn the letter `A` (`show_inventory`) carries 0.99 after the
temperature. In Python, with the `rein` package, the same call and the whole game loop are a few lines
(`Engine.served(server, folder)` on the main card); without it, the main card's `decide()` is thirty lines of plain
`httpx`.

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
