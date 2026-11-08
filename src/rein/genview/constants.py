GENERIC_SYSTEM = "You are a helpful assistant."
CHAT_TOKENIZER_REPO = "Qwen/Qwen3.5-4B"

DEFAULT_SILENCE_POSITIONS = 4
REPLY_START = "reply_start"
SILENCE = "silence"
CONTINUATION_END = "continuation_end"

THINK_CLOSE = "</think>"
ANSWER_OPENING = "</think>\n\n"

TOOL_CATEGORY = "tool_trigger"
NULL_OPTIONS = frozenset({"none", "general"})
TOOL_PARAMETERS = {"type": "object", "properties": {}}
SCENARIO_GLOB = "*/scenarios.jsonl"

PROMPT_BASES = "bases"
PROMPT_NONE = "none"
PROMPT_RULES = "rules"
PROMPT_BLOCK = "block"
PROMPTS = (PROMPT_BASES, PROMPT_NONE, PROMPT_RULES, PROMPT_BLOCK)
RULES_TEMPLATE = "configs/template/qwen3.5-rules.jinja"
