from typing import Final

STATE_OPEN = "<|fim_prefix|>"
QUESTION_OPEN = "<|fim_middle|>"
QUERY_MARK = "<|fim_suffix|>"
OPTION_OPEN = "<|box_start|>"
OPTION_CLOSE = "<|box_end|>"
TURN_OPEN = "<|im_start|>"
TURN_CLOSE = "<|im_end|>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

CONTEXT_HEADER = "Context:\n"
CONVERSATION_HEADER = "Conversation:\n"
DRAFT_HEADER = "Reply in progress:\n"
DRAFT_ROLE = "assistant"
NEWLINE = "\n"

SANITIZE_OPEN = chr(0x2039)
SANITIZE_CLOSE = chr(0x203A)

DEFAULT_MAX_STATE_TOKENS = 8192
MIN_STATE_TOKENS = 16
MAX_ROW_TOKENS = 1024

HEAD_DIM = 256
HEAD_HEADS = 4
HEAD_FFN_MULT = 4

FULL_ATTENTION = "full_attention"
SDPA = "sdpa"
PAD_DECAY_LOGIT = -1.0e4

LAYOUT_ROWS: Final = "rows"
LAYOUT_LABELS: Final = "labels"
READOUT_SET: Final = "set"
READOUT_LABELS: Final = "labels"
ADAPTER_STATE_AND_ROWS: Final = "state_and_rows"
ADAPTER_ROWS: Final = "rows"
LABEL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LABEL_SEPARATOR = ". "
SOFT_TOKEN_FORMAT = "<|rein_{index}|>"
SOFT_INIT_VOCAB = 5000
