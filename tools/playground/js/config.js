// The model, the runtime and the decision protocol's constants. Everything else derives from these.

export const MODEL_REPO = "jmc0/Rein-NPC-4B-GGUF";
export const MODEL_FILE = "web/Rein-NPC-4B-Q4_K_M-00001-of-00004.gguf";
export const MODEL_SHARDS = 4;
export const MODEL_BYTES = 3066385152;
export const MODEL_LABEL = "Rein-NPC-4B, Q4_K_M";
export const MODEL_CARD_URL = "https://huggingface.co/jmc0/Rein-NPC-4B";
export const GGUF_CARD_URL = "https://huggingface.co/jmc0/Rein-NPC-4B-GGUF";

export const WLLAMA_VERSION = "3.8.1";
export const WLLAMA_ESM = `https://cdn.jsdelivr.net/npm/@wllama/wllama@${WLLAMA_VERSION}/esm/index.js`;
export const WLLAMA_WASM = `https://cdn.jsdelivr.net/npm/@wllama/wllama@${WLLAMA_VERSION}/esm/wasm/wllama.wasm`;
export const JINJA_ESM = "https://cdn.jsdelivr.net/npm/@huggingface/jinja@0.5.11/dist/index.js";
export const PARALLEL_DOWNLOADS = 3;

export const CONTEXT_TOKENS = 8192;
export const BATCH_TOKENS = 512;
export const REPLY_TOKENS = 300;
export const SAMPLING = { temperature: 0.7, top_p: 0.8, top_k: 20 };
export const MAX_CALLS_PER_TURN = 1;

export const SOFT_FRONT = 48;
export const SOFT_TAIL = 16;
export const DECISION_TEMPERATURE = 0.812;
export const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
export const TOP_LOGPROBS = 60;
export const MISSING_LOGPROB = -60;
export const ACT_PROBABILITY = 0.5;
export const VETO_PROBABILITY = 0.9;
export const NULL_OPTIONS = ["no", "none", "general"];
export const YES = "yes";
export const TOOL_CALL_TOKEN_ID = 248058;
export const BAN_BIAS = -100;

export const END_TURN = "<|im_end|>";
export const REMINDER_HEADER = "Reminder for this reply:";
export const TOOL_CALL_OPEN = "<tool_call>";
export const TOOL_CALL_CLOSE = "</tool_call>";
