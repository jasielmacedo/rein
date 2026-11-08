// Rein's decision protocol over wllama: the state and one row per rule, read from the letters' log-probabilities,
// and the acting rule that turns the answers into an opening, a veto and reminders.

import {
  ACT_PROBABILITY,
  DECISION_TEMPERATURE,
  LETTERS,
  MISSING_LOGPROB,
  NULL_OPTIONS,
  SOFT_FRONT,
  SOFT_TAIL,
  TOP_LOGPROBS,
  VETO_PROBABILITY,
  YES,
} from "./config.js";

const FRONT = softTokens(0, SOFT_FRONT);
const TAIL = softTokens(SOFT_FRONT, SOFT_FRONT + SOFT_TAIL);
const NO_HINT_KEYS = ["no", "yes"];

function softTokens(start, end) {
  let text = "";
  for (let index = start; index < end; index += 1) text += `<|rein_${index}|>`;
  return text;
}

// JSON as Python's json.dumps writes it with sort_keys=True and ensure_ascii=False: the tool calls in the state
// have to tokenize exactly as they did in training.
export function pythonJson(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(pythonJson).join(", ")}]`;
  const keys = Object.keys(value).sort();
  return `{${keys.map((key) => `${JSON.stringify(key)}: ${pythonJson(value[key])}`).join(", ")}}`;
}

export function contextText(sheet) {
  return [sheet.persona, sheet.scene].map((part) => part.trim()).filter(Boolean).join("\n\n");
}

function messageText(message) {
  const header = message.role === "tool" ? `tool ${message.name}` : message.role;
  let text = `<|im_start|>${header}\n${message.text}`;
  for (const call of message.toolCalls ?? []) {
    text += `<tool_call>\n${pythonJson({ arguments: call.arguments, name: call.name })}\n</tool_call>`;
  }
  return `${text}<|im_end|>\n`;
}

export function stateText(sheet, history) {
  return `<|fim_prefix|>Context:\n${contextText(sheet)}\nConversation:\n${history.map(messageText).join("")}`;
}

export function optionKeys(rule) {
  return rule.kind === "yes_no" ? NO_HINT_KEYS : rule.options.map((option) => option.key);
}

export function optionTexts(rule) {
  if (rule.kind === "yes_no") {
    return [rule.no ? `no: ${rule.no}` : "no", rule.yes ? `yes: ${rule.yes}` : "yes"];
  }
  return rule.options.map((option) => (option.when ? `${option.key}: ${option.when}` : option.key));
}

export function rowText(rule) {
  const lines = optionTexts(rule)
    .map((text, index) => `<|box_start|>${LETTERS[index]}. ${text}`)
    .join("");
  return `<|fim_middle|>${FRONT}${rule.question}${lines}${TAIL}<|fim_suffix|>`;
}

// llama.cpp's completion answers with the predicted token's top candidates as a list of {token, logprob} entries.
// The letters are exact tokens ("A", never " A"), so the table keeps every candidate under its exact text and the
// first (most probable) entry of a text wins.
export function topLogprobs(response) {
  const entry = response.choices?.[0]?.logprobs?.content?.[0];
  const candidates = entry?.top_logprobs ?? [];
  const table = {};
  for (const candidate of candidates) {
    const token = String(candidate.token);
    if (!(token in table)) table[token] = candidate.logprob;
  }
  return table;
}

export function letterProbabilities(top, count) {
  const scores = [];
  for (let index = 0; index < count; index += 1) {
    const letter = LETTERS[index];
    scores.push((letter in top ? top[letter] : MISSING_LOGPROB) / DECISION_TEMPERATURE);
  }
  const peak = Math.max(...scores);
  const weights = scores.map((score) => Math.exp(score - peak));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  return weights.map((weight) => weight / total);
}

// One completion request per rule: the state, the row, one predicted token, its top log-probabilities.
export async function decide(wllama, sheet, history) {
  const state = stateText(sheet, history);
  const answers = {};
  for (const rule of sheet.rules) {
    const response = await wllama.createCompletion({
      prompt: state + rowText(rule),
      max_tokens: 1,
      temperature: 0,
      logprobs: TOP_LOGPROBS,
    });
    const keys = optionKeys(rule);
    const probabilities = letterProbabilities(topLogprobs(response), keys.length);
    answers[rule.id] = Object.fromEntries(keys.map((key, index) => [key, probabilities[index]]));
  }
  return answers;
}

function nullMass(probabilities) {
  return Object.entries(probabilities)
    .filter(([key]) => NULL_OPTIONS.includes(key))
    .reduce((sum, [, value]) => sum + value, 0);
}

function bestTool(probabilities) {
  const tools = Object.entries(probabilities).filter(([key]) => !NULL_OPTIONS.includes(key));
  return tools.reduce((best, entry) => (entry[1] > best[1] ? entry : best))[0];
}

export function toolOpening(tool) {
  return `<tool_call>\n<function=${tool}>\n`;
}

// The acting rule: the most probable tool opens when the null options hold under half the mass; a sure "none"
// vetoes the call; a yes/no rule that fires lends its base as a reminder.
export function plan(sheet, answers) {
  let opening = "";
  let tool = null;
  let veto = false;
  const reminders = [];
  for (const rule of sheet.rules) {
    const probabilities = answers[rule.id];
    if (!probabilities) continue;
    if (rule.action === "tool") {
      const none = nullMass(probabilities);
      if (none < ACT_PROBABILITY) {
        tool = bestTool(probabilities);
        opening = toolOpening(tool);
      } else if (none >= VETO_PROBABILITY) {
        veto = true;
      }
    } else if (rule.kind === "yes_no" && probabilities[YES] >= ACT_PROBABILITY && rule.base) {
      reminders.push(rule.base);
    }
  }
  return { opening, tool, veto, reminders, answers };
}

export const AFTER_TOOL_RESULT = { opening: "", tool: null, veto: true, reminders: [], answers: {} };
