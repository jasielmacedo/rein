// The chat side: Qwen's own template rendered from the GGUF, the reply generated with the opening prefilled and the
// tool-call token banned when the acting rule says so, and the tool call parsed out of the reply.

import {
  BAN_BIAS,
  END_TURN,
  REMINDER_HEADER,
  REPLY_TOKENS,
  SAMPLING,
  TOOL_CALL_OPEN,
  TOOL_CALL_TOKEN_ID,
} from "./config.js";
import { contextText } from "./rein.js";

const CALL_PATTERN = /<tool_call>\s*<function=([A-Za-z0-9_.:\-]{1,64})>(.*?)<\/function>\s*<\/tool_call>/s;
const PARAMETER_PATTERN = /<parameter=([A-Za-z0-9_.:\-]{1,64})>\n?(.*?)\n?<\/parameter>/gs;
const THINK_PATTERN = /<think>.*?<\/think>/gs;

function templateMessage(message, index) {
  if (message.role === "tool") return { role: "tool", name: message.name, content: message.text };
  const payload = { role: message.role, content: message.text };
  if (message.toolCalls?.length) {
    payload.tool_calls = message.toolCalls.map((call, callIndex) => ({
      type: "function",
      id: `call_${index}_${callIndex}`,
      function: { name: call.name, arguments: call.arguments },
    }));
  }
  return payload;
}

function withReminders(messages, reminders) {
  if (!reminders.length) return messages;
  const note = [REMINDER_HEADER, ...reminders.map((reminder) => `- ${reminder}`)].join("\n");
  const last = messages[messages.length - 1];
  if (last?.role === "user") {
    return [...messages.slice(0, -1), { ...last, content: `${last.content}\n\n${note}` }];
  }
  return [...messages, { role: "user", content: note }];
}

// The generator sees the persona and scene as its system prompt, the tools, and the conversation; the rules never.
export function chatPrompt(template, sheet, history, reminders) {
  const messages = withReminders(
    [{ role: "system", content: contextText(sheet) }, ...history.map(templateMessage)],
    reminders,
  );
  return template.render({
    messages,
    tools: sheet.tools.map((tool) => ({ type: "function", function: tool })),
    add_generation_prompt: true,
    enable_thinking: false,
  });
}

export async function generate(wllama, prompt, actions, onToken) {
  const request = {
    prompt: prompt + actions.opening,
    max_tokens: REPLY_TOKENS,
    stop: [END_TURN],
    ...SAMPLING,
  };
  if (actions.veto) request.logit_bias = { [TOOL_CALL_TOKEN_ID]: BAN_BIAS };
  let text = "";
  if (onToken) {
    for await (const chunk of await wllama.createCompletion({ ...request, stream: true })) {
      const piece = chunk.choices?.[0]?.text ?? "";
      text += piece;
      onToken(actions.opening + text);
    }
  } else {
    const response = await wllama.createCompletion(request);
    text = response.choices?.[0]?.text ?? "";
  }
  return actions.opening + text;
}

export function parseToolCall(raw) {
  const match = raw.match(CALL_PATTERN);
  if (!match) return null;
  const args = {};
  for (const parameter of match[2].matchAll(PARAMETER_PATTERN)) args[parameter[1]] = parameter[2];
  return { name: match[1], arguments: args };
}

export function visibleText(raw) {
  return raw.split(TOOL_CALL_OPEN, 1)[0].replace(THINK_PATTERN, "").trim();
}
