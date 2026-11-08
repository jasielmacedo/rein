const KEY_STORAGE = "rein-play-key";
const KEY_HEADER = "X-Play-Key";

function remember(value) {
  try {
    localStorage.setItem(KEY_STORAGE, value);
  } catch {
    return;
  }
}

function remembered() {
  try {
    return localStorage.getItem(KEY_STORAGE) || "";
  } catch {
    return "";
  }
}

function readKey() {
  const fromAddress = new URLSearchParams(location.hash.slice(1)).get("key");
  if (!fromAddress) return remembered();
  remember(fromAddress);
  history.replaceState(null, "", location.pathname);
  return fromAddress;
}

const key = readKey();

async function request(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: { [KEY_HEADER]: key, ...(body ? { "Content-Type": "application/json" } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (response.status === 401) {
    throw new Error("Chave de acesso ausente ou errada: abra o endereço com #key= que o servidor mostrou.");
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(typeof detail.detail === "string" ? detail.detail : `Erro ${response.status}`);
  }
  return response.json();
}

export const api = {
  characters: () => request("GET", "/api/characters"),
  conversations: (character) => request("GET", `/api/characters/${encodeURIComponent(character)}/conversations`),
  start: (character) => request("POST", "/api/conversations", { character }),
  conversation: (id) => request("GET", `/api/conversations/${id}`),
  remove: (id) => request("DELETE", `/api/conversations/${id}`),
  send: (id, text) => request("POST", `/api/conversations/${id}/messages`, { text }),
  label: (messageId, label) => request("PUT", `/api/messages/${messageId}/label`, { label }),
};
