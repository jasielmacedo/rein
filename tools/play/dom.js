export function el(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  if (options.text !== undefined) node.textContent = options.text;
  for (const [name, value] of Object.entries(options.attrs || {})) node.setAttribute(name, value);
  for (const [name, handler] of Object.entries(options.on || {})) node.addEventListener(name, handler);
  for (const child of children) if (child) node.append(child);
  return node;
}

export function clear(node) {
  node.replaceChildren();
}

export function show(node, visible) {
  node.hidden = !visible;
}
