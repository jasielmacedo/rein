"use strict";
(function (Gold) {
  const el = (...args) => Gold.el(...args);

  function speakerName(sheet, from) {
    if (from === "npc") return (sheet && sheet.npc.trim()) || Gold.SPEAKERS.npc;
    return Gold.SPEAKERS[from];
  }

  function outcome(sheet, message) {
    if (message.skip) return el("span", { className: "chip muted-chip", textContent: "fora do gold set" });
    const rules = Gold.rulesFor(sheet, message);
    if (!rules.length) return null;
    const missing = Gold.messageProblems(sheet, message);
    if (missing.length) return el("span", { className: "chip warn-chip", textContent: "falta marcar" });
    const firing = rules.filter((rule) => Gold.fires(rule, Gold.answerOf(message, rule)));
    if (!firing.length) return el("span", { className: "chip calm-chip", textContent: "nada dispara" });
    return firing.map((rule) => {
      const answer = Gold.answerOf(message, rule);
      const what = rule.kind === "yes_no" ? Gold.labelOf(Gold.CATEGORIES, rule.category) : `${Gold.labelOf(Gold.CATEGORIES, rule.category)}: ${answer}`;
      return el("span", { className: "chip fire-chip", textContent: what });
    });
  }

  function bubble(sheet, message, index, selected, app) {
    const side = message.from === "player" ? "right" : message.from === "npc" ? "left" : "center";
    return el("div", { className: `bubble-row ${side}` },
      el("div", {
        className: `bubble ${message.from}` + (selected ? " selected" : ""),
        title: "Clique para marcar o que deve acontecer",
        on: { click: () => app.selectMessage(index) },
      },
        el("div", { className: "who", textContent: `${speakerName(sheet, message.from)} · ${index + 1}` }),
        el("div", { className: "text", textContent: message.text }),
        message.tool_calls.length ? el("div", { className: "calls" }, message.tool_calls.map((name) => el("span", { className: "chip tool-chip", textContent: `chamou ${name}` }))) : null,
        el("div", { className: "outcome" }, outcome(sheet, message))));
  }

  function composer(conversation, sheet, app) {
    const ui = app.ui;
    const placeholders = {
      player: "Escreva o que o jogador diz. Enter envia, Shift+Enter quebra a linha.",
      npc: `Escreva o que ${speakerName(sheet, "npc")} responde.`,
      narration: "Descreva a cena, ex.: *A neblina engole o farol.*",
    };
    const box = el("textarea", { rows: 2, placeholder: placeholders[ui.speaker], value: ui.draftText || "" });
    const send = () => {
      const text = box.value.trim();
      if (!text) return;
      conversation.messages.push(Gold.blankMessage(ui.speaker, text));
      const sent = conversation.messages.length - 1;
      ui.draftText = "";
      ui.speaker = ui.speaker === "player" ? "npc" : ui.speaker === "npc" ? "player" : "npc";
      app.selectMessage(sent, true);
    };
    box.addEventListener("input", () => { ui.draftText = box.value; });
    box.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
    return el("div", { className: "composer" },
      el("div", { className: "speakers" },
        el("span", { className: "muted", textContent: "Quem fala:" }),
        ["player", "npc", "narration"].map((from) => el("button", {
          className: "seg" + (ui.speaker === from ? " on" : ""), textContent: speakerName(sheet, from),
          on: { click: () => { ui.speaker = from; ui.draftText = box.value; app.render(); } },
        }))),
      el("div", { className: "send" }, box, el("button", { className: "primary", textContent: "Enviar", on: { click: send } })));
  }

  function answerButtons(rule, message, app) {
    const current = Gold.answerOf(message, rule);
    const fallback = Gold.defaultAnswer(rule);
    const descriptions = rule.kind === "yes_no"
      ? { yes: rule.yes, no: rule.no }
      : rule.kind === "choice" ? Object.fromEntries(rule.options.map((o) => [o.key, o.when])) : {};
    return el("div", { className: "answers" }, Gold.optionKeys(rule).map((key) => el("button", {
      className: "answer" + (current === key ? " on" : "") + (Gold.fires(rule, key) ? " firing" : ""),
      on: { click: () => { message.answers[rule.id] = key; app.changed(true); } },
    },
      el("strong", { textContent: Gold.answerLabel(rule, key) + (key === fallback ? " (padrão)" : "") }),
      descriptions[key] ? el("span", { textContent: descriptions[key] }) : null)));
  }

  function ruleQuestion(rule, message, app) {
    return el("div", { className: "question" },
      el("div", { className: "qhead" },
        el("span", { className: "tag", textContent: Gold.labelOf(Gold.CATEGORIES, rule.category) })),
      el("p", { className: "qtext", textContent: Gold.ruleTitle(rule) }),
      answerButtons(rule, message, app));
  }

  function formChooser(message, app) {
    return el("div", { className: "forms" }, Gold.FORMS.map(([key, label, hint]) => el("button", {
      className: "seg" + (message.form === key ? " on" : ""), textContent: label, title: hint,
      on: { click: () => { message.form = key; app.changed(true); } },
    })), el("p", { className: "hint", textContent: Gold.hintOf(Gold.FORMS, message.form) }));
  }

  function toolCalls(sheet, message, app) {
    if (!sheet || !sheet.tools.length) return null;
    return Gold.field("O NPC chamou alguma ferramenta nesta resposta?", "Marque se a resposta já vem com a chamada. Se o NPC só contou que fez algo, não marque.",
      el("div", { className: "checks" }, sheet.tools.filter((t) => t.name).map((tool) => el("label", { className: "check" },
        el("input", { type: "checkbox", checked: message.tool_calls.includes(tool.name), on: { change: (e) => {
          message.tool_calls = e.target.checked ? [...message.tool_calls, tool.name] : message.tool_calls.filter((n) => n !== tool.name);
          app.changed(true);
        } } }),
        tool.name))));
  }

  function panel(conversation, sheet, index, app) {
    const message = conversation.messages[index];
    if (!message) {
      return el("div", { className: "panel-empty" },
        el("h3", { textContent: "Como marcar" }),
        el("ol", {},
          el("li", { textContent: "Escreva a conversa como num chat, alternando jogador e NPC." }),
          el("li", { textContent: "Clique numa mensagem do jogador." }),
          el("li", { textContent: "Para cada regra, diga o que deve acontecer ao receber essa mensagem. Se nada deve acontecer, não precisa mexer: 'não dispara' já vem marcado." })),
        el("p", { className: "hint", textContent: "Cada mensagem marcada vira um caso do gold set, com tudo o que veio antes dela como contexto." }));
    }
    const rules = Gold.rulesFor(sheet, message);
    const isNpc = message.from === "npc";
    const intro = isNpc
      ? "Olhando o que o NPC escreveu, o que as regras de resposta devem decidir?"
      : "O NPC acabou de receber esta mensagem. Para cada regra, o que deve acontecer? Por padrão, nada dispara.";
    const testsPairs = [["", "Escolher sozinho (a regra que dispara)"], ...rules.map((rule) => [rule.id, Gold.ruleTitle(rule)])];
    return el("div", { className: "panel-body" },
      el("div", { className: "panel-top" },
        el("h3", { textContent: `${speakerName(sheet, message.from)} · mensagem ${index + 1}` }),
        el("span", { className: "spacer" }),
        el("button", { className: "small", textContent: "↑", title: "subir", on: { click: () => app.moveMessage(index, -1) } }),
        el("button", { className: "small", textContent: "↓", title: "descer", on: { click: () => app.moveMessage(index, 1) } }),
        el("button", { className: "small danger", textContent: "apagar", on: { click: () => app.deleteMessage(index) } })),
      Gold.field("Texto", null, Gold.textArea(message.text, (v) => {
        message.text = v;
        const shown = document.querySelector(".bubble.selected .text");
        if (shown) shown.textContent = v;
        app.changed();
      }, "", 3)),
      isNpc ? toolCalls(sheet, message, app) : null,
      rules.length
        ? el("div", {},
          el("h4", { textContent: "O que deve acontecer aqui?" }),
          el("p", { className: "hint", textContent: intro }),
          rules.map((rule) => ruleQuestion(rule, message, app)),
          el("details", { className: "more", open: Boolean(message.tests || (message.form && message.form !== "auto") || message.note) },
            el("summary", { textContent: "Detalhes do caso (opcional)" }),
            Gold.field("Esta mensagem está testando qual regra?", "Use quando ela tenta enganar uma regra sem disparar, ex.: pergunta o nome sem tentar trocá-lo.",
              Gold.choiceSelect(testsPairs, message.tests, (v) => { message.tests = v; app.changed(true); })),
            Gold.field("Como a mensagem foi escrita?", null, formChooser(message, app)),
            Gold.field("Por que essa é a resposta certa?", null, Gold.textArea(message.note, (v) => { message.note = v; app.changed(); }, "anotação para você", 2))),
          el("label", { className: "check" },
            el("input", { type: "checkbox", checked: message.skip, on: { change: (e) => { message.skip = e.target.checked; app.changed(true); } } }),
            "Não usar esta mensagem no gold set (ela fica só como contexto)"))
        : el("p", { className: "hint", textContent: isNpc
          ? "Esta ficha não tem regras que olham a resposta do NPC, então esta mensagem só serve de contexto para as próximas."
          : "A ficha ainda não tem regras para a mensagem do jogador. Abra a ficha e crie uma." }));
  }

  Gold.renderChat = function renderChat(center, side, conversation, sheet, app) {
    const selected = app.ui.message;
    const log = el("div", { className: "log" }, conversation.messages.length
      ? conversation.messages.map((message, i) => bubble(sheet, message, i, i === selected, app))
      : el("div", { className: "panel-empty" },
        el("p", { textContent: `Comece a conversa. Escolha quem fala embaixo e escreva como num chat.` })));
    center.replaceChildren(
      el("div", { className: "chat" },
        el("div", { className: "chat-top" },
          Gold.textInput(conversation.title, (v) => { conversation.title = v; app.changed(); }, "título da conversa (opcional)"),
          Gold.choiceSelect(Gold.LANGUAGES, conversation.language, (v) => { conversation.language = v; app.changed(); }),
          el("button", { className: "small", textContent: "ficha", title: "Abrir a ficha do NPC", on: { click: () => app.showSheet() } }),
          el("button", { className: "small danger", textContent: "apagar conversa", on: { click: () => app.deleteConversation() } })),
        log,
        composer(conversation, sheet, app)));
    side.replaceChildren(panel(conversation, sheet, selected, app));
    const picked = log.querySelector(".bubble.selected");
    if (picked) picked.scrollIntoView({ block: "nearest" });
    else log.scrollTop = log.scrollHeight;
  };
})(window.Gold = window.Gold || {});
