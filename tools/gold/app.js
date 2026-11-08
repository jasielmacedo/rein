"use strict";
(function (Gold) {
  const el = (...args) => Gold.el(...args);
  const SAVE_NAME = "gold-project.json";
  const FILE_TYPES = [{ description: "Projeto do gold set", accept: { "application/json": [".json"] } }];

  const restored = Gold.restore();
  const app = {
    project: restored.project,
    ui: { sheet: restored.sheet, conversation: restored.conversation, message: null, view: "chat", speaker: "player", draftText: "", help: false },
    fileHandle: null,
  };
  Gold.app = app;

  const sheetById = (id) => app.project.sheets.find((s) => s.id === id) || null;
  const conversationById = (id) => app.project.conversations.find((c) => c.id === id) || null;
  const conversationsOf = (sheet) => app.project.conversations.filter((c) => c.sheet === (sheet && sheet.id));

  function persist() {
    Gold.persist({ project: app.project, sheet: app.ui.sheet, conversation: app.ui.conversation });
  }

  app.changed = function changed(rerender = false) {
    persist();
    if (rerender) app.render();
    else renderSidebar();
  };

  function ensureIds() {
    for (const sheet of app.project.sheets) for (const rule of sheet.rules) Gold.ensureRuleId(sheet, rule);
  }

  app.selectSheet = function selectSheet(id) {
    app.ui.sheet = id;
    const first = conversationsOf(sheetById(id))[0];
    app.ui.conversation = first ? first.id : null;
    app.ui.message = null;
    app.ui.view = "chat";
    app.changed(true);
  };

  app.selectConversation = function selectConversation(id) {
    app.ui.conversation = id;
    app.ui.message = null;
    app.ui.view = "chat";
    app.changed(true);
  };

  app.selectMessage = function selectMessage(index) {
    app.ui.message = index;
    app.changed(true);
  };

  app.moveMessage = function moveMessage(index, delta) {
    const messages = conversationById(app.ui.conversation).messages;
    const to = index + delta;
    if (to < 0 || to >= messages.length) return;
    [messages[index], messages[to]] = [messages[to], messages[index]];
    app.ui.message = to;
    app.changed(true);
  };

  app.deleteMessage = function deleteMessage(index) {
    if (!confirm("Apagar esta mensagem?")) return;
    conversationById(app.ui.conversation).messages.splice(index, 1);
    app.ui.message = null;
    app.changed(true);
  };

  app.deleteConversation = function deleteConversation() {
    const conversation = conversationById(app.ui.conversation);
    if (!conversation || !confirm("Apagar esta conversa inteira?")) return;
    app.project.conversations = app.project.conversations.filter((c) => c !== conversation);
    const next = conversationsOf(sheetById(app.ui.sheet))[0];
    app.ui.conversation = next ? next.id : null;
    app.ui.message = null;
    app.changed(true);
  };

  app.deleteSheet = function deleteSheet(id) {
    const sheet = sheetById(id);
    if (!sheet) return;
    const count = conversationsOf(sheet).length;
    const name = sheet.npc.trim() || "esta ficha";
    if (!confirm(`Apagar a ficha de ${name}${count ? ` e as ${count} conversas dela` : ""}? Não dá para desfazer.`)) return;
    app.project.sheets = app.project.sheets.filter((s) => s !== sheet);
    app.project.conversations = app.project.conversations.filter((c) => c.sheet !== sheet.id);
    const next = app.project.sheets[0];
    app.ui.sheet = next ? next.id : null;
    const conversation = next ? conversationsOf(next)[0] : null;
    app.ui.conversation = conversation ? conversation.id : null;
    app.ui.message = null;
    app.ui.view = "chat";
    app.changed(true);
    Gold.toast(`Ficha de ${name} apagada`);
  };

  app.showSheet = function showSheet() {
    app.ui.view = "sheet";
    app.changed(true);
  };

  app.showChat = function showChat() {
    ensureIds();
    app.ui.view = "chat";
    app.changed(true);
  };

  function newSheet() {
    const sheet = Gold.blankSheet(app.project);
    sheet.rules.push(Gold.blankRule());
    app.project.sheets.push(sheet);
    app.ui.sheet = sheet.id;
    app.ui.conversation = null;
    app.ui.view = "sheet";
    app.changed(true);
  }

  function addExample() {
    const { sheet, conversation } = Gold.addMarenExample(app.project);
    app.ui.sheet = sheet.id;
    app.ui.conversation = conversation.id;
    app.ui.message = null;
    app.ui.view = "chat";
    app.changed(true);
    Gold.toast("Exemplo da Maren adicionado");
  }

  function newConversation() {
    const sheet = sheetById(app.ui.sheet);
    if (!sheet) return;
    ensureIds();
    const conversation = Gold.blankConversation(app.project, sheet);
    app.project.conversations.push(conversation);
    app.ui.conversation = conversation.id;
    app.ui.message = null;
    app.ui.speaker = "player";
    app.ui.view = "chat";
    app.changed(true);
  }

  function conversationLabel(conversation) {
    if (conversation.title.trim()) return conversation.title.trim();
    const first = conversation.messages.find((m) => m.text.trim());
    return first ? first.text.trim() : "(conversa vazia)";
  }

  function renderSidebar() {
    const sheet = sheetById(app.ui.sheet);
    const { decisions, firing } = Gold.summary(app.project);
    Gold.fill(Gold.$("#sidebar"),
      el("h4", { textContent: "Fichas de NPC" }),
      app.project.sheets.map((s) => el("div", { className: "item" + (s.id === app.ui.sheet ? " active" : ""), on: { click: () => app.selectSheet(s.id) } },
        el("span", { className: "name", textContent: s.npc.trim() || "(ficha sem nome)" }),
        el("span", { className: "count", textContent: `${s.rules.length} regras` }),
        Gold.sheetProblems(s).length ? el("span", { className: "dot", title: "a ficha tem pendências" }) : null,
        el("button", {
          className: "small danger remove", textContent: "×", title: "apagar esta ficha e as conversas dela",
          on: { click: (e) => { e.stopPropagation(); app.deleteSheet(s.id); } },
        }))),
      el("div", { className: "item-actions" },
        el("button", { className: "small", textContent: "+ nova ficha", on: { click: newSheet } }),
        el("button", { className: "small", textContent: "+ exemplo: Maren", on: { click: addExample } })),
      sheet ? el("h4", { textContent: `Conversas com ${sheet.npc.trim() || "este NPC"}` }) : null,
      sheet ? conversationsOf(sheet).map((c) => {
        const missing = c.messages.some((m) => Gold.messageProblems(sheet, m).length);
        return el("div", { className: "item" + (c.id === app.ui.conversation && app.ui.view === "chat" ? " active" : ""), on: { click: () => app.selectConversation(c.id) } },
          el("span", { className: "name", textContent: conversationLabel(c) }),
          el("span", { className: "count", textContent: `${c.messages.length} msgs` }),
          missing ? el("span", { className: "dot", title: "há mensagens sem resposta" }) : null);
      }) : null,
      sheet ? el("div", { className: "item-actions" }, el("button", { className: "small", textContent: "+ nova conversa", on: { click: newConversation } })) : null,
      el("p", { className: "summary", textContent: `Gold set: ${decisions} casos, ${firing} com alguma regra disparando.` }));
  }

  function welcome(center, side) {
    center.replaceChildren(el("div", { className: "welcome" },
      el("h2", { textContent: "Vamos montar o gold set" }),
      el("p", { textContent: "Você cria a ficha de um NPC e escreve conversas com ele como num chat. Depois clica nas mensagens do jogador e marca o que o NPC deveria fazer. Cada mensagem marcada vira um caso de teste." }),
      el("div", { className: "welcome-actions" },
        el("button", { className: "primary big", textContent: "Começar pelo exemplo da Maren", on: { click: addExample } }),
        el("button", { className: "big", textContent: "Criar uma ficha do zero", on: { click: newSheet } }))));
    side.replaceChildren(helpBody());
  }

  function helpBody() {
    return el("div", { className: "panel-empty" },
      el("h3", { textContent: "Como funciona" }),
      el("ol", {},
        el("li", { textContent: "Ficha: quem é o NPC, as ferramentas que ele pode chamar e as regras, escritas como perguntas." }),
        el("li", { textContent: "Conversa: escreva o que o jogador diz e o que o NPC responde, como num chat." }),
        el("li", { textContent: "Marcação: clique numa mensagem do jogador e diga o que cada regra deve decidir ali. 'Não dispara' já vem marcado." }),
        el("li", { textContent: "Salvar: gera um arquivo .json. O pipeline lê esse arquivo do jeito que está." })),
      el("p", { className: "hint", textContent: "O rascunho fica salvo neste navegador enquanto você trabalha, mas só o arquivo salvo vai para o pipeline." }));
  }

  app.render = function render() {
    const center = Gold.$("#center");
    const side = Gold.$("#side");
    renderSidebar();
    Gold.$("#help").hidden = !app.ui.help;
    if (app.ui.help) Gold.$("#help").replaceChildren(el("div", { className: "help-card" }, helpBody(), el("button", { textContent: "Fechar", on: { click: () => { app.ui.help = false; app.render(); } } })));
    const sheet = sheetById(app.ui.sheet);
    if (!sheet) { welcome(center, side); return; }
    if (app.ui.view === "sheet") {
      Gold.renderSheet(center, sheet, app);
      side.replaceChildren(el("div", { className: "panel-empty" },
        el("h3", { textContent: "Dicas para a ficha" }),
        el("ul", {},
          el("li", { textContent: "Escreva as regras como perguntas que alguém responderia lendo só a conversa." }),
          el("li", { textContent: "Para ferramentas, use uma regra 'uma opção de uma lista' e o botão 'usar as ferramentas da ficha'." }),
          el("li", { textContent: "Regras sobre o que o NPC escreveu (ex.: contou que fez algo sem chamar a ferramenta) olham 'a resposta do NPC'." }))));
      return;
    }
    const conversation = conversationById(app.ui.conversation);
    if (!conversation || conversation.sheet !== sheet.id) {
      center.replaceChildren(el("div", { className: "welcome" },
        el("h2", { textContent: `Conversas com ${sheet.npc.trim() || "este NPC"}` }),
        el("p", { textContent: "Crie uma conversa e escreva como num chat." }),
        el("button", { className: "primary big", textContent: "+ nova conversa", on: { click: newConversation } })));
      side.replaceChildren(helpBody());
      return;
    }
    Gold.renderChat(center, side, conversation, sheet, app);
  };

  async function writeTo(handle) {
    const writable = await handle.createWritable();
    await writable.write(Gold.serialize(app.project));
    await writable.close();
  }

  function download() {
    const url = URL.createObjectURL(new Blob([Gold.serialize(app.project)], { type: "application/json" }));
    el("a", { href: url, download: SAVE_NAME }).click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function save(askWhere) {
    ensureIds();
    persist();
    if (!window.showSaveFilePicker) { download(); Gold.toast(`Baixado como ${SAVE_NAME}`); return; }
    try {
      if (!app.fileHandle || askWhere) app.fileHandle = await window.showSaveFilePicker({ suggestedName: SAVE_NAME, types: FILE_TYPES });
      await writeTo(app.fileHandle);
      Gold.$("#file").textContent = app.fileHandle.name;
      const { decisions } = Gold.summary(app.project);
      Gold.toast(`Salvo: ${decisions} casos`);
    } catch (error) {
      if (error.name !== "AbortError") { download(); Gold.toast("Não deu para gravar no arquivo; baixei uma cópia"); }
    }
  }

  function load(text, name) {
    app.project = Gold.parseProject(text);
    const first = app.project.sheets[0];
    app.ui.sheet = first ? first.id : null;
    const conversation = first ? conversationsOf(first)[0] : null;
    app.ui.conversation = conversation ? conversation.id : null;
    app.ui.message = null;
    app.ui.view = "chat";
    Gold.$("#file").textContent = name;
    app.changed(true);
    Gold.toast(`Aberto: ${app.project.sheets.length} fichas, ${app.project.conversations.length} conversas`);
  }

  async function open() {
    if (!window.showOpenFilePicker) { Gold.$("#picker").click(); return; }
    try {
      const [handle] = await window.showOpenFilePicker({ types: FILE_TYPES });
      load(await (await handle.getFile()).text(), handle.name);
      app.fileHandle = handle;
    } catch (error) {
      if (error.name !== "AbortError") Gold.toast("Não consegui abrir: " + error.message);
    }
  }

  Gold.$("#picker").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try { load(await file.text(), file.name); } catch (error) { Gold.toast("Não consegui abrir: " + error.message); }
  });
  Gold.$("#open").addEventListener("click", open);
  Gold.$("#save").addEventListener("click", () => save(false));
  Gold.$("#saveas").addEventListener("click", () => save(true));
  Gold.$("#howto").addEventListener("click", () => { app.ui.help = !app.ui.help; app.render(); });
  document.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "s") { e.preventDefault(); save(false); } });

  app.render();
})(window.Gold = window.Gold || {});
