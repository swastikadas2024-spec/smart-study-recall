/* RAG Chatbot: interactive study chatbot with standard/hybrid RAG toggle,
   live architecture visualisation, quiz/notes generation, and retrieval diagnostics. */
document.addEventListener("DOMContentLoaded", async () => {
  const S = window.SSR;
  if (!S.requireLogin()) return;
  const $ = (id) => document.getElementById(id);
  const state = {
    materials: [],
    active: null,
    studentId: null,
    mode: "hybrid",
    chatHistory: [],
    chatStarted: false,
    showDiag: false,
    lastDiagnostics: null,
    archData: null,
    flowTimer: null,
    flowStep: 0,
    lastQuestion: "",
  };
  const modeShort = (mode) => ({
    standard: "Standard",
    hybrid: "Hybrid",
    adaptive: "Recall-aware",
  }[mode] || "RAG");
  const modeName = (mode) => `${modeShort(mode)} RAG`;

  const run = async (work) => {
    try { await work(); } catch (err) { S.showError("pageError", err); }
  };

  // ---------- Icon map for pipeline steps ----------
  const STEP_ICONS = {
    "Document Upload": "📄",
    "Chunking": "✂️",
    "Keyword Index": "🔑",
    "Dual Indexing": "🔑",
    "Query": "❓",
    "BM25 Search": "🔍",
    "BM25 Keyword Search": "🔍",
    "Semantic Search": "🧠",
    "Rank Fusion (RRF)": "⚡",
    "Recall Gap Boost": "🎯",
    "Learner State": "🧠",
    "Intent-Aware Reranking": "🎯",
    "Generate": "✨",
    "Answer": "✅",
  };

  const STEP_CLASS = {
    "BM25 Search": "is-keyword",
    "BM25 Keyword Search": "is-keyword",
    "Keyword Index": "is-keyword",
    "Dual Indexing": "is-keyword",
    "Semantic Search": "is-semantic",
    "Rank Fusion (RRF)": "is-fusion",
    "Intent-Aware Reranking": "is-fusion",
    "Generate": "is-generate",
    "Answer": "is-generate",
  };

  // ---------- Load materials ----------
  async function loadMaterials() {
    const materials = await S.api("/materials");
    state.materials = materials;
    const select = $("materialSelect");
    select.innerHTML = '<option value="">— Choose a material —</option>';
    materials.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = `${m.title} (${m.chunk_count} passages)`;
      select.append(opt);
    });
  }

  function saveSetup() {
    if (!state.studentId) return;
    try {
      localStorage.setItem(`ssr.rag.setup.${state.studentId}`, JSON.stringify({
        material_id: state.active?.id || null,
        mode: state.mode,
      }));
    } catch (error) { /* Saving these preferences is a convenience. */ }
  }

  function setMode(mode) {
    state.mode = mode;
    $("ragMode").value = mode;
    const descriptions = {
      standard: "Fast keyword search (BM25). Best when the question uses the same terms as the source.",
      hybrid: "Recommended · combines keyword matching with TF-IDF similarity. The answer-generation provider is configured by the project administrator.",
      adaptive: "Hybrid retrieval, then a transparent rerank using your material quiz gaps and time since quizzing.",
    };
    $("ragModeHelp").textContent = descriptions[mode];
    $("ragModeIndicator").textContent = modeName(mode);
    $("toolsRagMode").textContent = modeShort(mode);
    saveSetup();
    if (!$("architecturePanel").hidden && state.archData) renderArchitecture(state.archData, mode);
  }

  $("ragMode").addEventListener("change", () => setMode($("ragMode").value));

  // ---------- Material selection ----------
  async function selectMaterial(id) {
    if (!id) {
      state.active = null;
      state.archData = null;
      if (state.flowTimer) window.clearInterval(state.flowTimer);
      state.flowTimer = null;
      $("chatSection").hidden = true;
      $("toolsSection").hidden = true;
      $("architectureSection").hidden = true;
      $("diagnosticsSection").hidden = true;
      $("materialInfo").hidden = true;
      saveSetup();
      return;
    }
    const material = state.materials.find((m) => m.id === id);
    if (!material) return;
    state.active = material;
    state.chatHistory = [];
    state.chatStarted = false;
    state.lastQuestion = "";
    resetChat();
    saveSetup();

    $("chatSection").hidden = false;
    $("toolsSection").hidden = false;
    $("architectureSection").hidden = false;
    $("architecturePanel").hidden = false;
    $("toggleArch").textContent = "Hide pipeline";
    $("diagnosticsSection").hidden = true;
    $("materialInfo").hidden = false;
    $("materialChunks").textContent = `${material.chunk_count} indexed passages`;
    $("ragModeIndicator").textContent = modeName(state.mode);
    $("toolsRagMode").textContent = modeShort(state.mode);

    await run(async () => {
      const archData = await S.api(`/rag/${id}/architecture`);
      state.archData = archData;
      $("materialGaps").textContent = `${archData.recall_gap_count} recall gaps tracked`;
      renderArchitecture(archData, state.mode);
    });
    try {
      await loadSavedHistory(id);
    } catch (error) {
      resetChat();
      const detail = error.status === 404
        ? `The running backend does not recognize the chat-history route. Stop and restart the main FastAPI backend from this project, then refresh. (${error.message})`
        : error.status >= 500
          ? `Chat history failed on the backend. Confirm migrations/004_rag_chat_history.sql has been run in Supabase. (${error.message})`
          : `Chat history could not be loaded. (${error.message})`;
      S.showError("pageError", new Error(detail));
    }
  }

  async function loadSavedHistory(materialId) {
    const result = await S.api(`/rag/${materialId}/history`);
    state.chatHistory = [];
    state.chatStarted = false;
    state.lastQuestion = "";
    resetChat();
    (result.messages || []).forEach((message) => {
      if (message.role === "user") {
        state.lastQuestion = message.content;
        state.chatHistory.push({ role: "user", content: message.content });
        addMessage("user", `<p>${S.esc(message.content)}</p>`);
      } else if (message.role === "assistant") {
        state.chatHistory.push({ role: "assistant", content: message.content });
        addMessage("assistant", `<p>${S.esc(message.content)}</p>${citationsHTML(message.citations)}
          <div class="rag-msg-meta"><span class="rag-msg-tag">${S.esc(modeName(message.mode))}</span><span class="rag-msg-tag">${S.esc(message.retrieval_method || "retrieval")}</span></div>
          ${feedbackHTML(message.interaction_id, message.supported_by_sources)}`);
      }
    });
    if (!state.chatStarted) resetChat();
  }

  function feedbackHTML(interactionId, value) {
    if (!interactionId) return "";
    const status = value === true ? "You marked this answer supported by its sources."
      : value === false ? "You marked this answer unsupported by its sources."
        : "Did the cited passages support this answer?";
    return `<div class="rag-feedback"><span class="rag-feedback-status">${status}</span>
      ${value === null || value === undefined ? `<button class="btn small ghost" type="button" data-supported="true" data-interaction="${interactionId}">Yes</button>
        <button class="btn small ghost" type="button" data-supported="false" data-interaction="${interactionId}">No</button>` : ""}</div>`;
  }

  $("materialSelect").addEventListener("change", async () => {
    await selectMaterial(Number($("materialSelect").value));
  });

  // ---------- Architecture visualisation ----------
  $("toggleArch").addEventListener("click", () => {
    const panel = $("architecturePanel");
    const isHidden = panel.hidden;
    panel.hidden = !isHidden;
    $("toggleArch").textContent = isHidden ? "Hide pipeline" : "Show pipeline";
    if (!isHidden && state.flowTimer) {
      window.clearInterval(state.flowTimer);
      state.flowTimer = null;
      $("pipelineFlow").querySelectorAll(".is-flow-active").forEach((stage) => stage.classList.remove("is-flow-active"));
    }
    if (isHidden && state.archData) {
      renderArchitecture(state.archData, state.mode);
    }
  });

  function renderArchitecture(data, mode) {
    const arch = data.architectures[mode];
    if (!arch) return;

    $("archModeName").textContent = `${modeName(mode)} Pipeline`;
    $("archModeTag").textContent = mode.toUpperCase();

    // Render pipeline flow
    const flow = $("pipelineFlow");
    flow.innerHTML = "";
    const steps = arch.pipeline_steps || [];

    // For hybrid mode, group the parallel search steps
    if (mode === "hybrid") {
      renderHybridFlow(flow, steps);
    } else {
      renderStandardFlow(flow, steps);
    }
    startFlowAnimation(flow);

    // Render retrieval details
    const details = $("retrievalDetails");
    if (mode === "hybrid") {
      const methods = arch.retrieval?.methods || [];
      const fusion = arch.retrieval?.fusion || {};
      details.innerHTML = methods.map((m) => `
        <div class="rag-detail-card">
          <h4>${S.esc(m.name.replace(/_/g, " "))}</h4>
          <p>${S.esc(m.description)}</p>
          <p>${Object.entries(m.parameters || {}).map(([k, v]) => `<code>${S.esc(k)}: ${S.esc(v)}</code>`).join(" ")}</p>
        </div>
      `).join("") + `
        <div class="rag-detail-card">
          <h4>${S.esc(fusion.method?.replace(/_/g, " ") || "Fusion")}</h4>
          <p>${S.esc(fusion.description || "")}</p>
          <p><code>k = ${S.esc(fusion.parameter_k || 60)}</code></p>
        </div>
      `;
    } else if (mode === "adaptive") {
      const retrieval = arch.retrieval || {};
      details.innerHTML = `
        <div class="rag-detail-card">
          <h4>Observed learner state</h4>
          <p>${S.esc((retrieval.state_features || []).join(" · "))}</p>
          <p>${S.esc(retrieval.status || "")}</p>
        </div>
        <div class="rag-detail-card">
          <h4>Reranking rule</h4>
          <p>${S.esc(retrieval.state_scoring || "")}</p>
          <p>${S.esc(retrieval.intent_rule || "")}</p>
        </div>
      `;
    } else {
      const retrieval = arch.retrieval || {};
      details.innerHTML = `
        <div class="rag-detail-card">
          <h4>${S.esc(retrieval.method || "BM25")}</h4>
          <p>${S.esc(retrieval.description || "")}</p>
          <p>${Object.entries(retrieval.parameters || {}).map(([k, v]) => `<code>${S.esc(k)}: ${S.esc(v)}</code>`).join(" ")}</p>
        </div>
        <div class="rag-detail-card">
          <h4>Chunking Strategy</h4>
          <p>Sliding window: ${arch.chunking?.window_size || 210} words, ${arch.chunking?.overlap || 35} word overlap</p>
        </div>
      `;
    }

    // Render comparison
    const compData = data.architectures;
    const comparison = $("archComparison");
    comparison.innerHTML = `
      <div class="rag-compare-card ${mode === 'standard' ? 'is-active' : ''}">
        <h4>🔍 Standard RAG</h4>
        ${(compData.standard?.retrieval ? [compData.standard.retrieval] : []).map(() => "").join("")}
        <p class="rag-pro">Fast keyword matching</p>
        <p class="rag-pro">Transparent term matching</p>
        <p class="rag-pro">Low memory footprint</p>
        <p class="rag-con">Misses synonyms and paraphrases</p>
        <p class="rag-con">No semantic understanding</p>
      </div>
      <div class="rag-compare-card ${mode === 'hybrid' ? 'is-active' : ''}">
        <h4>⚡ Hybrid RAG</h4>
        <p class="rag-pro">Keyword + semantic matching</p>
        <p class="rag-pro">RRF fusion balances precision & recall</p>
        <p class="rag-pro">Better paraphrase handling</p>
        <p class="rag-pro">Robust for diverse queries</p>
        <p class="rag-con">TF-IDF vectors are rebuilt at query time</p>
        <p class="rag-con">No persistent embedding/vector index</p>
        <p class="rag-con">More complex debug surface</p>
      </div>
      <div class="rag-compare-card ${mode === 'adaptive' ? 'is-active' : ''}">
        <h4>🎯 Recall-aware RAG</h4>
        <p class="rag-pro">Uses repeated material-quiz gaps</p>
        <p class="rag-pro">Uses time since a concept was quizzed</p>
        <p class="rag-pro">Shows why a study action was suggested</p>
        <p class="rag-con">Heuristic signals are not calibrated forgetting estimates</p>
        <p class="rag-con">No judged retrieval evaluation yet</p>
      </div>
    `;
  }

  function startFlowAnimation(container) {
    if (state.flowTimer) window.clearInterval(state.flowTimer);
    const stages = [...container.querySelectorAll(".rag-step")];
    if (stages.length < 2 || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    state.flowStep = 0;
    const activate = () => {
      stages.forEach((stage, index) => stage.classList.toggle("is-flow-active", index === state.flowStep));
      state.flowStep = (state.flowStep + 1) % stages.length;
    };
    activate();
    state.flowTimer = window.setInterval(activate, 1500);
  }

  function renderStandardFlow(container, steps) {
    steps.forEach((step, i) => {
      if (i > 0) {
        const connector = document.createElement("div");
        connector.className = "rag-step-connector";
        connector.innerHTML = "→";
        container.append(connector);
      }
      const el = document.createElement("div");
      el.className = "rag-step";
      const cls = STEP_CLASS[step.name] || "";
      el.innerHTML = `
        <div class="rag-step-node ${cls}">${STEP_ICONS[step.name] || "📌"}</div>
        <div class="rag-step-name">${S.esc(step.name)}</div>
        <div class="rag-step-detail">${S.esc(step.detail)}</div>
      `;
      container.append(el);
    });
  }

  function renderHybridFlow(container, steps) {
    // Steps 1-4 are sequential, 5-6 are parallel, 7-10 are sequential
    const preSteps = steps.filter((s) => s.step <= 4);
    const parallelSteps = steps.filter((s) => s.step === 5 || s.step === 6);
    const postSteps = steps.filter((s) => s.step >= 7);

    // Pre-parallel steps
    preSteps.forEach((step, i) => {
      if (i > 0) {
        const connector = document.createElement("div");
        connector.className = "rag-step-connector";
        connector.innerHTML = "→";
        container.append(connector);
      }
      const el = document.createElement("div");
      el.className = "rag-step";
      const cls = STEP_CLASS[step.name] || "";
      el.innerHTML = `
        <div class="rag-step-node ${cls}">${STEP_ICONS[step.name] || "📌"}</div>
        <div class="rag-step-name">${S.esc(step.name)}</div>
        <div class="rag-step-detail">${S.esc(step.detail)}</div>
      `;
      container.append(el);
    });

    // Connector before parallel block
    if (parallelSteps.length > 0) {
      const connector = document.createElement("div");
      connector.className = "rag-step-connector";
      connector.innerHTML = "→";
      container.append(connector);

      // Parallel search block
      const parallel = document.createElement("div");
      parallel.className = "rag-parallel-block";
      parallel.innerHTML = `
        <span class="rag-parallel-label">Parallel retrieval</span>
        <div class="rag-parallel-rows">
          ${parallelSteps.map((s) => `
            <div class="rag-parallel-row">
              <span class="rag-parallel-icon">${STEP_ICONS[s.name] || "📌"}</span>
              <div class="rag-parallel-text">
                <span class="rag-parallel-name">${S.esc(s.name)}</span>
                <span class="rag-parallel-desc">${S.esc(s.detail)}</span>
              </div>
            </div>
          `).join("")}
        </div>
      `;
      container.append(parallel);
    }

    // Post-parallel steps
    postSteps.forEach((step) => {
      const connector = document.createElement("div");
      connector.className = "rag-step-connector";
      connector.innerHTML = "→";
      container.append(connector);
      const el = document.createElement("div");
      el.className = "rag-step";
      const cls = STEP_CLASS[step.name] || "";
      el.innerHTML = `
        <div class="rag-step-node ${cls}">${STEP_ICONS[step.name] || "📌"}</div>
        <div class="rag-step-name">${S.esc(step.name)}</div>
        <div class="rag-step-detail">${S.esc(step.detail)}</div>
      `;
      container.append(el);
    });
  }

  // ---------- Chat ----------
  function resetChat() {
    $("chatMessages").innerHTML = `
      <div class="rag-chat-welcome">
        <div class="rag-welcome-icon">🤖</div>
        <h3>Ready to help you study</h3>
        <p>Ask any question about your selected study material. I'll find the most relevant passages and provide cited answers.</p>
        <div class="rag-welcome-suggestions" id="suggestions">
          <button class="rag-suggestion" type="button" data-q="Summarize the key concepts">Summarize key concepts</button>
          <button class="rag-suggestion" type="button" data-q="What are the main topics covered?">Main topics covered</button>
          <button class="rag-suggestion" type="button" data-q="Explain the most important ideas">Important ideas</button>
        </div>
      </div>
    `;
    bindSuggestions();
  }

  function bindSuggestions() {
    document.querySelectorAll(".rag-suggestion").forEach((btn) => {
      btn.addEventListener("click", () => {
        $("chatInput").value = btn.dataset.q;
        $("chatForm").dispatchEvent(new Event("submit", { cancelable: true }));
      });
    });
  }
  bindSuggestions();

  function addMessage(kind, html) {
    if (!state.chatStarted) {
      $("chatMessages").innerHTML = "";
      state.chatStarted = true;
    }
    const msg = document.createElement("article");
    msg.className = `rag-msg ${kind}`;
    msg.innerHTML = html;
    $("chatMessages").append(msg);
    msg.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function addTypingIndicator() {
    const typing = document.createElement("div");
    typing.className = "rag-typing";
    typing.id = "typingIndicator";
    typing.innerHTML = '<div class="rag-typing-dot"></div><div class="rag-typing-dot"></div><div class="rag-typing-dot"></div>';
    $("chatMessages").append(typing);
    typing.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function removeTypingIndicator() {
    const el = $("typingIndicator");
    if (el) el.remove();
  }

  function citationsHTML(citations) {
    if (!citations || !citations.length) return "";
    return `
      <details class="rag-msg-citations">
        <summary>📎 ${citations.length} source${citations.length === 1 ? "" : "s"} cited</summary>
        ${citations.map((c) => `
          <div class="rag-msg-citation">
            <span>Passage ${S.esc(c.chunk_index + 1)}</span>
            ${S.esc(c.excerpt)}
          </div>
        `).join("")}
      </details>
    `;
  }

  async function compareModes() {
    if (!state.active) throw new Error("Choose a material first.");
    const question = $("chatInput").value.trim() || state.lastQuestion;
    if (question.length < 3) throw new Error("Ask a question first, then compare retrieval strategies.");
    const panel = $("modeComparison");
    panel.hidden = false;
    panel.innerHTML = '<p class="muted">Running the same question through all three retrieval strategies…</p>';
    const result = await S.api(`/rag/${state.active.id}/compare`, { method: "POST", body: { question } });
    const help = {
      latency: "Retrieval elapsed time: 1000 × (time after retrieval − time before retrieval), in milliseconds. Excludes generation and network delay.",
      retrieved: "Number of passages returned in the top-six result set.",
      top: "Final ranking score for the first retrieved passage. Query-relative diagnostic, not a probability.",
      mean: "Mean final ranking score of returned top-six passages: sum of their scores ÷ number returned.",
      support: "Learner feedback proxy: answers marked Yes ÷ all rated answers for this material and strategy. Not objective accuracy.",
      accuracy: "Answer accuracy = human-judged correct answers ÷ all human-judged answers. No labeled benchmark is configured, so this is not evaluated.",
      confidence: "Calibrated confidence needs predicted probabilities checked against labeled outcomes (for example, a reliability curve or Brier score). This system does not produce calibrated answer probabilities.",
    };
    const label = (text, formula) => `<abbr class="metric-help" tabindex="0" title="${S.esc(formula)}">${text} ⓘ</abbr>`;
    panel.innerHTML = `<div class="rag-comparison-heading"><div><h3>Retrieval strategy comparison</h3><p class="muted">Same question: “${S.esc(result.question)}”</p></div><button type="button" class="btn small ghost" id="closeModeComparison">Close</button></div>
      <p class="rag-comparison-note">Retrieval is measured; answer generation is not repeated. Ranking scores are method-specific diagnostics, not confidence probabilities. Hover or focus ⓘ for metric definitions and formulas.</p>
      <div class="rag-metrics-scroll"><table class="rag-metrics-table"><thead><tr><th>Strategy</th><th>${label("Latency", help.latency)}</th><th>${label("Passages", help.retrieved)}</th><th>${label("Top score", help.top)}</th><th>${label("Mean top-6", help.mean)}</th><th>${label("Source support", help.support)}</th><th>${label("Accuracy", help.accuracy)}</th><th>${label("Confidence", help.confidence)}</th></tr></thead><tbody>
      ${result.modes.map((mode) => `<tr><th>${S.esc(modeShort(mode.mode))}</th><td>${Number(mode.retrieval_latency_ms).toFixed(2)} ms</td><td>${mode.retrieved_count}</td><td>${Number(mode.top_rank_score).toFixed(4)}</td><td>${Number(mode.mean_top_k_rank_score).toFixed(4)}</td><td>${mode.source_support_rate === null ? "Not rated" : `${(mode.source_support_rate * 100).toFixed(0)}%`}<small>${mode.source_support_votes} rating(s)</small></td><td>Not evaluated</td><td>Not calibrated</td></tr>`).join("")}
      </tbody></table></div><div class="rag-compare-passages">${result.modes.map((mode) => `<details><summary>${S.esc(modeShort(mode.mode))}: top retrieved passages</summary>${mode.top_passages.slice(0, 3).map((passage, index) => `<p><strong>#${passage.chunk_index + 1} · rank ${index + 1} · ${Number(passage.rank_score).toFixed(4)}</strong><br>${S.esc(passage.excerpt)}</p>`).join("") || '<p class="muted">No passages.</p>'}</details>`).join("")}</div>`;
    $("closeModeComparison").addEventListener("click", () => { panel.hidden = true; });
  }

  $("compareModes").addEventListener("click", () => run(compareModes));

  $("chatMessages").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-interaction]");
    if (!button || !state.active) return;
    run(async () => {
      button.disabled = true;
      const interactionId = Number(button.dataset.interaction);
      const supported = button.dataset.supported === "true";
      await S.api(`/rag/${state.active.id}/interactions/${interactionId}/feedback`, {
        method: "POST", body: { supported_by_sources: supported },
      });
      button.closest(".rag-feedback").outerHTML = feedbackHTML(interactionId, supported);
      S.toast("Thanks — your source-support rating was saved.");
      if (!$("modeComparison").hidden) await compareModes();
    }).catch(() => { button.disabled = false; });
  });

  $("chatForm").addEventListener("submit", (event) => {
    event.preventDefault();
    run(async () => {
      if (!state.active) throw new Error("Choose a material first.");
      const question = $("chatInput").value.trim();
      if (!question) return;
      $("chatInput").value = "";
      $("chatSend").disabled = true;

      addMessage("user", `<p>${S.esc(question)}</p>`);
      addTypingIndicator();

      const result = await S.api(`/rag/${state.active.id}/chat`, {
        method: "POST",
        body: {
          question,
          mode: state.mode,
          history: state.chatHistory.slice(-8),
          show_diagnostics: state.showDiag,
        },
      });

      removeTypingIndicator();
      state.lastQuestion = question;

      const gapHTML = result.recall_gaps?.length
        ? `<p class="muted" style="font-size:12px;margin-top:6px">Recall compass considered: ${result.recall_gaps.map(S.esc).join(", ")}</p>`
        : "";
      const interventionHTML = result.intervention
        ? `<p class="notice" style="margin-top:10px"><strong>Study suggestion · ${S.esc(result.intervention.action.replace(/_/g, " "))}</strong><br>${S.esc(result.intervention.reason)}<br><small>${S.esc(result.intervention.basis)}</small></p>`
        : "";

      addMessage("assistant", `
        <p>${S.esc(result.answer)}</p>
        ${citationsHTML(result.citations)}
        ${gapHTML}
        ${interventionHTML}
        <div class="rag-msg-meta">
          <span class="rag-msg-tag">${S.esc(modeName(result.rag_mode))}</span>
          <span class="rag-msg-tag">${S.esc(result.retrieval_method)}</span>
          ${result.mode === "grounded_ai" ? '<span class="rag-msg-tag">AI generated</span>' : '<span class="rag-msg-tag">Extracted</span>'}
        </div>
        ${feedbackHTML(result.interaction_id, null)}
      `);

      state.chatHistory.push(
        { role: "user", content: question },
        { role: "assistant", content: result.answer },
      );

      if (result.diagnostics) {
        state.lastDiagnostics = result.diagnostics;
        renderDiagnostics(result.diagnostics);
      }

      $("chatSend").disabled = false;
    }).catch(() => {
      removeTypingIndicator();
      $("chatSend").disabled = false;
    });
  });

  // ---------- Diagnostics ----------
  $("showDiagnostics").addEventListener("click", () => {
    state.showDiag = !state.showDiag;
    $("showDiagnostics").textContent = state.showDiag ? "Hide retrieval scores" : "Show retrieval scores";
    if (!state.showDiag) {
      $("diagnosticsSection").hidden = true;
    } else if (state.lastDiagnostics) {
      renderDiagnostics(state.lastDiagnostics);
    }
  });

  $("closeDiagnostics").addEventListener("click", () => {
    $("diagnosticsSection").hidden = true;
    state.showDiag = false;
    $("showDiagnostics").textContent = "Show retrieval scores";
  });

  function renderDiagnostics(diagnostics) {
    if (!state.showDiag) return;
    $("diagnosticsSection").hidden = false;
    $("diagnosticsContent").innerHTML = `
      <table class="rag-diag-table">
        <thead>
          <tr>
            <th>Chunk</th>
            <th>BM25 Score</th>
            <th>Semantic Score</th>
            <th>Combined</th>
            <th>Gap Boost</th>
            <th>Recency Boost</th>
            <th>Final</th>
            <th>Method</th>
          </tr>
        </thead>
        <tbody>
          ${diagnostics.map((d) => `
            <tr>
              <td>#${d.chunk_index + 1}</td>
              <td>
                <div class="rag-score-bar">
                  <div class="rag-score-track"><div class="rag-score-fill bm25" style="width:${Math.round(d.bm25_score * 100)}%"></div></div>
                  <span class="rag-score-val">${(d.bm25_score * 100).toFixed(0)}%</span>
                </div>
              </td>
              <td>
                <div class="rag-score-bar">
                  <div class="rag-score-track"><div class="rag-score-fill semantic" style="width:${Math.round(d.semantic_score * 100)}%"></div></div>
                  <span class="rag-score-val">${(d.semantic_score * 100).toFixed(0)}%</span>
                </div>
              </td>
              <td>
                <div class="rag-score-bar">
                  <div class="rag-score-track"><div class="rag-score-fill combined" style="width:${Math.round(d.combined_score * 100)}%"></div></div>
                  <span class="rag-score-val">${(d.combined_score * 100).toFixed(0)}%</span>
                </div>
              </td>
              <td>${d.gap_boost > 0 ? `+${(d.gap_boost * 100).toFixed(0)}%` : "—"}</td>
              <td>${d.recency_boost > 0 ? `+${(d.recency_boost * 100).toFixed(0)}%` : "—"}</td>
              <td><strong>${(d.final_score * 100).toFixed(0)}%</strong></td>
              <td><span class="tag">${S.esc(d.retrieval_method)}</span></td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    $("diagnosticsSection").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ---------- Study Tools: Notes ----------
  $("makeRagNotes").addEventListener("click", () => run(async () => {
    if (!state.active) return;
    $("makeRagNotes").disabled = true;
    $("makeRagNotes").textContent = "Generating…";
    const result = await S.api(`/rag/${state.active.id}/notes`, {
      method: "POST",
      body: { mode: state.mode },
    });
    $("ragNotesOutput").hidden = false;
    $("ragNotesOutput").innerHTML = `
      <div class="rag-notes-output">
        <div class="panel-head">
          <span class="tag">${S.esc(modeName(result.rag_mode))}</span>
          <span class="tag">${result.mode === "extractive" ? "Extracted" : "AI generated"}</span>
        </div>
        ${result.intervention ? `<p class="notice"><strong>${S.esc(result.intervention.action.replace(/_/g, " "))}</strong> · ${S.esc(result.intervention.reason)}</p>` : ""}
        ${result.notes.map((n) => `
          <div class="rag-note">
            <h4>${S.esc(n.heading)}</h4>
            <p>${S.esc(n.text)}</p>
            ${citationsHTML(n.citations)}
          </div>
        `).join("")}
      </div>
    `;
    $("makeRagNotes").disabled = false;
    $("makeRagNotes").textContent = "Regenerate notes";
    S.toast("Study notes generated.");
  }));

  // ---------- Study Tools: Quiz ----------
  $("makeRagQuiz").addEventListener("click", () => run(async () => {
    if (!state.active) return;
    $("makeRagQuiz").disabled = true;
    $("makeRagQuiz").textContent = "Generating…";
    const count = Number($("ragQuizCount").value);
    const result = await S.api(`/rag/${state.active.id}/quiz`, {
      method: "POST",
      body: { question_count: count, mode: state.mode },
    });
    $("ragQuizOutput").hidden = false;
    $("ragQuizOutput").innerHTML = `
      <div class="rag-quiz-output">
        <div class="panel-head">
          <span class="tag">${S.esc(modeName(result.rag_mode))}</span>
          <span class="tag">${S.esc(result.retrieval_method)}</span>
        </div>
        ${result.recall_gaps?.length ? `<p class="notice" style="margin-bottom:12px">Recall compass prioritised: ${result.recall_gaps.map(S.esc).join(", ")}</p>` : ""}
        ${result.intervention ? `<p class="notice" style="margin-bottom:12px"><strong>${S.esc(result.intervention.action.replace(/_/g, " "))}</strong> · ${S.esc(result.intervention.reason)}</p>` : ""}
        ${result.questions.map((q) => `
          <div class="rag-quiz-question">
            <span class="eyebrow">${S.esc(q.concept)}</span>
            <strong>${S.esc(q.question)}</strong>
            ${citationsHTML([q.citation])}
            <details style="margin-top:8px">
              <summary class="muted" style="cursor:pointer;font-size:13px">Show answer</summary>
              <p style="margin-top:6px;padding:10px 12px;background:var(--surface-2);border-radius:8px;font-size:14px">${S.esc(q.answer)}</p>
            </details>
          </div>
        `).join("")}
      </div>
    `;
    $("makeRagQuiz").disabled = false;
    $("makeRagQuiz").textContent = "Generate quiz";
    S.toast(`${result.questions.length} quiz questions generated.`);
  }));

  // ---------- Initialise ----------
  await run(async () => {
    await loadMaterials();
    const student = await S.currentStudent();
    state.studentId = student.id;
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(`ssr.rag.setup.${state.studentId}`) || "{}"); } catch (error) { /* Use defaults. */ }
    if (["standard", "hybrid", "adaptive"].includes(saved.mode)) setMode(saved.mode);
    if (saved.material_id && state.materials.some((material) => material.id === Number(saved.material_id))) {
      $("materialSelect").value = String(saved.material_id);
      await selectMaterial(Number(saved.material_id));
    }
  });
});
