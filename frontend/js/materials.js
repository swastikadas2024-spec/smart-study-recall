/* Shared course library, source-cited study assets, and recall-gap study chat. */
document.addEventListener("DOMContentLoaded", async () => {
  const S = window.SSR;
  if (!S.requireLogin()) return;
  const $ = (id) => document.getElementById(id);
  const state = { me: null, materials: [], active: null, quiz: null, chatStarted: false, chatHistory: [] };
  const run = async (work) => {
    try { await work(); } catch (err) { S.showError("pageError", err); }
  };
  const form = (values) => {
    const body = new FormData();
    Object.entries(values).forEach(([key, value]) => body.append(key, value));
    return body;
  };

  function renderLibrary() {
    $("materialCount").textContent = `${state.materials.length} material${state.materials.length === 1 ? "" : "s"}`;
    $("emptyMaterials").hidden = state.materials.length > 0;
    $("materialList").innerHTML = state.materials.map((m) => `
      <article class="panel material-card">
        <div class="panel-head"><span class="tag">${S.esc(m.subject || "Course material")}</span><span class="muted">${m.published ? "Published" : "Private"}</span></div>
        <h3>${S.esc(m.title)}</h3><p class="muted">${S.esc(m.filename)} · ${m.chunk_count} evidence passages</p>
        <div class="topic-actions"><button class="btn small primary" type="button" data-open="${m.id}">Study this</button>${m.can_manage ? `<button class="btn small ghost danger" type="button" data-delete="${m.id}">Delete</button>` : ""}</div>
      </article>`).join("");
  }

  function citationsHTML(citations = []) {
    if (!citations.length) return "";
    return `<div class="citations"><strong>Sources</strong>${citations.map((c) => `<blockquote><span>Passage ${S.esc(c.chunk_index + 1)}</span><p>${S.esc(c.excerpt)}</p></blockquote>`).join("")}</div>`;
  }

  async function load() {
    state.me = await S.currentStudent();
    const materials = await S.api("/materials");
    state.materials = materials;
    renderLibrary();
  }

  $("uploadForm").addEventListener("submit", (event) => {
    event.preventDefault();
    run(async () => {
      const f = event.currentTarget;
      const file = f.elements.file.files[0];
      if (!file) throw new Error("Choose a study material to upload.");
      $("uploadButton").disabled = true;
      const response = await S.api("/materials", {
        method: "POST",
        body: form({
          title: f.elements.title.value.trim(),
          subject: f.elements.subject.value.trim(),
          published: String(f.elements.published.checked),
          file,
        }),
      });
      f.reset();
      f.elements.published.checked = true;
      S.toast(`Indexed ${response.title} into ${response.chunk_count} source passages.`);
      await load();
      await openMaterial(response.id);
    }).finally(() => { $("uploadButton").disabled = false; });
  });

  async function openMaterial(id) {
    const material = state.materials.find((item) => item.id === id);
    if (!material) return;
    state.active = material;
    state.quiz = null;
    $("studyWorkspace").hidden = false;
    $("workspaceTitle").textContent = material.title;
    $("workspaceStatus").textContent = "Material ready. Notes, quizzes, and chat are grounded in its indexed passages.";
    $("notesOutput").hidden = true;
    $("quizOutput").hidden = true;
    $("chatMessages").innerHTML = "";
    state.chatStarted = false;
    state.chatHistory = [];
    $("studyWorkspace").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  $("materialList").addEventListener("click", (event) => {
    const open = event.target.closest("[data-open]");
    const remove = event.target.closest("[data-delete]");
    if (open) run(() => openMaterial(Number(open.dataset.open)));
    if (remove) run(async () => {
      if (!confirm("Delete this material and its generated study data?")) return;
      await S.api(`/materials/${Number(remove.dataset.delete)}`, { method: "DELETE" });
      if (state.active?.id === Number(remove.dataset.delete)) $("studyWorkspace").hidden = true;
      S.toast("Material deleted.");
      await load();
    });
  });

  $("makeNotes").addEventListener("click", () => run(async () => {
    if (!state.active) return;
    $("workspaceStatus").textContent = "Searching the material and generating cited notes… Usually under 30 seconds.";
    const result = await S.api(`/materials/${state.active.id}/notes`, { method: "POST", body: {} });
    $("notesOutput").hidden = false;
    $("notesOutput").innerHTML = `<div class="panel-head"><h3>Study notes</h3><span class="tag">${result.mode === "extractive" ? "Extracted from source" : "AI generated"}</span></div>${result.notes.map((n) => `<article class="note-item"><h4>${S.esc(n.heading)}</h4><p>${S.esc(n.text)}</p>${citationsHTML(n.citations)}</article>`).join("")}`;
    $("workspaceStatus").textContent = result.mode === "extractive"
      ? "Gemini did not respond; these notes were extracted from the cited source passages."
      : "Notes cite the passages they came from.";
  }));

  function renderQuiz(quiz) {
    state.quiz = quiz;
    $("quizOutput").hidden = false;
    const gaps = quiz.recall_gaps?.length ? `<p class="notice">Recall Compass prioritized previous misses: ${quiz.recall_gaps.map(S.esc).join(", ")}</p>` : "";
    $("quizOutput").innerHTML = `<div class="panel-head"><h3>Source-based quiz</h3><span class="tag">${quiz.mode === "recall_gap_guided" ? "Recall Compass" : "Practice"}</span></div>${gaps}<form id="quizForm" class="quiz-form">${quiz.questions.map((q) => `<label class="quiz-question"><span class="eyebrow">${S.esc(q.concept)}</span><strong>${S.esc(q.question)}</strong><input name="answer-${q.index}" autocomplete="off" required placeholder="Your answer" />${citationsHTML([q.citation])}</label>`).join("")}<button class="btn primary" type="submit">Check answers</button></form><div id="quizResults" aria-live="polite"></div>`;
    $("quizForm").addEventListener("submit", (event) => {
      event.preventDefault();
      run(async () => {
        const answers = quiz.questions.map((q) => event.currentTarget.elements[`answer-${q.index}`].value);
        const result = await S.api(`/materials/${state.active.id}/quiz/${quiz.session_id}/submit`, { method: "POST", body: { answers } });
        $("quizForm").hidden = true;
        $("quizResults").innerHTML = `<div class="score-card"><p class="eyebrow">Your score</p><strong>${result.score}%</strong><p>${result.correct} of ${result.total} correct</p></div>${result.results.map((r) => `<p class="quiz-result ${r.correct ? "is-correct" : "is-missed"}">${r.correct ? "Correct" : "Review"} · ${S.esc(r.concept)} — ${S.esc(r.expected_answer)}</p>`).join("")}${result.recall_gaps.length ? `<p class="notice">Recall Compass will revisit: ${result.recall_gaps.map(S.esc).join(", ")}</p>` : "<p class=\"notice\">No weak concepts detected from this material yet.</p>"}`;
        $("workspaceStatus").textContent = "Quiz performance saved. Future retrieval will use these concept-level signals.";
      });
    });
  }

  $("makeQuiz").addEventListener("click", () => run(async () => {
    if (!state.active) return;
    $("workspaceStatus").textContent = "Searching the material and generating a source-based quiz… Usually under 30 seconds.";
    const result = await S.api(`/materials/${state.active.id}/quiz`, { method: "POST", body: { question_count: 5 } });
    renderQuiz(result);
    $("workspaceStatus").textContent = "Answer in your own words. Correct answers are kept hidden until submission.";
  }));

  function addChatMessage(kind, html) {
    if (!state.chatStarted) { $("chatMessages").innerHTML = ""; state.chatStarted = true; }
    const article = document.createElement("article");
    article.className = `chat-message ${kind}`;
    article.innerHTML = html;
    $("chatMessages").append(article);
    article.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  $("chatForm").addEventListener("submit", (event) => {
    event.preventDefault();
    run(async () => {
      if (!state.active) throw new Error("Choose a material first.");
      const question = $("chatQuestion").value.trim();
      $("chatQuestion").value = "";
      addChatMessage("user", `<p>${S.esc(question)}</p>`);
      $("workspaceStatus").textContent = "Searching the material and preparing a cited answer… Usually under 30 seconds.";
      const result = await S.api(`/materials/${state.active.id}/chat`, { method: "POST", body: { question, history: state.chatHistory.slice(-8) } });
      const gap = result.recall_gap_context?.length ? `<p class="muted">Recall Compass considered: ${result.recall_gap_context.map(S.esc).join(", ")}</p>` : "";
      addChatMessage("assistant", `<p>${S.esc(result.answer)}</p>${citationsHTML(result.citations)}${gap}`);
      state.chatHistory.push({ role: "user", content: question }, { role: "assistant", content: result.answer });
      $("workspaceStatus").textContent = result.mode === "extractive"
        ? "Gemini did not respond; this answer uses retrieved source passages."
        : "Answer grounded in retrieved material.";
    });
  });

  await run(async () => {
    await load();
    $("providerNotice").hidden = false;
    $("providerNotice").textContent = "Any signed-in user can upload. If no AI provider is configured, notes and answers use extracted source passages; add AI_API_KEY and AI_MODEL to the backend environment for generated notes, questions, and responses.";
  });
});
