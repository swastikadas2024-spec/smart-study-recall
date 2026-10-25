/* Topics: subjects, flashcards (this browser), sessions, forecasts and recall checks. */
document.addEventListener("DOMContentLoaded", async () => {
  const S = window.SSR;
  if (!S.requireLogin()) return;
  const $ = (id) => document.getElementById(id);
  const { esc, pct } = S;
  const L = window.SSR_LOCAL;
  const val = (form, name) => form.querySelector(`[name="${name}"]`).value;
  let me = null;
  let topics = [];
  let events = [];

  function statsFor(topicId) {
    const evs = events.filter((e) => e.topic_id === topicId && e.questions_seen > 0);
    const seen = evs.reduce((a, e) => a + e.questions_seen, 0);
    const correct = evs.reduce((a, e) => a + e.questions_correct, 0);
    return { sessions: evs.length, seen, accuracy: seen ? (correct / seen) * 100 : null };
  }

  function forecastLine(topicId) {
    return `<div class="forecast-line" id="topicForecast-${topicId}" aria-live="polite"><span class="muted">Checking next-session score scenarios…</span></div>`;
  }

  function scenarioMarkup(result) {
    if (!result.available) return `<span class="muted">${esc(result.reason)}</span>`;
    const options = (result.scenarios || []).map((scenario) => `
      <li class="row"><span>Review in ${esc(scenario.review_in_days)} day${Number(scenario.review_in_days) === 1 ? "" : "s"}</span>
      <strong>${pct(Number(scenario.predicted_next_session_recall) * 100)}</strong></li>`).join("");
    return `<span class="tag estimate">Next-session score · experimental</span>
      <strong>Review in ${esc(result.recommended_review_in_days)} days</strong>
      <span>Expected next-session score ${pct(Number(result.recommended_predicted_score) * 100)}</span>
      ${options ? `<details><summary>Compare candidate intervals</summary><ul class="rows scenario-list">${options}</ul></details>` : ""}
      <span class="muted">This is not a long-term memory probability.</span>`;
  }

  async function refreshPredictions() {
    await Promise.all(topics.map(async (topic) => {
      const target = document.getElementById(`topicForecast-${topic.id}`);
      if (!target) return;
      try {
        target.innerHTML = scenarioMarkup(await S.api(`/topic-recall/${topic.id}`));
      } catch (error) {
        target.innerHTML = `<span class="muted">${esc(error.message)}</span>`;
      }
    }));
  }

  function render() {
    $("topicCount").textContent = `${topics.length} topic${topics.length === 1 ? "" : "s"}`;
    $("emptyTopics").hidden = topics.length > 0;

    $("topicList").innerHTML = topics.map((t) => {
      const st = statsFor(t.id);
      const card = L.cards(me.id)[String(t.id)];
      const cardHint = card
        ? `<p class="topic-meta">Card: ${esc(card.question)}</p>`
        : `<p class="topic-meta">No flashcard yet.</p>`;
      return `<article class="topic reveal">
        <div>
          <h3 class="topic-name">${esc(t.topic_name)}</h3>
          <p class="topic-meta">${esc(t.subject || "No subject")}</p>
        </div>
        <div>
          <div class="meter-row"><span>Accuracy</span><span>${pct(st.accuracy)}</span></div>
          <div class="meter"><span style="width:${st.accuracy ?? 0}%"></span></div>
        </div>
        <p class="topic-meta">${st.sessions} session${st.sessions === 1 ? "" : "s"} · ${st.seen} questions</p>
        ${forecastLine(t.id)}
        ${cardHint}
        <div class="recall-out" hidden></div>
        <div class="topic-foot">
          <div class="topic-actions">
            <button class="btn small primary" type="button" data-action="log" data-id="${t.id}">Log session</button>
            <button class="btn small" type="button" data-action="card" data-id="${t.id}">${card ? "Edit card" : "Add card"}</button>
            <button class="btn small" type="button" data-action="recall" data-id="${t.id}">Check recall</button>
          </div>
          <div class="topic-actions">
            <button class="btn small ghost" type="button" data-action="edit" data-id="${t.id}">Edit</button>
            <button class="btn small ghost danger" type="button" data-action="delete" data-id="${t.id}">Delete</button>
          </div>
        </div>
      </article>`;
    }).join("");

    const select = $("sessionTopic");
    const previous = select.value;
    select.innerHTML = topics.map((t) => `<option value="${t.id}">${esc(t.topic_name)}</option>`).join("");
    if (previous && topics.some((t) => String(t.id) === previous)) select.value = previous;
    const hasTopics = topics.length > 0;
    select.disabled = !hasTopics;
    $("sessionForm").querySelector("button[type=submit]").disabled = !hasTopics;
    window.SSR_FX.animateCounts();
    void refreshPredictions();
  }

  async function load() {
    me = await S.currentStudent();
    [topics, events] = await Promise.all([
      S.api("/topics"),
      S.api(`/learning-events?student_id=${me.id}`),
    ]);
    L.pruneCards(me.id, topics.map((t) => t.id));
    render();
  }

  async function run(work) {
    try {
      await work();
    } catch (err) {
      S.toast(err.message, "error");
    }
  }

  // Add a topic.
  $("topicForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.target;
    const name = val(f, "topic_name").trim();
    const subject = val(f, "subject").trim();
    if (!name) return S.toast("Give the topic a name.", "error");
    run(async () => {
      await S.api("/topics", { method: "POST", body: { student_id: me.id, topic_name: name, subject: subject || null } });
      f.reset();
      S.toast("Topic added.");
      await load();
    });
  });

  // Log a learning session.
  $("sessionForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.target;
    const topicId = Number(val(f, "topic_id"));
    const rawSeen = val(f, "questions_seen").trim();
    const rawCorrect = val(f, "questions_correct").trim();
    const seen = Number(rawSeen);
    const correct = Number(rawCorrect);

    if (!Number.isInteger(topicId) || topicId <= 0) return S.toast("Choose a topic first.", "error");
    if (rawSeen === "" || !Number.isInteger(seen) || seen < 1) return S.toast("Questions attempted must be a whole number, at least 1.", "error");
    if (rawCorrect === "" || !Number.isInteger(correct) || correct < 0) return S.toast("Correct answers must be a whole number, 0 or more.", "error");
    if (correct > seen) return S.toast("Correct answers cannot exceed questions attempted.", "error");

    run(async () => {
      await S.api("/learning-events", {
        method: "POST",
        body: { student_id: me.id, topic_id: topicId, questions_seen: seen, questions_correct: correct },
      });
      f.reset();
      S.toast("Session saved.");
      await load();
    });
  });

  // Card actions.
  $("topicList").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;
    const id = Number(btn.dataset.id);
    const topic = topics.find((t) => t.id === id);
    if (!topic) return;

    if (btn.dataset.action === "log") {
      $("sessionTopic").value = String(id);
      $("sessionForm").scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }

    if (btn.dataset.action === "card") {
      const f = $("cardForm");
      const card = L.cards(me.id)[String(id)] || { question: "", answer: "" };
      f.querySelector('[name="topic_id"]').value = String(id);
      f.querySelector('[name="question"]').value = card.question;
      f.querySelector('[name="answer"]').value = card.answer;
      $("cardTitle").textContent = topic.topic_name;
      $("cardDialog").showModal();
      return;
    }

    if (btn.dataset.action === "recall") {
      const out = btn.closest(".topic").querySelector(".recall-out");
      btn.disabled = true;
      out.hidden = false;
      out.innerHTML = `<p class="muted">Checking…</p>`;
      S.api(`/topic-recall/${id}`)
        .then((r) => { out.innerHTML = S.recallHTML(r); })
        .catch((err) => { out.innerHTML = `<p class="muted">${esc(err.message)}</p>`; })
        .finally(() => { btn.disabled = false; });
      return;
    }

    if (btn.dataset.action === "edit") {
      const f = $("editForm");
      f.querySelector('[name="id"]').value = String(topic.id);
      f.querySelector('[name="topic_name"]').value = topic.topic_name;
      f.querySelector('[name="subject"]').value = topic.subject || "";
      $("editDialog").showModal();
      return;
    }

    if (btn.dataset.action === "delete") {
      const ok = confirm(
        `Delete "${topic.topic_name}"? Its revision plans and flashcard will be removed. ` +
        "Past sessions still count in your totals but lose their topic link."
      );
      if (!ok) return;
      run(async () => {
        await S.api(`/topics/${id}`, { method: "DELETE" });
        L.deleteCard(me.id, id);
        S.toast("Topic deleted.");
        await load();
      });
    }
  });

  // Flashcard dialog.
  $("cardCancel").addEventListener("click", () => $("cardDialog").close());
  $("cardDelete").addEventListener("click", () => {
    const id = Number(val($("cardForm"), "topic_id"));
    run(async () => {
      L.deleteCard(me.id, id);
      $("cardDialog").close();
      S.toast("Flashcard removed.");
      render();
    });
  });
  $("cardForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.target;
    const id = Number(val(f, "topic_id"));
    const q = val(f, "question").trim();
    const a = val(f, "answer").trim();
    if (!q || !a) return S.toast("Write both a question and an answer.", "error");
    run(async () => {
      L.setCard(me.id, id, q, a);
      $("cardDialog").close();
      S.toast("Flashcard saved.");
      render();
    });
  });

  // Edit dialog.
  $("editCancel").addEventListener("click", () => $("editDialog").close());
  $("editForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.target;
    const id = val(f, "id");
    const name = val(f, "topic_name").trim();
    const subject = val(f, "subject").trim();
    if (!name) return S.toast("Topic name cannot be empty.", "error");
    run(async () => {
      await S.api(`/topics/${id}`, { method: "PATCH", body: { topic_name: name, subject: subject || null } });
      $("editDialog").close();
      S.toast("Topic updated.");
      await load();
    });
  });

  try {
    await load();
  } catch (err) {
    S.showError("pageError", err);
  }
});
