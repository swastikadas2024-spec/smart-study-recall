/* Study: focus timer (saved in this browser) and the recall quiz (saved to the backend). */
document.addEventListener("DOMContentLoaded", async () => {
  const S = window.SSR;
  if (!S.requireLogin()) return;
  const $ = (id) => document.getElementById(id);
  const { esc, pct } = S;
  const M = window.SSR_MODEL;
  const L = window.SSR_LOCAL;
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  let me = null;
  let topics = [];
  let events = [];

  // ---------- Focus timer ----------
  const POMODORO = { work: 25 * 60000, break: 5 * 60000 };
  const timer = { mode: "stopwatch", running: false, studiedMs: 0, phase: "work", phaseLeft: 0, lastTick: 0, handle: null };

  const fmtClock = (ms) => {
    const total = Math.max(0, Math.round(ms / 1000));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    const p = (n) => String(n).padStart(2, "0");
    return h ? `${p(h)}:${p(m)}:${p(s)}` : `${p(m)}:${p(s)}`;
  };

  function beep() {
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      const ctx = new Ctx();
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.frequency.value = 880;
      g.gain.value = 0.05;
      o.connect(g);
      g.connect(ctx.destination);
      o.start();
      o.stop(ctx.currentTime + 0.35);
    } catch (e) { /* audio unavailable: silent */ }
  }

  const countdownTarget = () => {
    const v = Number($("countdownMinutes").value);
    return clamp(Number.isFinite(v) ? v : 45, 1, 240) * 60000;
  };

  function renderReadout() {
    const phase = $("timerPhase");
    if (timer.mode === "stopwatch") {
      $("timerReadout").textContent = fmtClock(timer.studiedMs);
      phase.textContent = timer.running ? "Studying…" : timer.studiedMs ? "Paused" : "Ready when you are.";
    } else if (timer.mode === "countdown") {
      $("timerReadout").textContent = fmtClock(timer.phaseLeft);
      phase.textContent = timer.running ? "Focus." : timer.studiedMs ? "Paused" : "Set minutes, then start.";
    } else {
      $("timerReadout").textContent = fmtClock(timer.phaseLeft);
      phase.textContent = timer.phase === "work" ? (timer.running ? "Focus block" : "Focus block, ready") : "Break";
    }
  }

  function setButtons() {
    const hasTime = timer.studiedMs > 0;
    $("tStart").textContent = timer.running ? "Running" : hasTime ? "Resume" : "Start";
    $("tStart").disabled = timer.running;
    $("tPause").disabled = !timer.running;
    $("tStop").disabled = !(hasTime || timer.running);
  }

  function stopInterval() {
    if (timer.handle) clearInterval(timer.handle);
    timer.handle = null;
  }

  function resetTimer() {
    stopInterval();
    timer.running = false;
    timer.studiedMs = 0;
    timer.phase = "work";
    timer.phaseLeft = timer.mode === "countdown" ? countdownTarget() : POMODORO.work;
    setButtons();
    renderReadout();
  }

  function tick() {
    const now = Date.now();
    const delta = now - timer.lastTick;
    timer.lastTick = now;
    if (!timer.running) return;

    if (timer.mode === "stopwatch") {
      timer.studiedMs += delta;
    } else if (timer.mode === "countdown") {
      timer.phaseLeft -= delta;
      timer.studiedMs += delta;
      if (timer.phaseLeft <= 0) {
        timer.phaseLeft = 0;
        beep();
        pauseTimer();
        S.toast("Countdown complete. Saving your time.");
        saveStudy();
      }
    } else {
      timer.phaseLeft -= delta;
      if (timer.phase === "work") timer.studiedMs += delta;
      if (timer.phaseLeft <= 0) {
        beep();
        timer.phase = timer.phase === "work" ? "break" : "work";
        timer.phaseLeft = POMODORO[timer.phase];
      }
    }
    renderReadout();
  }

  function startTimer() {
    if (timer.mode === "countdown" && timer.studiedMs === 0) timer.phaseLeft = countdownTarget();
    timer.running = true;
    timer.lastTick = Date.now();
    stopInterval();
    timer.handle = setInterval(tick, 250);
    setButtons();
    renderReadout();
  }

  function pauseTimer() {
    timer.running = false;
    stopInterval();
    setButtons();
    renderReadout();
  }

  function saveStudy() {
    const topicId = Number($("studyTopic").value);
    const t = topics.find((x) => x.id === topicId);
    if (timer.studiedMs < 60000) {
      S.toast("Less than a minute is not saved.", "error");
      resetTimer();
      return;
    }
    if (!t) {
      S.toast("Choose a topic, then save.", "error");
      return;
    }
    const minutes = Math.max(1, Math.round(timer.studiedMs / 60000));
    L.addStudy(me.id, { topic_id: t.id, topic_name: t.topic_name, minutes, mode: timer.mode });
    S.toast(`Saved ${minutes} min to ${t.topic_name}.`);
    resetTimer();
    renderLog();
  }

  function setMode(mode) {
    if (timer.running) pauseTimer();
    timer.mode = mode;
    document.querySelectorAll(".mode-btn").forEach((b) => {
      const on = b.dataset.mode === mode;
      b.classList.toggle("is-active", on);
      b.setAttribute("aria-checked", String(on));
    });
    $("countdownSetup").hidden = mode !== "countdown";
    resetTimer();
  }

  // ---------- Study log (this browser) ----------
  function renderLog() {
    const logs = L.studyLog(me.id);
    const startToday = new Date(); startToday.setHours(0, 0, 0, 0);
    const today = logs.filter((s) => new Date(s.date).getTime() >= startToday.getTime());
    const todayMin = today.reduce((a, s) => a + s.minutes, 0);
    $("todayTotal").textContent = `${todayMin} min`;
    $("todayList").innerHTML = today.length
      ? today.slice().reverse().map((s) => `<li class="row">
          <div><strong>${esc(s.topic_name)}</strong><span class="muted">${esc(new Date(s.date).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }))} · ${esc(s.mode)}</span></div>
          <span class="muted">${esc(s.minutes)} min</span>
        </li>`).join("")
      : `<li class="empty-line">Nothing saved today yet.</li>`;

    const weekAgo = Date.now() - 7 * M.DAY_MS;
    const byTopic = new Map();
    logs.filter((s) => new Date(s.date).getTime() >= weekAgo).forEach((s) => {
      byTopic.set(s.topic_name, (byTopic.get(s.topic_name) || 0) + s.minutes);
    });
    const rows = [...byTopic.entries()].sort((a, b) => b[1] - a[1]);
    $("weekList").innerHTML = rows.length
      ? rows.map(([name, min]) => `<li class="row"><div><strong>${esc(name)}</strong></div><span class="muted">${min} min</span></li>`).join("")
      : `<li class="empty-line">No study time saved in the last 7 days.</li>`;
  }

  // ---------- Recall quiz ----------
  let quiz = null;

  const shuffle = (arr) => {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  };

  // Blank the longest word of the answer for fill-in-the-blank questions.
  function clozeOf(answer) {
    const words = answer.split(/\s+/);
    let best = 0;
    words.forEach((w, i) => {
      if (w.replace(/[^A-Za-z]/g, "").length > words[best].replace(/[^A-Za-z]/g, "").length) best = i;
    });
    return words.map((w, i) => (i === best ? "_____" : w)).join(" ");
  }

  function buildQuiz(count) {
    const cards = L.cards(me.id);
    const withCards = topics.filter((t) => cards[String(t.id)]);
    if (!withCards.length) return null;
    const useChoice = withCards.length >= 3;
    const ordered = withCards
      .map((t) => {
        const f = M.forecast(events.filter((e) => e.topic_id === t.id));
        return { t, r: f ? f.recallNow : 0 };
      })
      .sort((a, b) => a.r - b.r)
      .slice(0, count)
      .map((x) => x.t);

    return ordered.map((t) => {
      const card = cards[String(t.id)];
      if (!useChoice) return { topic: t, type: "cloze", prompt: clozeOf(card.answer), answer: card.answer };
      const others = shuffle([...new Set(withCards.filter((x) => x.id !== t.id).map((x) => cards[String(x.id)].answer))]).slice(0, 3);
      const options = shuffle([card.answer, ...others]);
      return {
        topic: t,
        type: "choice",
        prompt: card.question,
        answer: card.answer,
        options,
        correct: options.indexOf(card.answer),
      };
    });
  }

  function recordResult(topic, fullyCorrect) {
    const r = quiz.results.get(topic.id) || { topic, seen: 0, correct: 0 };
    r.seen += 1;
    if (fullyCorrect) r.correct += 1;
    quiz.results.set(topic.id, r);
  }

  function showStage() {
    $("quizSetup").hidden = true;
    $("quizResult").hidden = true;
    $("quizStage").hidden = false;
  }

  function renderQuestion() {
    const q = quiz.items[quiz.index];
    $("quizChoices").replaceChildren();
    $("quizReveal").hidden = true;
    $("quizGrade").hidden = true;
    $("quizAction").hidden = true;
    $("quizFeedback").textContent = "";
    $("quizMode").textContent = q.type === "choice" ? "Multiple choice" : "Fill in the blank";
    $("quizProgress").textContent = `Question ${quiz.index + 1} of ${quiz.items.length} · ${q.topic.topic_name}`;
    $("quizPrompt").textContent = q.prompt;

    if (q.type === "choice") {
      q.options.forEach((opt, i) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "btn choice";
        b.textContent = opt;
        b.addEventListener("click", () => chooseOption(i));
        $("quizChoices").append(b);
      });
    } else {
      $("quizAction").hidden = false;
      $("quizAction").textContent = "Reveal answer";
    }
  }

  function chooseOption(i) {
    const q = quiz.items[quiz.index];
    const buttons = Array.from($("quizChoices").querySelectorAll("button"));
    buttons.forEach((b) => { b.disabled = true; });
    buttons[q.correct].classList.add("is-right");
    const right = i === q.correct;
    if (!right) buttons[i].classList.add("is-wrong");
    if (right) quiz.score += 1;
    recordResult(q.topic, right);
    $("quizFeedback").textContent = right ? "Correct." : `Not quite. The card says: ${q.answer}`;
    $("quizAction").hidden = false;
    $("quizAction").textContent = quiz.index + 1 < quiz.items.length ? "Next" : "See result";
  }

  function gradeCloze(score) {
    const q = quiz.items[quiz.index];
    quiz.score += score;
    recordResult(q.topic, score === 1);
    advance();
  }

  function advance() {
    quiz.index += 1;
    if (quiz.index >= quiz.items.length) finishQuiz();
    else renderQuestion();
  }

  async function saveResults() {
    const rows = [...quiz.results.values()].filter((r) => r.seen > 0);
    $("quizRetry").hidden = true;
    $("quizSaveNote").textContent = "Saving to your account…";
    try {
      await Promise.all(rows.map((r) => S.api("/learning-events", {
        method: "POST",
        body: { student_id: me.id, topic_id: r.topic.id, questions_seen: r.seen, questions_correct: r.correct },
      })));
      $("quizSaveNote").textContent = `Saved ${rows.length} topic result${rows.length === 1 ? "" : "s"} to your account. Forecasts update from here.`;
      events = await S.api(`/learning-events?student_id=${me.id}`);
      quiz.saved = true;
    } catch (err) {
      $("quizSaveNote").textContent = `Could not save: ${err.message}`;
      $("quizRetry").hidden = false;
    }
  }

  function finishQuiz() {
    $("quizStage").hidden = true;
    $("quizResult").hidden = false;
    const n = quiz.items.length;
    const fully = [...quiz.results.values()].reduce((a, r) => a + r.correct, 0);
    const total = [...quiz.results.values()].reduce((a, r) => a + r.seen, 0);
    $("quizScore").textContent = quiz.items[0].type === "choice"
      ? `${quiz.score} / ${n}`
      : `${fully} / ${total} fully`;
    $("quizBreakdown").innerHTML = [...quiz.results.values()].map((r) => `<li class="row">
        <div><strong>${esc(r.topic.topic_name)}</strong><span class="muted">${r.correct} of ${r.seen} counted as correct</span></div>
      </li>`).join("");
    saveResults();
  }

  $("quizStart").addEventListener("click", () => {
    const count = Number($("quizCount").value);
    const items = buildQuiz(count);
    if (!items) {
      S.toast("Add a flashcard to a topic first.", "error");
      $("quizHint").hidden = false;
      return;
    }
    quiz = { items, index: 0, score: 0, results: new Map(), saved: false };
    showStage();
    renderQuestion();
  });

  $("quizAction").addEventListener("click", () => {
    const q = quiz.items[quiz.index];
    if (q.type === "cloze" && $("quizReveal").hidden) {
      $("quizAnswer").textContent = q.answer;
      $("quizReveal").hidden = false;
      $("quizGrade").hidden = false;
      $("quizAction").hidden = true;
      return;
    }
    advance();
  });

  $("quizGrade").querySelectorAll("[data-score]").forEach((b) =>
    b.addEventListener("click", () => gradeCloze(Number(b.dataset.score)))
  );

  $("quizRetry").addEventListener("click", saveResults);
  $("quizAgain").addEventListener("click", () => {
    quiz = null;
    $("quizResult").hidden = true;
    $("quizSetup").hidden = false;
    $("quizHint").hidden = topics.some((t) => L.cards(me.id)[String(t.id)]);
  });

  // ---------- Wiring ----------
  document.querySelectorAll(".mode-btn").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("tStart").addEventListener("click", startTimer);
  $("tPause").addEventListener("click", pauseTimer);
  $("tStop").addEventListener("click", () => {
    pauseTimer();
    saveStudy();
  });
  $("tReset").addEventListener("click", resetTimer);
  $("countdownMinutes").addEventListener("change", () => {
    if (!timer.running && timer.studiedMs === 0) resetTimer();
  });

  try {
    me = await S.currentStudent();
    [topics, events] = await Promise.all([
      S.api("/topics"),
      S.api(`/learning-events?student_id=${me.id}`),
    ]);
    $("studyTopic").innerHTML = topics.map((t) => `<option value="${t.id}">${esc(t.topic_name)}</option>`).join("");
    $("studyTopic").disabled = topics.length === 0;
    $("quizHint").hidden = topics.some((t) => L.cards(me.id)[String(t.id)]);
    resetTimer();
    renderLog();
    if (location.hash === "#quiz") $("quiz").scrollIntoView();
    window.SSR_FX.animateCounts();
  } catch (err) {
    S.showError("pageError", err);
  }
});
