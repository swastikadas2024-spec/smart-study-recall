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

  function buildQuiz(count) {
    const cards = L.cards(me.id);
    const withCards = topics.filter((t) => cards[String(t.id)]);
    const answerBank = [...new Set(withCards.map((t) => String(cards[String(t.id)].answer || "").trim()).filter(Boolean))];
    if (withCards.length < 2 || answerBank.length < 2) return null;
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
      const answer = String(card.answer).trim();
      const others = shuffle(answerBank.filter((option) => option !== answer)).slice(0, 3);
      const options = shuffle([answer, ...others]);
      return {
        topic: t,
        type: "choice",
        prompt: card.question,
        answer,
        options,
        correct: options.indexOf(answer),
      };
    });
  }

  function refreshQuizAvailability() {
    const cards = L.cards(me.id);
    const available = topics.filter((t) => cards[String(t.id)] && String(cards[String(t.id)].answer || "").trim());
    const uniqueAnswers = new Set(available.map((t) => String(cards[String(t.id)].answer).trim()));
    const canBuild = available.length >= 2 && uniqueAnswers.size >= 2;
    Array.from($("quizCount").options).forEach((option) => {
      option.hidden = !canBuild || Number(option.value) > available.length;
      option.disabled = option.hidden;
    });
    if (canBuild) {
      const allowed = Array.from($("quizCount").options).filter((option) => !option.disabled);
      if (!allowed.some((option) => option.value === $("quizCount").value)) {
        $("quizCount").value = allowed[allowed.length - 1].value;
      }
      $("quizStart").disabled = false;
      $("quizHint").hidden = true;
      $("quizAvailability").textContent = `${available.length} flashcards ready · up to 4 answer choices per question · quiz results feed the next-session forecast.`;
    } else {
      $("quizStart").disabled = true;
      $("quizHint").hidden = false;
      $("quizAvailability").textContent = available.length
        ? "Add a flashcard with a different answer so the quiz can offer a real distractor."
        : "No flashcards yet. Add question-and-answer cards to at least two topics to start.";
    }
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
    $("quizAction").hidden = true;
    $("quizFeedback").textContent = "";
    $("quizMode").textContent = "Multiple choice";
    $("quizProgress").textContent = `Question ${quiz.index + 1} of ${quiz.items.length} · ${q.topic.topic_name}`;
    const progress = Math.round((quiz.index / quiz.items.length) * 100);
    $("quizProgressFill").style.width = `${progress}%`;
    $("quizStage").querySelector("[role=progressbar]").setAttribute("aria-valuenow", String(progress));
    $("quizPrompt").textContent = q.prompt;

    q.options.forEach((opt, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "btn choice";
      b.textContent = opt;
      b.addEventListener("click", () => chooseOption(i));
      $("quizChoices").append(b);
    });
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
    $("quizFeedback").textContent = right ? "Correct — answer saved." : `Not quite. Correct answer: ${q.answer}`;
    const completed = Math.round(((quiz.index + 1) / quiz.items.length) * 100);
    $("quizProgressFill").style.width = `${completed}%`;
    $("quizStage").querySelector("[role=progressbar]").setAttribute("aria-valuenow", String(completed));
    $("quizAction").hidden = false;
    $("quizAction").textContent = quiz.index + 1 < quiz.items.length ? "Next question" : "Finish quiz";
  }

  function advance() {
    quiz.index += 1;
    if (quiz.index >= quiz.items.length) finishQuiz();
    else renderQuestion();
  }

  async function saveResults() {
    const rows = [...quiz.results.values()].filter((r) => r.seen > 0 && !quiz.savedTopicIds.has(r.topic.id));
    $("quizRetry").hidden = true;
    $("quizAgain").disabled = true;
    $("quizSaveNote").textContent = "Saving to your account…";
    try {
      for (const r of rows) {
        await S.api("/learning-events", {
          method: "POST",
          body: { student_id: me.id, topic_id: r.topic.id, questions_seen: r.seen, questions_correct: r.correct },
        });
        quiz.savedTopicIds.add(r.topic.id);
      }
      events = await S.api(`/learning-events?student_id=${me.id}`);
      quiz.saved = true;
      const savedCount = quiz.savedTopicIds.size;
      $("quizSaveNote").textContent = `Saved ${savedCount} topic result${savedCount === 1 ? "" : "s"}. Your answers are now included in the next-session model input; the model weights are not retrained after each quiz.`;
      $("quizDashboardLink").hidden = false;
      await refreshQuizForecast();
      $("quizAgain").disabled = false;
    } catch (err) {
      $("quizSaveNote").textContent = `Could not save every result: ${err.message}`;
      $("quizRetry").hidden = false;
      $("quizAgain").disabled = false;
    }
  }

  async function refreshQuizForecast() {
    const impact = $("quizModelImpact");
    impact.hidden = false;
    impact.innerHTML = '<p class="muted">Refreshing your topic forecasts…</p>';
    try {
      const response = await S.api(`/recommendations/${me.id}`);
      const affected = response.recommendations.filter((item) => quiz.results.has(item.topic_id));
      impact.innerHTML = `<h3>Updated from this quiz</h3>${affected.map((item) => {
        const prediction = item.next_session_prediction;
        if (!prediction?.available) {
          return `<p><strong>${esc(item.topic_name)}:</strong> answers saved; forecast unavailable right now. ${esc(prediction?.reason || "")}</p>`;
        }
        const score = Number(prediction.current_predicted_score);
        const scoreText = Number.isFinite(score) ? `${Math.round(score * 100)}%` : "unavailable";
        return `<p><strong>${esc(item.topic_name)}:</strong> estimated next-session score ${scoreText} · suggested review in ${esc(prediction.recommended_review_in_days)} day(s).</p>`;
      }).join("")}`;
    } catch (err) {
      impact.innerHTML = `<p>Your quiz answers were saved as model inputs. The forecast could not be refreshed right now: ${esc(err.message)}</p>`;
    }
  }

  function finishQuiz() {
    $("quizStage").hidden = true;
    $("quizResult").hidden = false;
    const n = quiz.items.length;
    $("quizScore").textContent = `${quiz.score} / ${n}`;
    $("quizAgain").disabled = true;
    $("quizBreakdown").innerHTML = [...quiz.results.values()].map((r) => `<li class="row">
        <div><strong>${esc(r.topic.topic_name)}</strong><span class="muted">${r.correct} of ${r.seen} counted as correct</span></div>
      </li>`).join("");
    saveResults();
  }

  $("quizStart").addEventListener("click", () => {
    const count = Number($("quizCount").value);
    const items = buildQuiz(count);
    if (!items) {
      $("quizHint").hidden = false;
      return;
    }
    quiz = { items, index: 0, score: 0, results: new Map(), saved: false, savedTopicIds: new Set() };
    showStage();
    renderQuestion();
  });

  $("quizAction").addEventListener("click", () => {
    advance();
  });

  $("quizRetry").addEventListener("click", saveResults);
  $("quizAgain").addEventListener("click", () => {
    quiz = null;
    $("quizResult").hidden = true;
    $("quizSetup").hidden = false;
    $("quizModelImpact").hidden = true;
    $("quizDashboardLink").hidden = true;
    refreshQuizAvailability();
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
  $("manualStudyForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const topicId = Number($("manualStudyTopic").value);
    const topic = topics.find((item) => item.id === topicId);
    const hours = Number($("manualStudyHours").value);
    if (!topic || !Number.isFinite(hours) || hours <= 0 || hours > 24) {
      S.toast("Choose a topic and enter between 0.01 and 24 hours.", "error");
      return;
    }
    const minutes = Math.max(1, Math.round(hours * 60));
    try {
      L.addStudy(me.id, {
        topic_id: topic.id,
        topic_name: topic.topic_name,
        minutes,
        mode: "manual entry",
      });
      $("manualStudyHours").value = "";
      S.toast(`Logged ${hours} hour${hours === 1 ? "" : "s"} for ${topic.topic_name}.`);
      renderLog();
    } catch (err) {
      S.toast(err.message || "Could not save study time.", "error");
    }
  });

  try {
    me = await S.currentStudent();
    [topics, events] = await Promise.all([
      S.api("/topics"),
      S.api(`/learning-events?student_id=${me.id}`),
    ]);
    const topicOptions = topics.map((t) => `<option value="${t.id}">${esc(t.topic_name)}</option>`).join("");
    $("studyTopic").innerHTML = topicOptions;
    $("manualStudyTopic").innerHTML = topicOptions;
    $("studyTopic").disabled = topics.length === 0;
    $("manualStudyTopic").disabled = topics.length === 0;
    $("manualStudySave").disabled = topics.length === 0;
    refreshQuizAvailability();
    resetTimer();
    renderLog();
    if (location.hash === "#quiz") $("quiz").scrollIntoView();
    window.SSR_FX.animateCounts();
  } catch (err) {
    S.showError("pageError", err);
  }
});
