/* Progress: measured results first, forgetting-curve estimates second, clearly labelled. */
document.addEventListener("DOMContentLoaded", async () => {
  const S = window.SSR;
  if (!S.requireLogin()) return;
  const $ = (id) => document.getElementById(id);
  const { esc, pct, parseTime } = S;
  const M = window.SSR_MODEL;
  let topics = [];
  let events = [];
  let revs = [];

  function topicEvents(topicId) {
    return events.filter((e) => e.topic_id === topicId && e.questions_seen > 0);
  }

  // Compares the first and second half of a topic's sessions by time.
  function trend(evs) {
    const sorted = [...evs].sort((a, b) => parseTime(a.event_time) - parseTime(b.event_time));
    if (sorted.length < 4) return "Not enough sessions";
    const half = Math.floor(sorted.length / 2);
    const acc = (list) => {
      const seen = list.reduce((a, e) => a + e.questions_seen, 0);
      const correct = list.reduce((a, e) => a + e.questions_correct, 0);
      return seen ? (correct / seen) * 100 : 0;
    };
    const diff = acc(sorted.slice(half)) - acc(sorted.slice(0, half));
    if (diff > 5) return `Improving (+${diff.toFixed(0)} pts)`;
    if (diff < -5) return `Declining (${diff.toFixed(0)} pts)`;
    return "Stable";
  }

  function renderMeasured() {
    const allSeen = events.reduce((a, e) => a + e.questions_seen, 0);
    const allCorrect = events.reduce((a, e) => a + e.questions_correct, 0);
    $("overallAccuracy").textContent = allSeen ? pct((allCorrect / allSeen) * 100) : "No data yet";

    $("measuredBody").innerHTML = topics.length
      ? topics.map((t) => {
          const evs = topicEvents(t.id);
          const seen = evs.reduce((a, e) => a + e.questions_seen, 0);
          const correct = evs.reduce((a, e) => a + e.questions_correct, 0);
          return `<tr>
            <td>${esc(t.topic_name)}<div class="muted">${esc(t.subject || "")}</div></td>
            <td>${evs.length}</td>
            <td>${seen}</td>
            <td>${pct(seen ? (correct / seen) * 100 : null)}</td>
            <td>${esc(trend(evs))}</td>
          </tr>`;
        }).join("")
      : `<tr><td colspan="5" class="muted">No topics yet.</td></tr>`;
  }

  // Forgetting curve for the chosen topic, drawn as SVG from the prototype model.
  function curveSVG(f, days) {
    const W = 900, H = 320, pad = { l: 56, r: 24, t: 20, b: 40 };
    const x = (t) => pad.l + (t / days) * (W - pad.l - pad.r);
    const y = (r) => pad.t + (1 - r) * (H - pad.t - pad.b);
    const pts = M.curve(f.H, f.elapsed, days, 90);
    const line = pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.r).toFixed(1)}`).join(" ");
    const area = `${line} L${x(days).toFixed(1)},${y(0)} L${x(0).toFixed(1)},${y(0)} Z`;

    const grid = [0, 0.2, 0.4, 0.6, 0.8, 1].map((r) =>
      `<line class="axis" x1="${pad.l}" x2="${W - pad.r}" y1="${y(r).toFixed(1)}" y2="${y(r).toFixed(1)}"/>
       <text class="label" x="${pad.l - 10}" y="${(y(r) + 4).toFixed(1)}" text-anchor="end">${Math.round(r * 100)}%</text>`).join("");

    const step = Math.max(1, Math.ceil(days / 6));
    const ticks = [];
    for (let d = 0; d <= days; d += step) {
      ticks.push(`<text class="label" x="${x(d).toFixed(1)}" y="${H - 14}" text-anchor="middle">${d === 0 ? "now" : `+${d}d`}</text>`);
    }

    const nowX = x(0), nowY = y(f.recallNow);
    return `<svg class="curve-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Predicted recall over ${days} days. Recall now ${Math.round(f.recallNow * 100)} percent.">
      ${grid}
      <line class="target" x1="${pad.l}" x2="${W - pad.r}" y1="${y(M.TARGET).toFixed(1)}" y2="${y(M.TARGET).toFixed(1)}"/>
      <path class="area" d="${area}"/>
      <path class="line curve-draw" d="${line}"/>
      ${ticks.join("")}
      <circle class="now" cx="${nowX.toFixed(1)}" cy="${nowY.toFixed(1)}" r="6"/>
    </svg>`;
  }

  function renderCurve() {
    const topicId = Number($("curveTopic").value);
    const days = Number($("curveDays").value);
    const t = topics.find((x) => x.id === topicId);
    const f = t ? M.forecast(topicEvents(t.id)) : null;
    if (!f) {
      $("curveOut").innerHTML = `<p class="muted">Log questions for this topic to draw its curve.</p>`;
      return;
    }
    const when = f.due
      ? "Due now"
      : `Forget around ${f.dueAt.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })}`;
    $("curveOut").innerHTML = `
      <div class="fx" style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px">
        <div class="kv" style="flex:1;min-width:160px"><dt>Recall now (est.)</dt><dd>${pct(f.recallNow * 100)}</dd></div>
        <div class="kv" style="flex:1;min-width:160px"><dt>80% point (est.)</dt><dd>${esc(when)}</dd></div>
        <div class="kv" style="flex:1;min-width:160px"><dt>Based on</dt><dd>${f.seen} questions, ${pct(f.accuracy * 100)} correct</dd></div>
      </div>
      ${curveSVG(f, days)}
      ${f.reliable ? "" : `<p class="muted" style="margin-top:10px">This forecast reaches beyond ${M.RELIABLE_DAYS} days, which the model's own notes say is unreliable.</p>`}`;
  }

  async function renderScenarios() {
    const topicId = Number($("curveTopic").value);
    if (!topicId) {
      $("curveOut").innerHTML = `<p class="muted">Add a topic and log a study session to compare review intervals.</p>`;
      return;
    }
    $("curveOut").innerHTML = `<p class="muted">Comparing candidate intervals…</p>`;
    try {
      $("curveOut").innerHTML = S.recallHTML(await S.api(`/topic-recall/${topicId}`));
    } catch (error) {
      $("curveOut").innerHTML = `<p class="muted">${esc(error.message)}</p>`;
    }
  }

  // Questions per week for the last 8 weeks (7-day windows ending today).
  function renderActivity() {
    const weeks = [];
    for (let w = 7; w >= 0; w--) {
      const end = new Date();
      end.setHours(23, 59, 59, 999);
      end.setDate(end.getDate() - w * 7);
      const start = new Date(end);
      start.setDate(start.getDate() - 6);
      start.setHours(0, 0, 0, 0);
      weeks.push({ start, end, value: 0 });
    }
    events.forEach((e) => {
      const t = parseTime(e.event_time);
      const week = weeks.find((w) => t >= w.start && t <= w.end);
      if (week) week.value += e.questions_seen;
    });
    const total = weeks.reduce((a, w) => a + w.value, 0);
    $("activityTotal").textContent = `${total} questions in 8 weeks`;
    S.renderBars($("activity"), weeks.map((w) => ({
      label: w.start.toLocaleDateString(undefined, { day: "numeric", month: "short" }),
      value: w.value,
    })));
  }

  function renderCompletion() {
    const done = revs.filter((r) => r.completed).length;
    const overdue = revs.filter((r) => !r.completed && parseTime(r.scheduled_at) < Date.now()).length;
    $("completionRate").textContent = revs.length ? pct((done / revs.length) * 100) : "—";
    $("completionDetail").textContent = `${done} of ${revs.length} revisions completed · ${overdue} overdue`;
  }

  function renderEstimates() {
    $("estimateList").innerHTML = topics.length
      ? topics.map((t) => `<li class="row">
          <div><strong>${esc(t.topic_name)}</strong><span class="muted">Not checked yet</span></div>
          <div class="row-actions">
            <button class="btn small" type="button" data-action="estimate" data-id="${t.id}">Check estimate</button>
          </div>
          <div class="estimate-out" hidden></div>
        </li>`).join("")
      : `<li class="empty-line">Add topics to see model estimates.</li>`;
  }

  $("estimateList").addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-action='estimate']");
    if (!btn) return;
    const out = btn.closest("li").querySelector(".estimate-out");
    btn.disabled = true;
    out.hidden = false;
    out.innerHTML = `<p class="muted">Checking…</p>`;
    try {
      out.innerHTML = S.recallHTML(await S.api(`/topic-recall/${btn.dataset.id}`));
    } catch (err) {
      out.innerHTML = `<p class="muted">${esc(err.message)}</p>`;
    } finally {
      btn.disabled = false;
    }
  });

  $("curveTopic").addEventListener("change", renderScenarios);

  try {
    const me = await S.currentStudent();
    [topics, events, revs] = await Promise.all([
      S.api("/topics"),
      S.api(`/learning-events?student_id=${me.id}`),
      S.api(`/revision-schedules?student_id=${me.id}`),
    ]);
    $("curveTopic").innerHTML = topics.map((t) => `<option value="${t.id}">${esc(t.topic_name)}</option>`).join("");
    $("curveTopic").disabled = topics.length === 0;
    renderMeasured();
    renderScenarios();
    renderActivity();
    renderCompletion();
    renderEstimates();
    window.SSR_FX.animateCounts();
  } catch (err) {
    S.showError("pageError", err);
  }
});
