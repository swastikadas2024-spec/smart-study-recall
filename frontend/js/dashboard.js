/* Dashboard: every total comes from stored records. Estimates are labelled. */
document.addEventListener("DOMContentLoaded", async () => {
  const S = window.SSR;
  if (!S.requireLogin()) return;
  const $ = (id) => document.getElementById(id);
  const { esc, pct, parseTime } = S;
  const L = window.SSR_LOCAL;

  // Each stat: label, numeric value for the count-up, display text, hint.
  function statCard({ label, count, decimals = 0, suffix = "", text, hint }) {
    const value = count !== undefined
      ? `<span class="stat-value" data-count="${count}" data-decimals="${decimals}" data-suffix="${esc(suffix)}">${esc(text)}</span>`
      : `<span class="stat-value">${esc(text)}</span>`;
    return `<div class="stat fx"><span class="stat-label">${esc(label)}</span>${value}<span class="stat-hint">${esc(hint)}</span></div>`;
  }

  function curveSVG(points, topicName, targetScore = 0.8, elapsedDays = 0) {
    if (!Array.isArray(points) || points.length < 2) return "";
    const width = 360, height = 92, left = 9, right = 7, top = 8, bottom = 18;
    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;
    const x = (day) => left + (Math.max(0, Math.min(30, Number(day) || 0)) / 30) * plotWidth;
    const y = (score) => top + (1 - Math.max(0, Math.min(1, Number(score) || 0)) ) * plotHeight;
    const path = points.map((point, index) => `${index ? "L" : "M"}${x(point.day_since_last_session).toFixed(1)},${y(point.predicted_next_session_recall).toFixed(1)}`).join(" ");
    const targetY = y(targetScore);
    const todayX = x(elapsedDays);
    return `<svg class="forgetting-curve" viewBox="0 0 ${width} ${height}" role="img" aria-label="Estimated recall decay curve for ${esc(topicName)}">
      <title>Estimated next-session score after the last study session</title>
      <line x1="${left}" y1="${targetY.toFixed(1)}" x2="${width - right}" y2="${targetY.toFixed(1)}" class="curve-target" />
      <line x1="${todayX.toFixed(1)}" y1="${top}" x2="${todayX.toFixed(1)}" y2="${height - bottom}" class="curve-today" />
      <path d="${path}" class="curve-path" />
      <text x="${left}" y="${height - 3}">0d</text><text x="${width - right}" y="${height - 3}" text-anchor="end">30d</text>
      <text x="${width - right - 2}" y="${Math.max(10, targetY - 3).toFixed(1)}" text-anchor="end">80% target</text>
    </svg>`;
  }

  function reminderLabel(prediction) {
    if (prediction.forgetting_forecast_status === "threshold_already_crossed") {
      return { text: "Review now", tone: "now", detail: prediction.predicted_forgetting_at
        ? `Estimated threshold date passed ${Math.ceil(prediction.days_past_predicted_forgetting || 0)} day(s) ago`
        : "The model estimates recall is already below the 80% target" };
    }
    if (prediction.forgetting_forecast_status === "below_threshold_at_first_sample") {
      return { text: "Review now", tone: "now", detail: "The first supported forecast point is already below the 80% target" };
    }
    if (prediction.predicted_forgetting_at) {
      const date = parseTime(prediction.predicted_forgetting_at);
      const when = date.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
      const remaining = Number(prediction.days_until_predicted_forgetting) || 0;
      const wait = remaining < 1 ? "within a day" : `in about ${Math.ceil(remaining)} days`;
      return { text: `Review by ${when}`, tone: remaining <= 2 ? "soon" : "later", detail: `Estimated to cross below 80% ${wait}` };
    }
    return { text: "Beyond 30-day horizon", tone: "later", detail: "The estimate stays at or above 80% through the model horizon" };
  }

  try {
    const me = await S.currentStudent();
    $("greeting").textContent = `Hello, ${me.student_name}.`;

    const [dash, topics, events, revs, recs] = await Promise.all([
      S.api(`/dashboard/${me.id}`),
      S.api("/topics"),
      S.api(`/learning-events?student_id=${me.id}`),
      S.api(`/revision-schedules?student_id=${me.id}`),
      S.api(`/recommendations/${me.id}`),
    ]);

    const names = Object.fromEntries(topics.map((t) => [t.id, t.topic_name]));
    const s = dash.summary;
    const rs = dash.revision_status;
    const minutesToday = L.minutesToday(me.id);

    $("statGrid").innerHTML = [
      statCard({ label: "Topics", count: s.topics_count, text: String(s.topics_count), hint: "Subjects you are tracking" }),
      statCard({ label: "Study sessions", count: s.learning_events_count, text: String(s.learning_events_count), hint: "Logged so far" }),
      statCard({ label: "Questions answered", count: s.questions_seen, text: String(s.questions_seen), hint: `${s.questions_correct} correct` }),
      statCard({ label: "Overall accuracy", count: s.overall_accuracy_percent ?? undefined, decimals: 1, suffix: "%", text: pct(s.overall_accuracy_percent), hint: "From recorded answers" }),
      statCard({ label: "Upcoming revisions", count: rs.upcoming, text: String(rs.upcoming), hint: `${rs.overdue} overdue` }),
      statCard({ label: "Completed revisions", count: rs.completed, text: String(rs.completed), hint: `of ${rs.total} scheduled` }),
    ].join("");

    // The backend evaluates the trained model at proposed review intervals;
    // review-date selection is a separate threshold policy.
    const rows = recs.recommendations
      .filter((r) => r.questions_seen > 0)
      .map((r) => ({
        t: topics.find((topic) => topic.id === r.topic_id),
        recommendation: r,
        prediction: r.next_session_prediction,
      }))
      .filter((row) => row.t)
      .sort((a, b) => {
        const aDays = a.prediction?.days_until_predicted_forgetting;
        const bDays = b.prediction?.days_until_predicted_forgetting;
        const aRank = a.prediction?.forgetting_forecast_status === "threshold_already_crossed" ? -1 : aDays ?? Infinity;
        const bRank = b.prediction?.forgetting_forecast_status === "threshold_already_crossed" ? -1 : bDays ?? Infinity;
        return aRank - bRank;
      });

    $("dueNow").textContent = String(rows.filter((row) => row.prediction?.available && (
      row.prediction.forgetting_forecast_status === "threshold_already_crossed" ||
      row.prediction.forgetting_forecast_status === "below_threshold_at_first_sample" ||
      (row.prediction.days_until_predicted_forgetting !== null && row.prediction.days_until_predicted_forgetting <= 2)
    )).length);
    $("todayMinutes").textContent = minutesToday >= 60
      ? `${Math.floor(minutesToday / 60)} h ${minutesToday % 60} min`
      : `${minutesToday} min`;

    $("forecastList").innerHTML = rows.length
      ? rows.map(({ t, recommendation, prediction }) => {
          if (!prediction?.available) {
            return `<li class="forgetting-row fx"><div><strong>${esc(t.topic_name)}</strong><span class="muted">${esc(prediction?.reason || "Next-session model unavailable.")}</span></div><span class="pill">Forecast unavailable</span></li>`;
          }
          const reminder = reminderLabel(prediction);
          const currentScore = prediction.current_predicted_score;
          return `<li class="row fx">
            <div class="forgetting-topic">
              <div class="forgetting-heading">
                <div><strong>${esc(t.topic_name)}</strong><span class="muted">${esc(recommendation.questions_seen)} questions studied · ${esc(prediction.model)} forecast</span></div>
                <span class="pill ${reminder.tone}">${esc(reminder.text)}</span>
              </div>
              ${curveSVG(prediction.forgetting_curve, t.topic_name, prediction.target_score, prediction.elapsed_since_last_session_days)}
              <div class="forgetting-footer"><span class="muted">${esc(reminder.detail)}</span><span class="muted">${currentScore === null || currentScore === undefined ? "Current estimate unavailable" : `${pct(currentScore * 100)} estimated now`}</span></div>
            </div>
          </li>`;
        }).join("")
      : `<li class="empty-line">Log a study session for a topic to see its forgetting curve and review-by reminder.</li>`;

    // Questions per day, last 7 days (local calendar days).
    const days = [];
    for (let i = 6; i >= 0; i--) {
      const d = new Date();
      d.setHours(0, 0, 0, 0);
      d.setDate(d.getDate() - i);
      days.push(d);
    }
    const totals = new Map(days.map((d) => [d.toDateString(), 0]));
    events.forEach((e) => {
      const key = parseTime(e.event_time).toDateString();
      if (totals.has(key)) totals.set(key, totals.get(key) + e.questions_seen);
    });
    const weekTotal = [...totals.values()].reduce((a, b) => a + b, 0);
    $("weekTotal").textContent = `${weekTotal} questions this week`;
    S.renderBars($("weekChart"), days.map((d) => ({
      label: d.toLocaleDateString(undefined, { weekday: "short" }),
      value: totals.get(d.toDateString()),
    })));

    // Topics needing attention (rule-based priority from the backend).
    const attention = recs.recommendations.filter((r) => r.priority !== "low").slice(0, 5);
    $("attention").innerHTML = attention.length
      ? attention.map((r) => `
        <li class="row fx">
          <div><strong>${esc(r.topic_name)}</strong><span class="muted">${esc(r.reason)}</span></div>
          <span class="pill ${r.priority === "high" ? "now" : "soon"}">${esc(r.priority)}</span>
        </li>`).join("")
      : `<li class="empty-line">Nothing needs attention right now.</li>`;

    // Upcoming revisions, earliest first.
    const now = Date.now();
    const upcoming = revs
      .filter((r) => !r.completed)
      .sort((a, b) => parseTime(a.scheduled_at) - parseTime(b.scheduled_at))
      .slice(0, 5);
    $("upcoming").innerHTML = upcoming.length
      ? upcoming.map((r) => {
          const overdue = parseTime(r.scheduled_at) < now;
          return `<li class="row fx">
            <div><strong>${esc(names[r.topic_id] || "Topic")}</strong><span class="muted">${esc(S.fmtDateTime(r.scheduled_at))}</span></div>
            <span class="pill ${overdue ? "now" : "soon"}">${overdue ? "Overdue" : "Pending"}</span>
          </li>`;
        }).join("")
      : `<li class="empty-line">No upcoming revisions. Schedule one in the Planner.</li>`;

    // Recent sessions.
    const recent = [...events]
      .sort((a, b) => parseTime(b.event_time) - parseTime(a.event_time))
      .slice(0, 8);
    $("recent").innerHTML = recent.length
      ? recent.map((e) => `<li class="row fx">
          <div><strong>${esc(names[e.topic_id] || "No topic")}</strong><span class="muted">${esc(S.fmtDateTime(e.event_time))}</span></div>
          <span class="muted">${esc(e.questions_correct)} of ${esc(e.questions_seen)} correct</span>
        </li>`).join("")
      : `<li class="empty-line">No sessions logged yet. Log one from the Topics page.</li>`;

    window.SSR_FX.animateCounts();
  } catch (err) {
    S.showError("pageError", err);
  }
});
