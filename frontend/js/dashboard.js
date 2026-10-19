/* Dashboard: every total comes from stored records. Estimates are labelled. */
document.addEventListener("DOMContentLoaded", async () => {
  const S = window.SSR;
  if (!S.requireLogin()) return;
  const $ = (id) => document.getElementById(id);
  const { esc, pct, parseTime } = S;
  const M = window.SSR_MODEL;
  const L = window.SSR_LOCAL;

  // Each stat: label, numeric value for the count-up, display text, hint.
  function statCard({ label, count, decimals = 0, suffix = "", text, hint }) {
    const value = count !== undefined
      ? `<span class="stat-value" data-count="${count}" data-decimals="${decimals}" data-suffix="${esc(suffix)}">${esc(text)}</span>`
      : `<span class="stat-value">${esc(text)}</span>`;
    return `<div class="stat fx"><span class="stat-label">${esc(label)}</span>${value}<span class="stat-hint">${esc(hint)}</span></div>`;
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

    // Forgetting forecast per topic, soonest first.
    const rows = topics
      .map((t) => {
        const evs = events.filter((e) => e.topic_id === t.id);
        return { t, f: M.forecast(evs) };
      })
      .filter((x) => x.f)
      .sort((a, b) => a.f.dueIn - b.f.dueIn);

    const dueNow = rows.filter((x) => x.f.due).length;
    $("dueNow").textContent = String(dueNow);
    $("todayMinutes").textContent = minutesToday >= 60
      ? `${Math.floor(minutesToday / 60)} h ${minutesToday % 60} min`
      : `${minutesToday} min`;

    $("forecastList").innerHTML = rows.length
      ? rows.slice(0, 6).map(({ t, f }) => {
          const when = f.due ? "Due now" : `Forget around ${f.dueAt.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })}`;
          const note = f.reliable ? "" : ` · beyond ${M.RELIABLE_DAYS} days, treat as rough`;
          return `<li class="row fx">
            <div><strong>${esc(t.topic_name)}</strong><span class="muted">${esc(when)}${esc(note)}</span></div>
            <div class="row-actions"><span class="pill ${f.due ? "now" : f.recallNow < 0.8 ? "soon" : "later"}">${pct(f.recallNow * 100)} now</span></div>
          </li>`;
        }).join("")
      : `<li class="empty-line">Log questions for a topic to see its forecast.</li>`;

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
