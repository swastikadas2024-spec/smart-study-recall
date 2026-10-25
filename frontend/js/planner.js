/* Planner: suggestions, scheduling, completion, rescheduling, a month calendar and .ics export. */
document.addEventListener("DOMContentLoaded", async () => {
  const S = window.SSR;
  if (!S.requireLogin()) return;
  const $ = (id) => document.getElementById(id);
  const { esc, parseTime } = S;
  const val = (form, name) => form.querySelector(`[name="${name}"]`).value;
  let me = null;
  let topics = [];
  let revs = [];
  let recs = [];
  let viewMonth = new Date();
  viewMonth.setDate(1);
  viewMonth.setHours(0, 0, 0, 0);
  let selectedDay = null;

  const byTime = (a, b) => parseTime(a.scheduled_at) - parseTime(b.scheduled_at);
  const priorityClass = (p) => (p === "high" ? "now" : p === "medium" ? "soon" : "later");
  const sameDay = (a, b) => a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();

  function statusOf(r) {
    const past = !r.completed && parseTime(r.scheduled_at) < Date.now();
    if (r.completed) return "done";
    if (past) return "now";
    return "soon";
  }

  function revisionItem(r, names) {
    const status = r.completed ? "Completed" : statusOf(r) === "now" ? "Overdue" : r.rescheduled_count > 0 ? "Rescheduled" : "Pending";
    const cls = r.completed ? "later" : statusOf(r) === "now" ? "now" : "soon";
    const rescheduleNote = r.rescheduled_count > 0 ? ` · rescheduled ${r.rescheduled_count}×` : "";
    const actions = r.completed ? "" : `
      <button class="btn small primary" type="button" data-action="complete" data-id="${r.id}">Mark complete</button>
      <button class="btn small ghost" type="button" data-action="toggle" data-id="${r.id}">Reschedule</button>`;
    const form = r.completed ? "" : `
      <form class="reschedule" data-id="${r.id}" hidden>
        <input type="datetime-local" name="scheduled_at" required aria-label="New date and time" />
        <button class="btn small" type="submit">Save new time</button>
      </form>`;
    return `<li class="row">
      <div><strong>${esc(names[r.topic_id] || "Topic")}</strong><span class="muted">${esc(S.fmtDateTime(r.scheduled_at))}${esc(rescheduleNote)}</span></div>
      <div class="row-actions"><span class="pill ${cls}">${status}</span>${actions}</div>
      ${form}
    </li>`;
  }

  // ---------- Calendar ----------
  function renderCalendar(names) {
    const monthName = viewMonth.toLocaleDateString(undefined, { month: "long", year: "numeric" });
    $("calTitle").textContent = monthName;

    const first = new Date(viewMonth);
    const offset = (first.getDay() + 6) % 7; // Monday first
    const start = new Date(first);
    start.setDate(first.getDate() - offset);

    const dows = [];
    for (let i = 0; i < 7; i++) {
      const d = new Date(2024, 0, 1 + i); // a Monday
      dows.push(`<div class="cal-dow" role="columnheader">${d.toLocaleDateString(undefined, { weekday: "short" })}</div>`);
    }

    const today = new Date();
    const cells = [];
    for (let i = 0; i < 42; i++) {
      const day = new Date(start);
      day.setDate(start.getDate() + i);
      const items = revs.filter((r) => sameDay(parseTime(r.scheduled_at), day)).sort(byTime);
      const dots = items.slice(0, 4).map((r) => `<span class="cal-dot ${statusOf(r)}" title="${esc(names[r.topic_id] || "Topic")}"></span>`).join("");
      const extra = items.length > 4 ? `<span class="muted">+${items.length - 4}</span>` : "";
      const classes = ["cal-day"];
      if (day.getMonth() !== viewMonth.getMonth()) classes.push("is-out");
      if (sameDay(day, today)) classes.push("is-today");
      if (selectedDay && sameDay(day, selectedDay)) classes.push("is-selected");
      cells.push(`<button class="${classes.join(" ")}" type="button" data-day="${day.getTime()}" aria-label="${esc(day.toDateString())}, ${items.length} revision${items.length === 1 ? "" : "s"}">
        <span class="cal-num">${day.getDate()}</span>
        <span class="cal-dots">${dots}${extra}</span>
      </button>`);
    }
    $("calGrid").innerHTML = dows.join("") + cells.join("");
  }

  function renderAgenda(names) {
    if (!selectedDay) {
      $("calAgendaTitle").textContent = "Select a day";
      $("calAgenda").innerHTML = `<li class="empty-line">Pick a day on the calendar to see its revisions.</li>`;
      return;
    }
    $("calAgendaTitle").textContent = selectedDay.toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" });
    const items = revs.filter((r) => sameDay(parseTime(r.scheduled_at), selectedDay)).sort(byTime);
    $("calAgenda").innerHTML = items.length
      ? items.map((r) => revisionItem(r, names)).join("")
      : `<li class="empty-line">No revisions on this day.</li>`;
  }

  function icsStamp(d) {
    return d.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
  }

  function exportIcs(names) {
    const pending = revs.filter((r) => !r.completed).sort(byTime);
    if (!pending.length) {
      S.toast("No pending revisions to export.", "error");
      return;
    }
    const now = new Date();
    const lines = [
      "BEGIN:VCALENDAR",
      "VERSION:2.0",
      "PRODID:-//Smart Study Recall//Revision planner//EN",
      "CALSCALE:GREGORIAN",
      "METHOD:PUBLISH",
    ];
    pending.forEach((r) => {
      const start = parseTime(r.scheduled_at);
      const end = new Date(start.getTime() + 30 * 60000);
      const title = `Review: ${names[r.topic_id] || "Topic"}`.replace(/[,;\\]/g, " ");
      lines.push(
        "BEGIN:VEVENT",
        `UID:revision-${r.id}@smart-study-recall`,
        `DTSTAMP:${icsStamp(now)}`,
        `DTSTART:${icsStamp(start)}`,
        `DTEND:${icsStamp(end)}`,
        `SUMMARY:${title}`,
        "DESCRIPTION:Scheduled by Smart Study Recall. Forecasts are estimates.",
        "END:VEVENT"
      );
    });
    lines.push("END:VCALENDAR");
    const blob = new Blob([lines.join("\r\n")], { type: "text/calendar" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "smart-study-revisions.ics";
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    S.toast(`Exported ${pending.length} revision${pending.length === 1 ? "" : "s"}.`);
  }

  function render() {
    const names = Object.fromEntries(topics.map((t) => [t.id, t.topic_name]));
    const now = Date.now();

    $("revTopic").innerHTML = topics.map((t) => `<option value="${t.id}">${esc(t.topic_name)}</option>`).join("");
    $("revTopic").disabled = topics.length === 0;
    $("revisionForm").querySelector("button[type=submit]").disabled = topics.length === 0;
    if (!$("revAt").value) {
      const tomorrow = new Date(Date.now() + 86400000);
      tomorrow.setHours(9, 0, 0, 0);
      $("revAt").value = S.toLocalInput(tomorrow);
    }

    $("suggestions").innerHTML = recs.recommendations.length
      ? recs.recommendations.map((r) => `<li class="row">
          <div><strong>${esc(r.topic_name)}</strong><span class="muted">${esc(r.reason)} ${esc(r.suggested_action)}</span>
            ${r.next_session_prediction?.available
              ? `<span class="muted">Model policy: review in ${esc(r.next_session_prediction.recommended_review_in_days)} days · expected next-session score ${S.pct(r.next_session_prediction.recommended_predicted_score * 100)}</span>`
              : `<span class="muted">${esc(r.next_session_prediction?.reason || "Rule-based schedule will be used if the model is unavailable.")}</span>`}</div>
          <div class="row-actions">
            <span class="pill ${priorityClass(r.priority)}">${esc(r.priority)}</span>
            <button class="btn small primary" type="button" data-action="schedule" data-topic="${r.topic_id}">Schedule review</button>
          </div>
        </li>`).join("")
      : `<li class="empty-line">Add topics and log sessions to get suggestions.</li>`;

    const pending = revs.filter((r) => !r.completed).sort(byTime);
    const overdue = pending.filter((r) => parseTime(r.scheduled_at) < now);
    const upcoming = pending.filter((r) => parseTime(r.scheduled_at) >= now);
    const done = revs.filter((r) => r.completed).sort((a, b) => parseTime(b.scheduled_at) - parseTime(a.scheduled_at));

    $("overdueCount").textContent = String(overdue.length);
    $("overdueList").innerHTML = overdue.length
      ? overdue.map((r) => revisionItem(r, names)).join("")
      : `<li class="empty-line">Nothing overdue.</li>`;

    $("upcomingCount").textContent = String(upcoming.length);
    $("upcomingList").innerHTML = upcoming.length
      ? upcoming.map((r) => revisionItem(r, names)).join("")
      : `<li class="empty-line">No upcoming revisions.</li>`;

    $("completedCount").textContent = String(done.length);
    $("completedList").innerHTML = done.length
      ? done.map((r) => revisionItem(r, names)).join("")
      : `<li class="empty-line">No completed revisions yet.</li>`;

    $("completionRate").textContent = revs.length ? S.pct((done.length / revs.length) * 100) : "—";
    $("completionDetail").textContent = `${done.length} of ${revs.length} revisions completed`;

    renderCalendar(names);
    renderAgenda(names);
    window.SSR_FX.animateCounts();
  }

  async function load() {
    me = await S.currentStudent();
    [topics, revs, recs] = await Promise.all([
      S.api("/topics"),
      S.api(`/revision-schedules?student_id=${me.id}`),
      S.api(`/recommendations/${me.id}`),
    ]);
    render();
  }

  async function run(work) {
    try {
      await work();
    } catch (err) {
      S.toast(err.message, "error");
    }
  }

  const planner = $("planner");

  // Calendar controls.
  $("calPrev").addEventListener("click", () => { viewMonth = new Date(viewMonth.getFullYear(), viewMonth.getMonth() - 1, 1); render(); });
  $("calNext").addEventListener("click", () => { viewMonth = new Date(viewMonth.getFullYear(), viewMonth.getMonth() + 1, 1); render(); });
  $("calToday").addEventListener("click", () => {
    const t = new Date();
    viewMonth = new Date(t.getFullYear(), t.getMonth(), 1);
    selectedDay = new Date(t.getFullYear(), t.getMonth(), t.getDate());
    render();
  });
  $("calGrid").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-day]");
    if (!btn) return;
    const d = new Date(Number(btn.dataset.day));
    selectedDay = d;
    if (d.getMonth() !== viewMonth.getMonth()) viewMonth = new Date(d.getFullYear(), d.getMonth(), 1);
    render();
  });
  $("exportIcs").addEventListener("click", () => exportIcs(Object.fromEntries(topics.map((t) => [t.id, t.topic_name]))));

  planner.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;
    const action = btn.dataset.action;

    if (action === "schedule") {
      run(async () => {
        await S.api(`/recommendations/${me.id}/schedule`, {
          method: "POST",
          body: { topic_id: Number(btn.dataset.topic) },
        });
        S.toast("Review scheduled.");
        await load();
      });
    } else if (action === "complete") {
      run(async () => {
        await S.api(`/revision-schedules/${btn.dataset.id}/complete`, { method: "PATCH" });
        S.toast("Marked complete.");
        await load();
      });
    } else if (action === "toggle") {
      const form = btn.closest("li").querySelector("form.reschedule");
      if (form) form.hidden = !form.hidden;
    }
  });

  planner.addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.target;

    if (f.id === "revisionForm") {
      const topicId = Number(val(f, "topic_id"));
      const iso = S.fromLocalInput(val(f, "scheduled_at"));
      if (!topicId || !iso) return S.toast("Choose a topic and a valid date and time.", "error");
      run(async () => {
        await S.api("/revision-schedules", {
          method: "POST",
          body: { student_id: me.id, topic_id: topicId, scheduled_at: iso },
        });
        S.toast("Revision scheduled.");
        await load();
      });
    } else if (f.classList.contains("reschedule")) {
      const iso = S.fromLocalInput(val(f, "scheduled_at"));
      if (!iso) return S.toast("Choose a valid new date and time.", "error");
      run(async () => {
        await S.api(`/revision-schedules/${f.dataset.id}/reschedule`, {
          method: "PATCH",
          body: { scheduled_at: iso },
        });
        S.toast("Revision rescheduled.");
        await load();
      });
    }
  });

  try {
    await load();
  } catch (err) {
    S.showError("pageError", err);
  }
});
