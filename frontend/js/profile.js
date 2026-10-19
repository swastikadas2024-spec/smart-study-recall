/* Profile: account details, password change, data export and clearing browser-only data. */
document.addEventListener("DOMContentLoaded", async () => {
  const S = window.SSR;
  if (!S.requireLogin()) return;
  const $ = (id) => document.getElementById(id);
  const L = window.SSR_LOCAL;
  let me = null;
  let snapshot = null;

  function renderLocal() {
    const cards = Object.keys(L.cards(me.id)).length;
    const minutes = L.studyLog(me.id).reduce((a, s) => a + (Number(s.minutes) || 0), 0);
    $("pCards").textContent = String(cards);
    $("pMinutes").textContent = String(minutes);
  }

  function download(name, text, type) {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  try {
    me = await S.currentStudent();
    const [topics, events, revs] = await Promise.all([
      S.api("/topics"),
      S.api(`/learning-events?student_id=${me.id}`),
      S.api(`/revision-schedules?student_id=${me.id}`),
    ]);
    snapshot = { profile: me, topics, learning_events: events, revision_schedules: revs };

    $("profileName").textContent = me.student_name;
    $("pName").textContent = me.student_name;
    $("pEmail").textContent = me.email || "—";
    $("pJoined").textContent = me.created_at ? S.fmtDateTime(me.created_at) : "—";
    $("pTotals").textContent = `${topics.length} · ${events.length}`;
    renderLocal();
    window.SSR_FX.animateCounts();
  } catch (err) {
    S.showError("pageError", err);
  }

  $("pwForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = e.target;
    const password = f.querySelector('[name="password"]').value;
    const confirm = f.querySelector('[name="confirm"]').value;
    if (password.length < 8) return S.toast("Use at least 8 characters.", "error");
    if (password !== confirm) return S.toast("The two passwords do not match.", "error");
    const btn = f.querySelector("button[type=submit]");
    btn.disabled = true;
    try {
      await S.setNewPassword(password);
      f.reset();
      S.toast("Password changed.");
    } catch (err) {
      S.toast(err.message, "error");
    } finally {
      btn.disabled = false;
    }
  });

  $("exportBtn").addEventListener("click", () => {
    if (!snapshot) return S.toast("Your data is still loading.", "error");
    const data = {
      exported_at: new Date().toISOString(),
      note: "Account data comes from the server. Flashcards and focus minutes come from this browser only.",
      ...snapshot,
      browser_only: {
        flashcards: L.cards(me.id),
        focus_minutes: L.studyLog(me.id),
      },
    };
    download("smart-study-recall-export.json", JSON.stringify(data, null, 2), "application/json");
    S.toast("Export downloaded.");
  });

  $("clearLocal").addEventListener("click", () => {
    if (!confirm("Remove the flashcards and focus minutes stored in this browser? Your account data is not affected.")) return;
    try {
      localStorage.removeItem("ssr.local.v1." + me.id);
      S.toast("Browser-only data cleared.");
      renderLocal();
    } catch (err) {
      S.toast("Could not clear browser storage.", "error");
    }
  });
});
