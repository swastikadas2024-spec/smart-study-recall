/* Shared module: Supabase Auth over REST, calls to the FastAPI backend,
   and small UI helpers. Loaded by every page as window.SSR. */
(() => {
  "use strict";

  const cfg = window.APP_CONFIG || {};
  const KEY = "ssr.session";
  const nowSec = () => Math.floor(Date.now() / 1000);

  // ---------- Session storage ----------
  const getSession = () => {
    try { return JSON.parse(localStorage.getItem(KEY)); } catch (e) { return null; }
  };
  const setSession = (s) => localStorage.setItem(KEY, JSON.stringify(s));
  const clearSession = () => localStorage.removeItem(KEY);

  const isConfigured = () =>
    !!cfg.SUPABASE_URL && !cfg.SUPABASE_URL.includes("YOUR-") &&
    !!cfg.SUPABASE_ANON_KEY && !cfg.SUPABASE_ANON_KEY.includes("YOUR-");

  const sessionFromTokens = (d) => ({
    access_token: d.access_token,
    refresh_token: d.refresh_token,
    expires_at: nowSec() + Number(d.expires_in || 3600),
    user: d.user || null,
  });

  // ---------- Supabase Auth (REST) ----------
  async function authRequest(path, { method = "POST", body, token } = {}) {
    if (!isConfigured()) {
      throw new Error("Set SUPABASE_URL and SUPABASE_ANON_KEY in frontend/js/config.js first.");
    }
    let res;
    try {
      res = await fetch(`${cfg.SUPABASE_URL.replace(/\/$/, "")}/auth/v1${path}`, {
        method,
        headers: {
          apikey: cfg.SUPABASE_ANON_KEY,
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: body !== undefined ? JSON.stringify(body) : undefined,
      });
    } catch (e) {
      throw new Error("Cannot reach Supabase. Check your internet connection and SUPABASE_URL.");
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.msg || data.error_description || data.message || data.error || `Authentication failed (${res.status}).`);
    }
    return data;
  }

  // Returns true when the account is signed in immediately, false when
  // email confirmation is required first.
  async function signUp(email, password, fullName) {
    const d = await authRequest("/signup", { body: { email, password, data: { full_name: fullName } } });
    if (d.access_token) {
      setSession(sessionFromTokens(d));
      return true;
    }
    return false;
  }

  async function signIn(email, password) {
    const d = await authRequest("/token?grant_type=password", { body: { email, password } });
    setSession(sessionFromTokens(d));
  }

  async function requestReset(email) {
    const redirectTo = new URL("index.html", location.href).href;
    await authRequest(`/recover?redirect_to=${encodeURIComponent(redirectTo)}`, { body: { email } });
  }

  async function setNewPassword(password) {
    const token = await getToken();
    if (!token) throw new Error("Your reset link has expired. Request a new one.");
    await authRequest("/user", { method: "PUT", token, body: { password } });
  }

  // Password-reset emails return to index.html#access_token=...&type=recovery
  function consumeRecoveryHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    if (p.get("type") !== "recovery" || !p.get("access_token")) return false;
    setSession({
      access_token: p.get("access_token"),
      refresh_token: p.get("refresh_token"),
      expires_at: nowSec() + Number(p.get("expires_in") || 3600),
      user: null,
    });
    history.replaceState(null, "", location.pathname + location.search);
    return true;
  }

  async function refreshSession() {
    const s = getSession();
    if (!s || !s.refresh_token) return null;
    try {
      const d = await authRequest("/token?grant_type=refresh_token", { body: { refresh_token: s.refresh_token } });
      setSession(sessionFromTokens(d));
      return getSession();
    } catch (e) {
      clearSession();
      return null;
    }
  }

  async function getToken() {
    const s = getSession();
    if (!s) return null;
    if (s.expires_at - 60 > nowSec()) return s.access_token;
    const refreshed = await refreshSession();
    return refreshed ? refreshed.access_token : null;
  }

  async function signOut() {
    const s = getSession();
    if (s && s.access_token) {
      try { await authRequest("/logout", { token: s.access_token }); } catch (e) { /* ignore */ }
    }
    clearSession();
    location.replace("index.html");
  }

  function requireLogin() {
    if (!getSession()) {
      location.replace("index.html");
      return false;
    }
    return true;
  }

  // ---------- Backend (FastAPI) ----------
  class ApiError extends Error {
    constructor(status, message) {
      super(message);
      this.status = status;
    }
  }

  function errorText(data, status) {
    const d = data && data.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      return d.map((x) => `${(x.loc || []).slice(1).join(".") || "input"}: ${x.msg}`).join("; ");
    }
    return `Request failed (${status}).`;
  }

  async function api(path, { method = "GET", body } = {}) {
    const token = await getToken();
    if (!token) {
      clearSession();
      location.replace("index.html");
      throw new ApiError(401, "Please log in again.");
    }

    let res;
    try {
      res = await fetch(cfg.API_URL + path, {
        method,
        headers: {
          Authorization: `Bearer ${token}`,
          ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
        },
        body: body !== undefined ? JSON.stringify(body) : undefined,
      });
    } catch (e) {
      throw new ApiError(0, `Cannot reach the backend at ${cfg.API_URL}. Is it running on port 8001?`);
    }

    if (res.status === 401) {
      clearSession();
      location.replace("index.html");
      throw new ApiError(401, "Your session expired. Please log in again.");
    }
    if (res.status === 204) return null;

    const data = await res.json().catch(() => null);
    if (!res.ok) throw new ApiError(res.status, errorText(data, res.status));
    return data;
  }

  let meCache = null;
  async function currentStudent() {
    if (!meCache) meCache = await api("/students/me");
    return meCache;
  }

  // ---------- Formatting and UI helpers ----------
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));

  const pct = (v) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "—" : `${Number(v).toFixed(1)}%`);

  // Postgres timestamps can have 6 fractional digits, which some browsers reject.
  const parseTime = (iso) => new Date(String(iso).replace(/(\.\d{3})\d+/, "$1"));

  const fmtDateTime = (iso) => {
    const d = parseTime(iso);
    return Number.isNaN(d.getTime())
      ? "—"
      : d.toLocaleString(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
  };

  const toLocalInput = (d) => {
    const p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
  };

  const fromLocalInput = (v) => {
    const d = new Date(v);
    return Number.isNaN(d.getTime()) ? null : d.toISOString();
  };

  let toastTimer;
  function toast(message, kind = "ok") {
    const el = document.getElementById("toast");
    if (!el) { alert(message); return; }
    el.textContent = message;
    el.className = `toast${kind === "error" ? " is-error" : ""}`;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, 4000);
  }

  function showError(id, err) {
    const el = document.getElementById(id);
    if (el) {
      el.textContent = err && err.message ? err.message : String(err);
      el.hidden = false;
    } else {
      toast(err && err.message ? err.message : String(err), "error");
    }
  }

  // items: [{label, value}] drawn as a fixed-height bar chart.
  function renderBars(el, items, unit = "") {
    const max = Math.max(1, ...items.map((i) => i.value));
    el.innerHTML = items.map((i, idx) => {
      const h = i.value > 0 ? Math.max(2, Math.round((i.value / max) * 170)) : 2;
      return `<div class="bar-col">
        <span class="bar-val">${esc(i.value)}${esc(unit)}</span>
        <div class="bar-track"><div class="bar" style="height:${h}px;--d:${idx * 70}ms"></div></div>
        <span class="bar-day">${esc(i.label)}</span>
      </div>`;
    }).join("");
  }

  // Result of GET /topic-recall/{id}. Always labelled as experimental.
  function recallHTML(r) {
    if (!r.available) return `<p class="muted">${esc(r.reason)}</p>`;
    const rate = r.prediction && r.prediction.predicted_session_recall_rate;
    return `<div class="estimate">
      <span class="stat-label">Model estimate · experimental</span>
      <p class="estimate-value">${rate === undefined || rate === null ? "n/a" : pct(rate * 100)}</p>
      <p class="muted">Based on your recorded accuracy of ${pct(r.accuracy_percent)} over ${esc(r.questions_seen)} questions, ${esc(r.elapsed_days)} days after your latest session. RF-1 model, not calibrated. This is not a measured recall result.</p>
    </div>`;
  }

  function setupShell() {
    document.querySelectorAll("[data-signout]").forEach((b) => b.addEventListener("click", signOut));
    const here = location.pathname.split("/").pop() || "index.html";
    document.querySelectorAll(".app-nav a").forEach((a) => {
      if (a.getAttribute("href") === here) a.setAttribute("aria-current", "page");
    });
  }
  document.addEventListener("DOMContentLoaded", setupShell);

  window.SSR = {
    isConfigured, getSession, setSession, clearSession,
    signUp, signIn, requestReset, setNewPassword, consumeRecoveryHash, signOut,
    requireLogin, currentStudent, api, ApiError,
    esc, pct, parseTime, fmtDateTime, toLocalInput, fromLocalInput,
    toast, showError, renderBars, recallHTML,
  };
})();
