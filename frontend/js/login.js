/* Screen 1: welcome, login, sign-up, forgot password and password reset. */
document.addEventListener("DOMContentLoaded", () => {
  const S = window.SSR;
  const $ = (id) => document.getElementById(id);
  const val = (form, name) => form.querySelector(`[name="${name}"]`).value;

  if (!S.isConfigured()) $("configWarn").hidden = false;

  function say(text, kind = "error") {
    const el = $("authMsg");
    el.textContent = text || "";
    el.className = `auth-msg is-${kind}`;
    el.hidden = !text;
  }

  function showTab(name) {
    document.querySelectorAll("[data-panel]").forEach((p) => { p.hidden = p.dataset.panel !== name; });
    document.querySelectorAll("[data-tab]").forEach((b) => {
      const on = b.dataset.tab === name;
      b.classList.toggle("is-active", on);
      b.setAttribute("aria-selected", String(on));
    });
    say("");
  }

  async function runForm(form, work) {
    const button = form.querySelector("button[type=submit]");
    button.disabled = true;
    try {
      await work();
    } catch (err) {
      say(err.message || "Something went wrong.");
    } finally {
      button.disabled = false;
    }
  }

  document.querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

  const linkError = new URLSearchParams(location.hash.slice(1)).get("error_description");

  if (S.consumeRecoveryHash()) {
    showTab("reset");
  } else if (S.getSession()) {
    location.replace("dashboard.html");
    return;
  } else {
    showTab("login");
  }
  if (linkError) say(linkError.replace(/\+/g, " "));

  $("loginForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.target;
    runForm(f, async () => {
      await S.signIn(val(f, "email").trim(), val(f, "password"));
      location.replace("dashboard.html");
    });
  });

  $("signupForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.target;
    runForm(f, async () => {
      const password = val(f, "password");
      if (password.length < 8) throw new Error("Use at least 8 characters for your password.");
      const signedIn = await S.signUp(val(f, "email").trim(), password, val(f, "full_name").trim());
      if (signedIn) {
        location.replace("dashboard.html");
        return;
      }
      f.reset();
      showTab("login");
      say("Account created. Check your email for a confirmation link, then log in.", "ok");
    });
  });

  $("forgotForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.target;
    runForm(f, async () => {
      await S.requestReset(val(f, "email").trim());
      say("If an account exists for that email, a reset link is on its way.", "ok");
    });
  });

  $("resetForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.target;
    runForm(f, async () => {
      const password = val(f, "password");
      if (password.length < 8) throw new Error("Use at least 8 characters.");
      if (password !== val(f, "confirm")) throw new Error("The two passwords do not match.");
      await S.setNewPassword(password);
      location.replace("dashboard.html");
    });
  });
});
