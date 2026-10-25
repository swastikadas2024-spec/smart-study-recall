/* Motion layer: staggered reveal on scroll, count-up numbers, and reveal for
   content rendered later. Everything respects prefers-reduced-motion. */
(() => {
  "use strict";

  document.documentElement.classList.add("js");
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const SELECTOR = ".panel, .stat, .topic, .row, .section-head, .empty, .fx";
  let observer = null;
  let sibling = 0;

  function show(el) {
    el.classList.add("in");
  }

  function track(el) {
    if (el.dataset.fx) return;
    el.dataset.fx = "1";
    el.classList.add("reveal");
    el.style.transitionDelay = `${(sibling++ % 8) * 60}ms`;
    if (reduce || !("IntersectionObserver" in window)) { show(el); return; }
    observer.observe(el);
  }

  function scan(root) {
    if (root.nodeType !== 1) return;
    if (root.matches && root.matches(SELECTOR)) track(root);
    root.querySelectorAll && root.querySelectorAll(SELECTOR).forEach(track);
  }

  function countUp(el, to, { decimals = 0, suffix = "" } = {}) {
    if (reduce || !Number.isFinite(to)) { el.textContent = to.toFixed(decimals) + suffix; return; }
    const start = performance.now();
    const dur = 900;
    const step = (now) => {
      const t = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = (to * eased).toFixed(decimals) + suffix;
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  // Elements with data-count="42" (and optional data-decimals, data-suffix) animate on render.
  function animateCounts(root = document) {
    root.querySelectorAll("[data-count]").forEach((el) => {
      countUp(el, Number(el.dataset.count), {
        decimals: Number(el.dataset.decimals || 0),
        suffix: el.dataset.suffix || "",
      });
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    const currentPage = location.pathname.split("/").pop() || "index.html";
    document.querySelectorAll(".app-nav a[href]").forEach((link) => {
      const destination = new URL(link.href, location.href).pathname.split("/").pop();
      if (destination === currentPage) link.setAttribute("aria-current", "page");
    });
    if (!reduce && "IntersectionObserver" in window) {
      observer = new IntersectionObserver((entries) => {
        entries.forEach((en) => {
          if (en.isIntersecting) { show(en.target); observer.unobserve(en.target); }
        });
      }, { threshold: 0.08 });
    }
    scan(document.body);
    // Reset the stagger counter after each burst of rendering so later rows still stagger.
    new MutationObserver((muts) => {
      sibling = 0;
      muts.forEach((m) => m.addedNodes.forEach(scan));
    }).observe(document.body, { childList: true, subtree: true });
  });

  window.SSR_FX = { animateCounts, countUp };
})();
