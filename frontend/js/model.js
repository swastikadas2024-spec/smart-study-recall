/* Forgetting forecast. The prototype half-life model from the repository
   (ml/models/hlr_parameters.json), computed here from your stored sessions.
   It is an estimate, not a measured memory result, and it is not calibrated. */
(() => {
  "use strict";

  const THETA = [0.8724197620504923, 5.188345250011773]; // [log(1 + questions seen), accuracy]
  const TARGET = 0.8;                                    // "due" when predicted recall reaches 80%
  const DAY_MS = 86400000;
  const RELIABLE_DAYS = 60;                              // the repository says longer forecasts are unreliable

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  // H = exp(θ1·log(1 + n) + θ2·accuracy), in days.
  function halfLifeDays(seen, correct) {
    const x1 = Math.log1p(seen);
    const x2 = seen > 0 ? correct / seen : 0;
    return Math.exp(clamp(THETA[0] * x1 + THETA[1] * x2, -20, 20));
  }

  // p = 2^(−t / H)
  const recallAfter = (days, H) => clamp(Math.pow(2, -days / H), 0, 1);

  // t* = H · log2(1 / 0.8)
  const daysToTarget = (H) => (-H * Math.log(TARGET)) / Math.LN2;

  const parseTime = (iso) => new Date(String(iso).replace(/(\.\d{3})\d+/, "$1"));

  // events: one topic's learning events [{questions_seen, questions_correct, event_time}]
  function forecast(events, now = Date.now()) {
    const evs = (events || []).filter((e) => e.questions_seen > 0);
    if (!evs.length) return null;
    const seen = evs.reduce((a, e) => a + e.questions_seen, 0);
    const correct = evs.reduce((a, e) => a + e.questions_correct, 0);
    const last = Math.max(...evs.map((e) => parseTime(e.event_time).getTime()));
    const H = halfLifeDays(seen, correct);
    const elapsed = Math.max(0, (now - last) / DAY_MS);
    const dueIn = daysToTarget(H) - elapsed;
    return {
      seen,
      correct,
      accuracy: correct / seen,
      H,
      elapsed,
      recallNow: recallAfter(elapsed, H),
      dueIn,
      dueAt: new Date(now + dueIn * DAY_MS),
      due: dueIn <= 0,
      reliable: dueIn <= RELIABLE_DAYS,
    };
  }

  // Predicted recall over the next `days` days, for the curve chart.
  function curve(H, elapsed, days = 30, steps = 60) {
    const pts = [];
    for (let i = 0; i <= steps; i++) {
      const t = (days * i) / steps;
      pts.push({ t, r: recallAfter(elapsed + t, H) });
    }
    return pts;
  }

  window.SSR_MODEL = { halfLifeDays, recallAfter, daysToTarget, forecast, curve, parseTime, RELIABLE_DAYS, TARGET, DAY_MS };
})();
