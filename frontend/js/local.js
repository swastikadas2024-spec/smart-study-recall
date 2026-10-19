/* Browser-only data: flashcard text and focus-timer minutes. The backend has no
   fields for these, so they stay in this browser, separate for each student. */
(() => {
  "use strict";

  const PREFIX = "ssr.local.v1.";
  const empty = () => ({ cards: {}, study: [] });

  function read(studentId) {
    try {
      const raw = localStorage.getItem(PREFIX + studentId);
      return raw ? { ...empty(), ...JSON.parse(raw) } : empty();
    } catch (e) {
      return empty();
    }
  }

  function write(studentId, data) {
    try {
      localStorage.setItem(PREFIX + studentId, JSON.stringify(data));
    } catch (e) {
      throw new Error("This browser is not saving data right now. Check that site storage is allowed.");
    }
  }

  const cards = (sid) => read(sid).cards;

  function setCard(sid, topicId, question, answer) {
    const d = read(sid);
    d.cards[String(topicId)] = { question: question.trim(), answer: answer.trim(), updated: new Date().toISOString() };
    write(sid, d);
  }

  function deleteCard(sid, topicId) {
    const d = read(sid);
    delete d.cards[String(topicId)];
    write(sid, d);
  }

  function pruneCards(sid, validTopicIds) {
    const d = read(sid);
    const keep = new Set(validTopicIds.map(String));
    let changed = false;
    Object.keys(d.cards).forEach((k) => { if (!keep.has(k)) { delete d.cards[k]; changed = true; } });
    if (changed) write(sid, d);
  }

  const studyLog = (sid) => read(sid).study;

  function addStudy(sid, entry) {
    const d = read(sid);
    d.study.push({ ...entry, date: new Date().toISOString() });
    write(sid, d);
  }

  // Minutes studied since a timestamp, optionally for one topic.
  function minutesSince(sid, sinceMs, topicId = null) {
    return read(sid).study
      .filter((s) => new Date(s.date).getTime() >= sinceMs)
      .filter((s) => topicId === null || String(s.topic_id) === String(topicId))
      .reduce((a, s) => a + (Number(s.minutes) || 0), 0);
  }

  const startOfToday = () => { const d = new Date(); d.setHours(0, 0, 0, 0); return d.getTime(); };
  const minutesToday = (sid) => minutesSince(sid, startOfToday());

  window.SSR_LOCAL = { cards, setCard, deleteCard, pruneCards, studyLog, addStudy, minutesSince, minutesToday };
})();
