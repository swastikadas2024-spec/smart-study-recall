"""Application adapter for the ML service's conditional next-session score."""

from datetime import datetime, timezone

import httpx

from app.config import settings


REVIEW_DELAY_CANDIDATES = [0.25, 1, 2, 3, 5, 7, 10, 14, 21, 30]
TARGET_NEXT_SESSION_SCORE = 0.80
MAX_TRAINED_GAP_DAYS = 30.0


def predict_review_scenarios(events, *, now: datetime | None = None) -> dict | None:
    """Predict scores for review delays and apply a separate threshold policy.

    `candidate_gap_days` sent to the model is the total time from the most
    recent session through the proposed review, including time already
    elapsed. The selected delay itself is not an input feature.
    """
    usable = sorted(
        (event for event in events if event.questions_seen > 0),
        key=lambda event: event.event_time,
    )
    if not usable:
        return None

    now = now or datetime.now(timezone.utc)
    latest = usable[-1].event_time
    latest = latest.replace(tzinfo=timezone.utc) if latest.tzinfo is None else latest
    elapsed = max(0.0, (now - latest).total_seconds() / 86400.0)
    if elapsed >= MAX_TRAINED_GAP_DAYS:
        return {
            "available": False,
            "reason": "The last session is outside the model's supported 30-day prediction interval.",
            "model": "next-session champion",
            "experimental": True,
        }

    history = []
    for index, event in enumerate(usable):
        item = {
            "questions_seen": int(event.questions_seen),
            "questions_correct": int(event.questions_correct),
        }
        if index < len(usable) - 1:
            following = usable[index + 1].event_time
            current = event.event_time
            following = following.replace(tzinfo=timezone.utc) if following.tzinfo is None else following
            current = current.replace(tzinfo=timezone.utc) if current.tzinfo is None else current
            item["gap_days"] = max(
                (following - current).total_seconds() / 86400.0,
                1.0 / 86400.0,
            )
        history.append(item)

    delays = [
        delay for delay in REVIEW_DELAY_CANDIDATES
        if elapsed + delay <= MAX_TRAINED_GAP_DAYS
    ]
    if not delays:
        return {
            "available": False,
            "reason": "No future review candidate fits the model's supported 30-day interval.",
            "model": "next-session champion",
            "experimental": True,
        }

    candidate_gaps = [elapsed + delay for delay in delays]
    try:
        response = httpx.post(
            f"{settings.ml_service_url.rstrip('/')}/predict/next-session/scenarios",
            json={"events": history, "candidate_gap_days": candidate_gaps},
            timeout=3.0,
        )
        response.raise_for_status()
        result = response.json()
    except (httpx.HTTPError, ValueError):
        return {
            "available": False,
            "reason": "The trained next-session model is unavailable. Use the conservative rule-based review suggestion.",
            "model": "next-session champion",
            "experimental": True,
        }

    predictions = result.get("predictions") or []
    if len(predictions) != len(delays):
        return {
            "available": False,
            "reason": "The model returned an incomplete set of review scenarios.",
            "model": "next-session champion",
            "experimental": True,
        }

    scenarios = []
    for delay, prediction in zip(delays, predictions):
        scenarios.append({
            "review_in_days": delay,
            "candidate_gap_days": prediction.get("candidate_gap_days"),
            "predicted_next_session_recall": prediction.get("predicted_next_session_recall"),
            "model": prediction.get("model"),
        })
    meeting_target = [
        scenario for scenario in scenarios
        if isinstance(scenario["predicted_next_session_recall"], (int, float))
        and scenario["predicted_next_session_recall"] >= TARGET_NEXT_SESSION_SCORE
    ]
    # Policy is deliberately separate from the predictor: select the latest
    # supported interval that meets the target; otherwise choose the earliest.
    selected = meeting_target[-1] if meeting_target else scenarios[0]
    return {
        "available": True,
        "model": predictions[0].get("model", "next-session champion"),
        "target_type": "next-session correct/seen fraction",
        "target_score": TARGET_NEXT_SESSION_SCORE,
        "elapsed_since_last_session_days": round(elapsed, 3),
        "recommended_review_in_days": selected["review_in_days"],
        "recommended_candidate_gap_days": selected["candidate_gap_days"],
        "recommended_predicted_score": selected["predicted_next_session_recall"],
        "policy": "latest candidate review delay with predicted next-session score at or above 80%; otherwise earliest candidate",
        "scenarios": scenarios,
        "experimental": True,
        "causal_effect_established": False,
    }
