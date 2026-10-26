"""Application adapter for the ML service's conditional next-session score."""

from datetime import datetime, timedelta, timezone

import httpx

from app.config import settings


REVIEW_DELAY_CANDIDATES = [0.25, 1, 2, 3, 5, 7, 10, 14, 21, 30]
TARGET_NEXT_SESSION_SCORE = 0.80
MAX_TRAINED_GAP_DAYS = 30.0
FORGETTING_CURVE_GAPS = [0.01, 0.05, 0.1, 0.25, 0.5, *range(1, 31)]


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

    # Evaluate the learned recall curve across the full supported horizon,
    # independently of which future review dates the policy may choose.
    # Include policy candidates and today's elapsed point in the same request.
    curve_gaps = set(float(gap) for gap in FORGETTING_CURVE_GAPS)
    curve_gaps.add(max(elapsed, 0.01))
    policy_gaps = {round(elapsed + delay, 4): delay for delay in delays}
    candidate_gaps = sorted(curve_gaps | set(policy_gaps))
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
    by_gap = {
        round(float(prediction.get("candidate_gap_days")), 4): prediction
        for prediction in predictions
    }
    scenarios = []
    for requested_gap, delay in policy_gaps.items():
        prediction = by_gap.get(requested_gap)
        if prediction is None:
            continue
        scenarios.append({
            "review_in_days": delay,
            "candidate_gap_days": prediction.get("candidate_gap_days"),
            "predicted_next_session_recall": prediction.get("predicted_next_session_recall"),
            "model": prediction.get("model"),
        })
    if len(scenarios) != len(delays):
        return {
            "available": False,
            "reason": "The model returned an incomplete set of review scenarios.",
            "model": "next-session champion",
            "experimental": True,
        }

    curve = []
    for gap in sorted(curve_gaps):
        prediction = by_gap.get(round(gap, 4))
        if prediction is None:
            continue
        curve.append({
            "day_since_last_session": gap,
            "days_from_now": round(gap - elapsed, 4),
            "predicted_next_session_recall": prediction.get("predicted_next_session_recall"),
        })

    # Define the displayed "forgetting date" as the estimated first time the
    # next-session score falls below the app's explicit 80% review threshold.
    # This is a model-derived review deadline, not an observed memory event.
    crossing_gap = None
    crossing_status = "beyond_model_horizon"
    for point in curve:
        score = point["predicted_next_session_recall"]
        if isinstance(score, (int, float)) and score < TARGET_NEXT_SESSION_SCORE:
            crossing_gap = point["day_since_last_session"]
            crossing_status = "below_threshold_at_first_sample"
            break
    for left, right in zip(curve, curve[1:]):
        left_score = left["predicted_next_session_recall"]
        right_score = right["predicted_next_session_recall"]
        if (
            isinstance(left_score, (int, float))
            and isinstance(right_score, (int, float))
            and left_score >= TARGET_NEXT_SESSION_SCORE
            and right_score < TARGET_NEXT_SESSION_SCORE
        ):
            fraction = (left_score - TARGET_NEXT_SESSION_SCORE) / (left_score - right_score)
            crossing_gap = left["day_since_last_session"] + fraction * (
                right["day_since_last_session"] - left["day_since_last_session"]
            )
            crossing_status = (
                "threshold_already_crossed" if crossing_gap <= elapsed else "upcoming_threshold_crossing"
            )
            break

    predicted_forgetting_at = None
    days_until_forgetting = None
    overdue_days = None
    if crossing_gap is not None:
        predicted_forgetting_at = (latest + timedelta(days=crossing_gap)).isoformat()
        days_until_forgetting = max(0.0, crossing_gap - elapsed)
        overdue_days = max(0.0, elapsed - crossing_gap)
        if crossing_status == "below_threshold_at_first_sample" and crossing_gap <= elapsed:
            crossing_status = "threshold_already_crossed"

    current_gap = min(max(elapsed, 0.01), MAX_TRAINED_GAP_DAYS)
    current_prediction = by_gap.get(round(current_gap, 4))
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
        "predicted_forgetting_at": predicted_forgetting_at,
        "predicted_forgetting_gap_days": round(crossing_gap, 3) if crossing_gap is not None else None,
        "days_until_predicted_forgetting": round(days_until_forgetting, 3) if days_until_forgetting is not None else None,
        "days_past_predicted_forgetting": round(overdue_days, 3) if overdue_days is not None else None,
        "forgetting_forecast_status": crossing_status,
        "current_predicted_score": current_prediction.get("predicted_next_session_recall") if current_prediction else None,
        "forgetting_curve": curve,
        "recommended_review_in_days": selected["review_in_days"],
        "recommended_candidate_gap_days": selected["candidate_gap_days"],
        "recommended_predicted_score": selected["predicted_next_session_recall"],
        "policy": "latest candidate review delay with predicted next-session score at or above 80%; otherwise earliest candidate",
        "scenarios": scenarios,
        "experimental": True,
        "causal_effect_established": False,
    }
