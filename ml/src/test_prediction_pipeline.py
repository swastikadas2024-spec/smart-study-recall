
from prediction_pipeline import predict_session_recall

examples = [
    (10, 8, 1),
    (10, 8, 7),
    (50, 45, 30),
]

for history_seen, history_correct, elapsed_days in examples:
    result = predict_session_recall(
        history_seen=history_seen,
        history_correct=history_correct,
        elapsed_days=elapsed_days,
    )

    print(
        f"\nHistory: {history_correct}/{history_seen}, "
        f"elapsed: {elapsed_days} days"
    )
    print(result)
