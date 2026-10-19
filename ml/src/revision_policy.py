TARGET_RETENTION = 0.80


def get_revision_priority(retention_probability):
    """
    Decide how urgently a concept should be revised.
    """

    if retention_probability < 0.60:
        return "Review now"

    elif retention_probability < TARGET_RETENTION:
        return "Review soon"

    else:
        return "Review later"


def get_revision_recommendation(retention_probability):
    """
    Return a user-friendly revision recommendation.
    """

    priority = get_revision_priority(retention_probability)

    if priority == "Review now":
        return {
            "priority": priority,
            "message": "Your memory of this concept may be fading. Review it now.",
        }

    elif priority == "Review soon":
        return {
            "priority": priority,
            "message": "You may forget this concept soon. Schedule a review.",
        }

    else:
        return {
            "priority": priority,
            "message": "Your retention is currently strong. You can review this later.",
        }