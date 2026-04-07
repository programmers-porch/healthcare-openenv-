VALID_URGENCY = {"low", "medium", "high"}
VALID_DEPT = {"general", "dermatology", "cardiology", "emergency"}
VALID_ACTIONS = {"classify", "route", "request_info", "escalate", "finalize"}


def normalize(value, valid):
    if value is None:
        return None
    v = str(value).strip().lower()
    return v if v in valid else None


def grade_urgency(pred, gold):
    pred = normalize(pred, VALID_URGENCY)
    gold = normalize(gold, VALID_URGENCY)
    return 1.0 if pred == gold else 0.0


def grade_department(pred, gold):
    pred = normalize(pred, VALID_DEPT)
    gold = normalize(gold, VALID_DEPT)
    return 1.0 if pred == gold else 0.0


def grade_escalation(pred, gold):
    if pred is None:
        return 0.0
    return 1.0 if bool(pred) == bool(gold) else 0.0


def grade_request_info(action_type, expected):
    if expected and action_type == "request_info":
        return 1.0
    if not expected and action_type == "request_info":
        return 0.2
    return 0.0


def action_valid(action_type):
    return action_type in VALID_ACTIONS
