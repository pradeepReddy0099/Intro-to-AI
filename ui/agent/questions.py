"""Match a plain-language question to one fixed type of analysis."""
from __future__ import annotations



INTENT_RULES = [
    ("predict", ("predict", "forecast")),
    ("trend", ("trend", "over time", "growth", "change")),
    ("correlation", ("correlat", "relationship", "relate", "drives", "drive", "affect", "influence")),
    ("compare", ("compare", "difference", "between", "group")),
    ("summary", ("summar", "overview", "describe", "what is")),
]

def _result(original, intent, target_column, focus, note) -> dict:
    """One result shape for every outcome of parse_question."""
    return {
        "original": original,
        "intent": intent,
        "target_column": target_column,
        "focus_column": focus,
        "note": note,
    }


def parse_question(question: str | None, columns: list) -> dict:
    """Map a user question to an intent, an optional target, and a focus column."""
    if not question or not question.strip():
        return _result(
            None,
            "full",
            None,
            None,
            "No question given; ran the full report.",
        )

    lowered = question.lower()
    focus = _find_named_column(lowered, columns)
    intent = None
    for name, words in INTENT_RULES:
        if any(word in lowered for word in words):
            intent = name
            break

    if intent is None:
        return _result(
            question,
            "full",
            None,
            focus,
            "Question did not match a supported intent "
            "(summary, compare, correlation, trend, predict); ran the full report.",
        )

    return _result(
        question,
        intent,
        focus if intent == "predict" else None,
        focus,
        "",
    )


def _find_named_column(lowered_question: str, columns: list) -> str | None:
    """Return the longest column name mentioned in the question, if any."""
    matches = [c for c in columns if str(c).lower() in lowered_question]
    if not matches:
        return None
    return max(matches, key=lambda c: len(str(c)))
