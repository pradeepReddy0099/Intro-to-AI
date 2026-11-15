"""Natural-language interface: rule-based question routing.

Questions are matched against a fixed set of intents with keyword rules and
column-name lookup. No generative model is used: the router can only
select one of the pipeline's existing analyses, never invent a new one.
"""
from __future__ import annotations

SUPPORTED_INTENTS = ["summary", "compare", "correlation", "trend", "predict", "full"]


def parse_question(question: str | None, columns: list) -> dict:
    """Map a user question to an intent, an optional target, and a focus column."""
    if not question or not question.strip():
        return {
            "original": None,
            "intent": "full",
            "target_column": None,
            "focus_column": None,
            "note": "No question given; ran the full report.",
        }

    lowered = question.lower()
    focus = _find_named_column(lowered, columns)
    intent = None
    if any(word in lowered for word in ("predict", "forecast")):
        intent = "predict"
    elif any(word in lowered for word in ("trend", "over time", "growth", "change")):
        intent = "trend"
    elif any(
        word in lowered
        for word in ("correlat", "relationship", "relate", "drives", "drive", "affect", "influence")
    ):
        intent = "correlation"
    elif any(word in lowered for word in ("compare", "difference", "between", "group")):
        intent = "compare"
    elif any(word in lowered for word in ("summar", "overview", "describe", "what is")):
        intent = "summary"

    if intent is None:
        return {
            "original": question,
            "intent": "full",
            "target_column": None,
            "focus_column": focus,
            "note": (
                "Question did not match a supported intent "
                "(summary, compare, correlation, trend, predict); ran the full report."
            ),
        }

    return {
        "original": question,
        "intent": intent,
        "target_column": focus if intent == "predict" else None,
        "focus_column": focus,
        "note": "",
    }


def _find_named_column(lowered_question: str, columns: list) -> str | None:
    """Return the longest column name mentioned in the question, if any."""
    matches = [c for c in columns if str(c).lower() in lowered_question]
    if not matches:
        return None
    return max(matches, key=lambda c: len(str(c)))
