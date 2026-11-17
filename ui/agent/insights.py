"""Plain-language insights and decision guidance, built only from a run's results.

Two lists come out of build_insights():
- insights: what the data says — past performance (trends, best and worst
  groups), the strongest relationships, and what a prediction model found.
- decisions: what to do next — each one tied to a specific insight above.

Every number in the text is copied from the computed results; the wording is
fixed templates, so the same input always produces the same insights.
"""
from __future__ import annotations


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _strip_feature_prefix(raw: str) -> str:
    """Drop the ColumnTransformer prefix (num__ / cat__) from a feature name."""
    for prefix in ("num__", "cat__"):
        if raw.startswith(prefix):
            return raw[len(prefix):]
    return raw


def _feature_name(raw: str) -> str:
    """Turn a model feature name like cat__region_South into 'region South'."""
    return _strip_feature_prefix(raw).replace("_", " ")


def filled_missing_count(clean_log: list) -> int:
    """Rows the cleaning step filled, counted from the cleaning log."""
    return sum(
        entry.get("rows_affected", 0)
        for entry in clean_log
        if entry["step"].startswith("fill_missing")
    )


def build_insights(results: dict) -> dict:
    """Return {'insights': [...], 'decisions': [...]} for a completed run."""
    if results.get("status") != "complete":
        return {"insights": [], "decisions": []}
    analysis = results["analysis"]
    insights: list[str] = []
    decisions: list[str] = []

    # --- Past performance: how the measures moved over time ---
    # Wording stays neutral about good/bad: the program cannot know whether
    # a rise in a column (say, churn or cost) is what the user wants.
    changes = analysis["trends"]["changes"]
    ranked_changes = sorted(
        changes, key=lambda item: abs(item["percent_change"] or 0), reverse=True
    )
    if ranked_changes:
        main = ranked_changes[0]
        pct = main["percent_change"]
        direction = "grew" if pct > 0 else "fell"
        ending = "higher" if pct > 0 else "lower"
        insights.append(
            f"Past performance: {main['column']} {direction} {abs(pct):.1f}% "
            f"between {main['first_period']} and {main['last_period']} "
            f"(from {_fmt(main['first_value'])} to {_fmt(main['last_value'])}) — "
            f"it ended the period {ending} than it began."
        )
        decisions.append(
            f"Biggest change over time: {main['column']} "
            f"({main['percent_change']:+.1f}%). Check the later part of the "
            f"period first — that is where the change is — and what drove it, "
            "before assuming the new level will continue."
        )

    # --- Past performance: which groups did best and worst ---
    summaries = analysis["group_summaries"]

    def relative_gap(item) -> float:
        top = abs(item["highest_mean"] or 0)
        return (item["gap"] or 0) / top if top else 0.0

    ranked_groups = sorted(summaries, key=relative_gap, reverse=True)
    seen_group_columns = set()
    seen_value_columns = set()
    group_insights = 0
    # Prefer variety: first pass takes a new group column AND a new value
    # column each time, so the two insights are not both about one measure.
    for require_new_value in (True, False):
        for item in ranked_groups:
            if group_insights >= 2:
                break
            if item["group_column"] in seen_group_columns or relative_gap(item) < 0.05:
                continue
            if require_new_value and item["value_column"] in seen_value_columns:
                continue
            seen_group_columns.add(item["group_column"])
            seen_value_columns.add(item["value_column"])
            group_insights += 1
            insights.append(
                f"Highest and lowest: for {item['value_column']} by "
                f"{item['group_column']}, {item['highest_group']} averages "
                f"highest ({_fmt(item['highest_mean'])}) and "
                f"{item['lowest_group']} lowest ({_fmt(item['lowest_mean'])}) — "
                f"a gap of {_fmt(item['gap'])}."
            )
            if group_insights == 1:
                decisions.append(
                    f"Largest group gap: {item['lowest_group']} averages "
                    f"{_fmt(item['lowest_mean'])} on {item['value_column']} "
                    f"against {_fmt(item['highest_mean'])} for "
                    f"{item['highest_group']}. Compare how these two groups "
                    "differ across the rest of the data — the gap says where "
                    "to look first."
                )
        if group_insights >= 2:
            break

    # --- Strongest relationship between measures ---
    pairs = analysis["correlations"]["pairs"]
    if pairs:
        top = pairs[0]
        strength = abs(top["correlation"])
        if strength >= 0.3:
            wording = "strongly" if strength >= 0.6 else "noticeably"
            insights.append(
                f"Strongest relationship: {top['column_a']} and {top['column_b']} "
                f"move together {wording} (correlation {top['correlation']:.2f}) — "
                "when one is high, the other usually is too."
            )
            decisions.append(
                f"{top['column_a']} is the strongest signal linked to "
                f"{top['column_b']} in this data. If you want to move "
                f"{top['column_b']}, start there — but correlation is not proof "
                "of cause, so test a small change before a big one."
            )
        else:
            insights.append(
                f"No strong relationships: no two measures move closely together "
                f"(the strongest is {top['column_a']} and {top['column_b']} at "
                f"just {top['correlation']:.2f}), so no single measure explains "
                "another on its own."
            )

    # --- Prediction: what the model found, and how far to trust it ---
    prediction = results.get("prediction")
    if prediction:
        score = prediction["best_test_score"]
        metric = prediction["metric"]
        identifiers = {
            column["name"]
            for column in results["profile"]["columns"]
            if column.get("is_identifier")
        }

        def is_identifier_feature(raw: str) -> bool:
            base = _strip_feature_prefix(raw)
            return any(
                base == name or base.startswith(name + "_") for name in identifiers
            )

        top_features = [
            _feature_name(item["feature"])
            for item in prediction["feature_importances"]
            if not is_identifier_feature(item["feature"])
        ][:3]
        feature_text = (
            " The signals that matter most are "
            + ", ".join(top_features)
            + "."
            if top_features
            else ""
        )
        insights.append(
            f"Prediction: {prediction['best_model']} can predict "
            f"{prediction['target']} with a test {metric} of {score:.3f}."
            + feature_text
        )
        strong = score >= 0.75 if metric == "accuracy" else score >= 0.5
        if strong:
            decisions.append(
                f"Use the model to act early on {prediction['target']}: it is "
                f"right often enough (test {metric} {score:.3f}) to prioritise "
                "attention where it points, especially when the top signals move."
            )
        else:
            decisions.append(
                f"Do not rely on the {prediction['target']} model alone yet "
                f"(test {metric} {score:.3f}) — treat it as a guide, not a "
                "verdict. More rows, or better measurements, would sharpen it."
            )
    elif results.get("prediction_refusal"):
        decisions.append(
            "Prediction was not possible this time: "
            + results["prediction_refusal"]
        )

    # --- How much data this rests on ---
    filled = filled_missing_count(results["clean_log"])
    cleaned = results.get("cleaned_rows", results["profile"]["row_count"])
    health = (
        f"Data behind this: {cleaned} rows after cleaning "
        f"({results['profile']['duplicate_rows']} duplicate row(s) removed, "
        f"{filled} missing value(s) filled)."
    )
    if cleaned < 50:
        health += (
            " That is a small base — treat the patterns above as early hints, "
            "not settled facts."
        )
    insights.append(health)

    return {"insights": insights[:6], "decisions": decisions[:4]}
