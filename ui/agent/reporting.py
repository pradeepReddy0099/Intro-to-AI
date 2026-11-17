"""Write HTML and Markdown reports from computed results only."""
from __future__ import annotations

import html
import json
from pathlib import Path

from agent.insights import _fmt as _format_number, build_insights, filled_missing_count

STYLE = """
body { font-family: sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #222; }
h1, h2 { color: #1a3a5c; }
table { border-collapse: collapse; margin: 1rem 0; }
th, td { border: 1px solid #ccc; padding: 0.4rem 0.7rem; text-align: left; }
th { background: #eef2f7; }
img { max-width: 100%; margin: 1rem 0; border: 1px solid #ddd; }
.answer { background: #f6f8fa; border-left: 4px solid #1a3a5c; padding: 0.8rem 1rem; }
code { background: #f0f0f0; padding: 0.1rem 0.3rem; }
"""


def build_report(results: dict, out_dir) -> dict:
    """Write report.html, report.md, and run artifacts. Returns file paths."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    html_path = out_path / "report.html"
    md_path = out_path / "report.md"
    # The two formats render the same collected data, so gather it once
    # here instead of once per format: _report_data rebuilds the insights
    # and the direct answer each time it runs.
    data = _report_data(results)
    html_path.write_text(_render_html(data), encoding="utf-8")
    md_path.write_text(_render_markdown(data), encoding="utf-8")
    side_files = {
        "config.json": results["config"],
        "transformation_log.json": results["clean_log"],
        "profile.json": results["profile"],
    }
    for file_name, side_data in side_files.items():
        (out_path / file_name).write_text(json.dumps(side_data, indent=2), encoding="utf-8")
    return {
        "html": str(html_path),
        "markdown": str(md_path),
        "config": str(out_path / "config.json"),
        "transformation_log": str(out_path / "transformation_log.json"),
        "profile": str(out_path / "profile.json"),
    }


def build_refusal_report(reason: str, out_dir) -> dict:
    """Write a short report explaining why the run was refused."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    html_text = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Analysis refused</title><style>{STYLE}</style></head><body>
<h1>Analysis refused</h1>
<p>The agent did not produce an analysis because a guardrail failed:</p>
<p class="answer">{html.escape(reason)}</p>
<p>Fix the issue above and run the command again.</p>
</body></html>"""
    md_text = f"# Analysis refused\n\nThe agent did not produce an analysis because a guardrail failed:\n\n> {reason}\n\nFix the issue above and run the command again.\n"
    (out_path / "report.html").write_text(html_text, encoding="utf-8")
    (out_path / "report.md").write_text(md_text, encoding="utf-8")
    return {"html": str(out_path / "report.html"), "markdown": str(out_path / "report.md")}


def direct_answer(results: dict) -> str:
    """One or two template sentences answering the user's question."""
    profile = results["profile"]
    analysis = results["analysis"]
    question_info = results["question_info"]
    intent = question_info["intent"]
    filled = filled_missing_count(results["clean_log"])
    summary = (
        f"The dataset has {profile['row_count']} rows and {profile['column_count']} columns. "
        f"Cleaning removed {profile['duplicate_rows']} duplicate row(s) "
        f"and filled {filled} missing value(s)."
    )
    if intent == "correlation":
        pairs = analysis["correlations"]["pairs"]
        if pairs:
            top = pairs[0]
            return summary + (
                f" The strongest relationship is between {top['column_a']} and "
                f"{top['column_b']} (correlation {top['correlation']:.2f})."
            )
        return summary + " No correlations could be computed."
    if intent == "trend":
        changes = analysis["trends"]["changes"]
        if changes:
            biggest = max(changes, key=lambda item: abs(item["percent_change"]))
            return (
                f"From {biggest['first_period']} to {biggest['last_period']}, "
                f"{biggest['column']} changed by {biggest['percent_change']:+.1f}% "
                f"({biggest['first_value']} to {biggest['last_value']})."
            )
        return summary + " No trend could be computed."
    if intent == "compare":
        summaries = analysis["group_summaries"]
        if summaries:
            # Answer about the column the question names, when it has one.
            # The focus is the longest column name in the question, which
            # can be the group column (e.g. product_category), so fall back
            # to the longest numeric column named in the question.
            value_column = None
            focus = question_info.get("focus_column")
            if focus and any(s["value_column"] == focus for s in summaries):
                value_column = focus
            else:
                original = (question_info.get("original") or "").lower()
                numeric = [
                    c["name"]
                    for c in profile["columns"]
                    if c.get("inferred_type") == "numeric"
                    and str(c["name"]).lower() in original
                ]
                if numeric:
                    value_column = max(numeric, key=lambda c: len(str(c)))
            if value_column:
                focused = [s for s in summaries if s["value_column"] == value_column]
                if focused:
                    summaries = focused
            biggest = max(summaries, key=lambda item: item["gap"] or 0)
            return (
                f"For {biggest['value_column']} grouped by {biggest['group_column']}, "
                f"the largest gap is {biggest['highest_group']} "
                f"(average {biggest['highest_mean']}) versus {biggest['lowest_group']} "
                f"(average {biggest['lowest_mean']})."
            )
        return summary + " No group comparison could be computed."
    if intent == "predict":
        prediction = results.get("prediction")
        if prediction:
            return (
                f"For target {prediction['target']}, the best model was "
                f"{prediction['best_model']} with a test {prediction['metric']} of "
                f"{prediction['best_test_score']:.3f}."
            )
        return results.get("prediction_refusal", "Prediction was not run.")
    return summary


PROFILE_COLUMNS = ["name", "inferred_type", "missing_count", "missing_rate", "unique_count"]
LOG_COLUMNS = ["step", "column", "detail"]
DESCRIPTIVE_COLUMNS = [
    "column", "type", "count", "mean", "std", "min", "median", "max",
    "unique_values", "most_common", "most_common_count",
]
CORRELATION_COLUMNS = ["column_a", "column_b", "correlation"]
GROUP_COLUMNS = [
    "grouped_by", "value", "highest_group", "highest_mean",
    "lowest_group", "lowest_mean", "gap",
]
TREND_COLUMNS = [
    "column", "first_period", "last_period", "first_value",
    "last_value", "percent_change",
]
MODEL_COLUMNS = ["model", "cv_mean", "cv_std", "test_score"]
FEATURE_COLUMNS = ["feature", "importance"]


def _report_data(results: dict) -> dict:
    """Collect everything the two report formats need, in one place."""
    profile = results["profile"]
    analysis = results["analysis"]
    summaries = analysis["group_summaries"]
    return {
        "input_file": results["input_file"],
        "version": results["config"]["version"],
        "seed": results["config"]["random_seed"],
        "question": results["question_info"],
        "answer": direct_answer(results),
        "insights": build_insights(results),
        "profile_rows": profile["columns"],
        "clean_rows": results["clean_log"],
        "descriptive_rows": analysis["descriptive"],
        "pairs": analysis["correlations"]["pairs"],
        "group_rows": _group_rows(summaries) if summaries else [],
        "changes": analysis["trends"]["changes"],
        "prediction": results.get("prediction"),
        "prediction_refusal": results.get("prediction_refusal", ""),
        "charts": results["charts"],
        "notes": analysis["notes"],
    }


def _html_table_section(sections: list, title: str, rows: list, columns: list, empty_text: str) -> None:
    """Append a heading plus a table, or a short note when there are no rows."""
    sections.append(f"<h2>{title}</h2>")
    if rows:
        sections.append(_records_html(rows, columns))
    else:
        sections.append(f"<p>{empty_text}</p>")


def _md_table_section(lines: list, title: str, rows: list, columns: list, empty_text: str) -> None:
    """Append a heading plus a table, or a short note when there are no rows."""
    lines.append(f"## {title}")
    lines.append("")
    lines.append(_records_markdown(rows, columns) if rows else empty_text)
    lines.append("")


def _html_list_section(sections: list, title: str, items: list) -> None:
    """Append a heading plus a bulleted list, or nothing when there are no items."""
    if not items:
        return
    sections.append(f"<h2>{title}</h2><ul>")
    sections.extend(f"<li>{html.escape(item)}</li>" for item in items)
    sections.append("</ul>")


def _md_list_section(lines: list, title: str, items: list) -> None:
    """Append a heading plus dash bullets, or nothing when there are no items."""
    if not items:
        return
    lines.append(f"## {title}")
    lines.append("")
    lines.extend(f"- {item}" for item in items)
    lines.append("")


def _prediction_header(prediction: dict) -> str:
    """One-line summary of a prediction run, shared by both report formats.

    The HTML renderer escapes the whole line; the values it contains
    (task type, metric, row counts) never hold characters that need
    escaping, and the target is escaped either way, so both renderings
    come out exactly as before.
    """
    return (
        f"Task: {prediction['task_type']} | Target: {prediction['target']} | "
        f"Metric: {prediction['metric']} | Train rows: {prediction['train_rows']} | "
        f"Test rows: {prediction['test_rows']}"
    )


def _prediction_scores(prediction: dict) -> str:
    """Cross-validation and test scores, in the parenthetical both formats use."""
    return (
        f"(cross-validation {prediction['metric']} "
        f"{prediction['best_cv_score']:.3f}, test {prediction['best_test_score']:.3f})"
    )


def _render_html(data: dict) -> str:
    question = data["question"]
    sections = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Data Analysis Report</title>",
        f"<style>{STYLE}</style></head><body>",
        "<h1>Data Analysis Report</h1>",
        f"<p>Input file: <code>{html.escape(str(data['input_file']))}</code></p>",
        f"<p>Pipeline version: {html.escape(str(data['version']))} | "
        f"Random seed: {data['seed']}</p>",
    ]
    if question["original"]:
        sections.append(
            f"<p>Question: <em>{html.escape(question['original'])}</em> "
            f"(intent: {html.escape(question['intent'])})</p>"
        )
    sections.append(
        f"<h2>Direct answer</h2><p class='answer'>{html.escape(data['answer'])}</p>"
    )

    _html_list_section(sections, "Key insights", data["insights"]["insights"])
    _html_list_section(sections, "What to do next", data["insights"]["decisions"])

    sections.append("<h2>Profile</h2>")
    sections.append(_records_html(data["profile_rows"], PROFILE_COLUMNS))

    _html_table_section(sections, "Cleaning log", data["clean_rows"], LOG_COLUMNS, "No cleaning changes were needed.")

    sections.append("<h2>Descriptive statistics</h2>")
    sections.append(_records_html(data["descriptive_rows"], DESCRIPTIVE_COLUMNS))

    _html_table_section(sections, "Correlations", data["pairs"], CORRELATION_COLUMNS, "No correlations computed.")
    _html_table_section(sections, "Group comparisons", data["group_rows"], GROUP_COLUMNS, "No group comparisons computed.")
    _html_table_section(sections, "Trends over time", data["changes"], TREND_COLUMNS, "No trends computed.")

    sections.append("<h2>Prediction</h2>")
    prediction = data["prediction"]
    if prediction:
        sections.append(
            f"<p>{html.escape(_prediction_header(prediction))}</p>"
        )
        sections.append(_records_html(prediction["models"], MODEL_COLUMNS))
        sections.append(
            f"<p>Best model: <strong>{html.escape(prediction['best_model'])}</strong> "
            f"{_prediction_scores(prediction)}.</p>"
        )
        if prediction["feature_importances"]:
            sections.append("<h3>Top features</h3>")
            sections.append(
                _records_html(prediction["feature_importances"], FEATURE_COLUMNS)
            )
        for note in prediction["notes"]:
            sections.append(f"<p>{html.escape(note)}</p>")
    else:
        sections.append(
            f"<p>Prediction not run. {html.escape(data['prediction_refusal'])}</p>"
        )

    sections.append("<h2>Charts</h2>")
    for chart in data["charts"]:
        sections.append(
            f"<h3>{html.escape(chart['title'])}</h3>"
            f"<img src='charts/{html.escape(chart['file'])}' "
            f"alt='{html.escape(chart['title'])}'>"
            f"<p>{html.escape(chart['description'])}</p>"
        )

    _html_list_section(sections, "Notes", data["notes"])

    sections.append("</body></html>")
    return "\n".join(sections)



def _render_markdown(data: dict) -> str:
    question = data["question"]
    lines = ["# Data Analysis Report", ""]
    lines.append(f"Input file: `{data['input_file']}`")
    lines.append(
        f"Pipeline version: {data['version']} | "
        f"Random seed: {data['seed']}"
    )
    lines.append("")
    if question["original"]:
        lines.append(
            f"Question: *{question['original']}* (intent: {question['intent']})"
        )
        lines.append("")
    lines.append("## Direct answer")
    lines.append("")
    lines.append(data["answer"])
    lines.append("")
    _md_list_section(lines, "Key insights", data["insights"]["insights"])
    _md_list_section(lines, "What to do next", data["insights"]["decisions"])
    lines.append("## Profile")
    lines.append("")
    lines.append(_records_markdown(data["profile_rows"], PROFILE_COLUMNS))
    _md_table_section(lines, "Cleaning log", data["clean_rows"], LOG_COLUMNS, "No cleaning changes were needed.")
    lines.append("## Descriptive statistics")
    lines.append("")
    lines.append(_records_markdown(data["descriptive_rows"], DESCRIPTIVE_COLUMNS))
    _md_table_section(lines, "Correlations", data["pairs"], CORRELATION_COLUMNS, "No correlations computed.")
    _md_table_section(lines, "Group comparisons", data["group_rows"], GROUP_COLUMNS, "No group comparisons computed.")
    _md_table_section(lines, "Trends over time", data["changes"], TREND_COLUMNS, "No trends computed.")
    lines.append("## Prediction")
    lines.append("")
    prediction = data["prediction"]
    if prediction:
        lines.append(_prediction_header(prediction))
        lines.append("")
        lines.append(_records_markdown(prediction["models"], MODEL_COLUMNS))
        lines.append(
            f"Best model: **{prediction['best_model']}** "
            f"{_prediction_scores(prediction)}."
        )
        lines.append("")
        if prediction["feature_importances"]:
            lines.append("### Top features")
            lines.append("")
            lines.append(
                _records_markdown(prediction["feature_importances"], FEATURE_COLUMNS)
            )
        for note in prediction["notes"]:
            lines.append(note)
    else:
        lines.append(f"Prediction not run. {data['prediction_refusal']}")
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    for chart in data["charts"]:
        lines.append(f"### {chart['title']}")
        lines.append("")
        lines.append(f"![{chart['title']}](charts/{chart['file']})")
        lines.append("")
        lines.append(chart["description"])
        lines.append("")
    _md_list_section(lines, "Notes", data["notes"])
    return "\n".join(lines)



def _group_rows(summaries: list) -> list:
    """Flatten group summaries into plain rows for a report table."""
    return [
        {
            "grouped_by": item["group_column"],
            "value": item["value_column"],
            "highest_group": item["highest_group"],
            "highest_mean": item["highest_mean"],
            "lowest_group": item["lowest_group"],
            "lowest_mean": item["lowest_mean"],
            "gap": item["gap"],
        }
        for item in summaries
    ]


def _table_cells(records: list, columns: list) -> list[list[str]]:
    """Each record as a row of formatted cell strings, one per column."""
    return [
        [_format_cell(record.get(col)) for col in columns] for record in records
    ]


def _records_html(records: list, columns: list) -> str:
    header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    rows = []
    for cells in _table_cells(records, columns):
        joined = "".join(f"<td>{html.escape(cell)}</td>" for cell in cells)
        rows.append(f"<tr>{joined}</tr>")
    return f"<table><tr>{header}</tr>{''.join(rows)}</table>"


def _records_markdown(records: list, columns: list) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(cells) + " |" for cells in _table_cells(records, columns)]
    return "\n".join([header, divider, *rows]) + "\n"


def _format_cell(value) -> str:
    if value is None:
        return ""
    # One float-display rule for the package (4 significant digits):
    # insights.py already defines it; a report cell is the same formatting.
    return _format_number(value)
