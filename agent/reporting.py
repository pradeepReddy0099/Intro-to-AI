"""Report generation: HTML and Markdown, from computed values only.

Every explanatory sentence is a fixed template filled in from statistics the
pipeline actually calculated. The report also embeds the configuration and
the transformation log, so a reader can audit exactly what was done.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

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
    html_path.write_text(_render_html(results), encoding="utf-8")
    md_path.write_text(_render_markdown(results), encoding="utf-8")
    (out_path / "config.json").write_text(
        json.dumps(results["config"], indent=2), encoding="utf-8"
    )
    (out_path / "transformation_log.json").write_text(
        json.dumps(results["clean_log"], indent=2), encoding="utf-8"
    )
    (out_path / "profile.json").write_text(
        json.dumps(results["profile"], indent=2), encoding="utf-8"
    )
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
    filled = sum(
        entry.get("rows_affected", 0)
        for entry in results["clean_log"]
        if entry["step"].startswith("fill_missing")
    )
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


def _render_html(results: dict) -> str:
    profile = results["profile"]
    analysis = results["analysis"]
    question_info = results["question_info"]
    sections = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Data Analysis Report</title>",
        f"<style>{STYLE}</style></head><body>",
        "<h1>Data Analysis Report</h1>",
        f"<p>Input file: <code>{html.escape(str(results['input_file']))}</code></p>",
        f"<p>Pipeline version: {html.escape(str(results['config']['version']))} | "
        f"Random seed: {results['config']['random_seed']}</p>",
    ]
    if question_info["original"]:
        sections.append(
            f"<p>Question: <em>{html.escape(question_info['original'])}</em> "
            f"(intent: {html.escape(question_info['intent'])})</p>"
        )
    sections.append(
        f"<h2>Direct answer</h2><p class='answer'>{html.escape(direct_answer(results))}</p>"
    )

    sections.append("<h2>Profile</h2>")
    sections.append(
        _records_html(
            profile["columns"],
            ["name", "inferred_type", "missing_count", "missing_rate", "unique_count"],
        )
    )

    sections.append("<h2>Cleaning log</h2>")
    if results["clean_log"]:
        sections.append(_records_html(results["clean_log"], ["step", "column", "detail"]))
    else:
        sections.append("<p>No cleaning changes were needed.</p>")

    sections.append("<h2>Descriptive statistics</h2>")
    sections.append(
        _records_html(
            analysis["descriptive"],
            ["column", "type", "count", "mean", "std", "min", "median", "max",
             "unique_values", "most_common", "most_common_count"],
        )
    )

    sections.append("<h2>Correlations</h2>")
    pairs = analysis["correlations"]["pairs"]
    if pairs:
        sections.append(_records_html(pairs, ["column_a", "column_b", "correlation"]))
    else:
        sections.append("<p>No correlations computed.</p>")

    sections.append("<h2>Group comparisons</h2>")
    summaries = analysis["group_summaries"]
    if summaries:
        rows = [
            {
                "grouped_by": s["group_column"],
                "value": s["value_column"],
                "highest_group": s["highest_group"],
                "highest_mean": s["highest_mean"],
                "lowest_group": s["lowest_group"],
                "lowest_mean": s["lowest_mean"],
                "gap": s["gap"],
            }
            for s in summaries
        ]
        sections.append(
            _records_html(
                rows,
                ["grouped_by", "value", "highest_group", "highest_mean",
                 "lowest_group", "lowest_mean", "gap"],
            )
        )
    else:
        sections.append("<p>No group comparisons computed.</p>")

    sections.append("<h2>Trends over time</h2>")
    changes = analysis["trends"]["changes"]
    if changes:
        sections.append(
            _records_html(
                changes,
                ["column", "first_period", "last_period", "first_value",
                 "last_value", "percent_change"],
            )
        )
    else:
        sections.append("<p>No trends computed.</p>")

    sections.append("<h2>Prediction</h2>")
    prediction = results.get("prediction")
    if prediction:
        sections.append(
            f"<p>Task: {prediction['task_type']} | Target: "
            f"{html.escape(prediction['target'])} | Metric: {prediction['metric']} | "
            f"Train rows: {prediction['train_rows']} | Test rows: {prediction['test_rows']}</p>"
        )
        sections.append(
            _records_html(prediction["models"], ["model", "cv_mean", "cv_std", "test_score"])
        )
        sections.append(
            f"<p>Best model: <strong>{html.escape(prediction['best_model'])}</strong> "
            f"(cross-validation {prediction['metric']} "
            f"{prediction['best_cv_score']:.3f}, test {prediction['best_test_score']:.3f}).</p>"
        )
        if prediction["feature_importances"]:
            sections.append("<h3>Top features</h3>")
            sections.append(
                _records_html(prediction["feature_importances"], ["feature", "importance"])
            )
        for note in prediction["notes"]:
            sections.append(f"<p>{html.escape(note)}</p>")
    else:
        sections.append(
            f"<p>Prediction not run. {html.escape(results.get('prediction_refusal', ''))}</p>"
        )

    sections.append("<h2>Charts</h2>")
    for chart in results["charts"]:
        sections.append(
            f"<h3>{html.escape(chart['title'])}</h3>"
            f"<img src='charts/{html.escape(chart['file'])}' "
            f"alt='{html.escape(chart['title'])}'>"
            f"<p>{html.escape(chart['description'])}</p>"
        )

    if analysis["notes"]:
        sections.append("<h2>Notes</h2><ul>")
        sections.extend(f"<li>{html.escape(note)}</li>" for note in analysis["notes"])
        sections.append("</ul>")

    sections.append("</body></html>")
    return "\n".join(sections)


def _render_markdown(results: dict) -> str:
    profile = results["profile"]
    analysis = results["analysis"]
    question_info = results["question_info"]
    lines = ["# Data Analysis Report", ""]
    lines.append(f"Input file: `{results['input_file']}`")
    lines.append(
        f"Pipeline version: {results['config']['version']} | "
        f"Random seed: {results['config']['random_seed']}"
    )
    lines.append("")
    if question_info["original"]:
        lines.append(
            f"Question: *{question_info['original']}* (intent: {question_info['intent']})"
        )
        lines.append("")
    lines.append("## Direct answer")
    lines.append("")
    lines.append(direct_answer(results))
    lines.append("")
    lines.append("## Profile")
    lines.append("")
    lines.append(
        _records_markdown(
            profile["columns"],
            ["name", "inferred_type", "missing_count", "missing_rate", "unique_count"],
        )
    )
    lines.append("## Cleaning log")
    lines.append("")
    if results["clean_log"]:
        lines.append(_records_markdown(results["clean_log"], ["step", "column", "detail"]))
    else:
        lines.append("No cleaning changes were needed.")
    lines.append("")
    lines.append("## Descriptive statistics")
    lines.append("")
    lines.append(
        _records_markdown(
            analysis["descriptive"],
            ["column", "type", "count", "mean", "std", "min", "median", "max",
             "unique_values", "most_common", "most_common_count"],
        )
    )
    lines.append("## Correlations")
    lines.append("")
    pairs = analysis["correlations"]["pairs"]
    lines.append(
        _records_markdown(pairs, ["column_a", "column_b", "correlation"])
        if pairs
        else "No correlations computed."
    )
    lines.append("")
    lines.append("## Group comparisons")
    lines.append("")
    summaries = analysis["group_summaries"]
    if summaries:
        rows = [
            {
                "grouped_by": s["group_column"],
                "value": s["value_column"],
                "highest_group": s["highest_group"],
                "highest_mean": s["highest_mean"],
                "lowest_group": s["lowest_group"],
                "lowest_mean": s["lowest_mean"],
                "gap": s["gap"],
            }
            for s in summaries
        ]
        lines.append(
            _records_markdown(
                rows,
                ["grouped_by", "value", "highest_group", "highest_mean",
                 "lowest_group", "lowest_mean", "gap"],
            )
        )
    else:
        lines.append("No group comparisons computed.")
    lines.append("")
    lines.append("## Trends over time")
    lines.append("")
    changes = analysis["trends"]["changes"]
    lines.append(
        _records_markdown(
            changes,
            ["column", "first_period", "last_period", "first_value",
             "last_value", "percent_change"],
        )
        if changes
        else "No trends computed."
    )
    lines.append("")
    lines.append("## Prediction")
    lines.append("")
    prediction = results.get("prediction")
    if prediction:
        lines.append(
            f"Task: {prediction['task_type']} | Target: {prediction['target']} | "
            f"Metric: {prediction['metric']} | Train rows: {prediction['train_rows']} | "
            f"Test rows: {prediction['test_rows']}"
        )
        lines.append("")
        lines.append(
            _records_markdown(prediction["models"], ["model", "cv_mean", "cv_std", "test_score"])
        )
        lines.append(
            f"Best model: **{prediction['best_model']}** "
            f"(cross-validation {prediction['metric']} {prediction['best_cv_score']:.3f}, "
            f"test {prediction['best_test_score']:.3f})."
        )
        lines.append("")
        if prediction["feature_importances"]:
            lines.append("### Top features")
            lines.append("")
            lines.append(
                _records_markdown(prediction["feature_importances"], ["feature", "importance"])
            )
        for note in prediction["notes"]:
            lines.append(note)
    else:
        lines.append(f"Prediction not run. {results.get('prediction_refusal', '')}")
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    for chart in results["charts"]:
        lines.append(f"### {chart['title']}")
        lines.append("")
        lines.append(f"![{chart['title']}](charts/{chart['file']})")
        lines.append("")
        lines.append(chart["description"])
        lines.append("")
    if analysis["notes"]:
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {note}" for note in analysis["notes"])
        lines.append("")
    return "\n".join(lines)


def _records_html(records: list, columns: list) -> str:
    header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    rows = []
    for record in records:
        cells = "".join(
            f"<td>{html.escape(_format_cell(record.get(col)))}</td>" for col in columns
        )
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><tr>{header}</tr>{''.join(rows)}</table>"


def _records_markdown(records: list, columns: list) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_format_cell(record.get(col)) for col in columns) + " |"
        for record in records
    ]
    return "\n".join([header, divider, *rows]) + "\n"


def _format_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
