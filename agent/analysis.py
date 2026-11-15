"""Analysis: descriptive statistics, correlations, group comparisons, trends.

Every number reported downstream comes from these functions. Nothing is
estimated, rounded away silently, or invented: if an analysis cannot be
computed (too few columns, no date column, ...), it is skipped and the
reason is recorded in `notes`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_table(frame: pd.DataFrame, profile: dict, max_group_categories: int = 10) -> dict:
    """Run the fixed analysis set on the cleaned frame."""
    notes = []
    numeric_columns = [
        c["name"]
        for c in profile["columns"]
        if c["inferred_type"] == "numeric" and c["name"] in frame.columns
    ]
    categorical_columns = [
        c["name"]
        for c in profile["columns"]
        if c["inferred_type"] == "categorical" and c["name"] in frame.columns
    ]
    datetime_columns = [
        c["name"]
        for c in profile["columns"]
        if c["inferred_type"] == "datetime" and c["name"] in frame.columns
    ]

    identifier_columns = {
        c["name"] for c in profile["columns"] if c.get("is_identifier")
    }
    measure_columns = [c for c in numeric_columns if c not in identifier_columns]

    return {
        "descriptive": _descriptive(frame, numeric_columns, categorical_columns),
        "correlations": _correlations(frame, measure_columns, notes),
        "group_summaries": _group_summaries(
            frame, measure_columns, categorical_columns, max_group_categories, notes
        ),
        "trends": _trends(frame, measure_columns, datetime_columns, notes),
        "notes": notes,
    }


def _descriptive(frame, numeric_columns, categorical_columns) -> list:
    rows = []
    for column in numeric_columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        rows.append(
            {
                "column": column,
                "type": "numeric",
                "count": int(series.count()),
                "mean": _round(series.mean()),
                "std": _round(series.std()),
                "min": _round(series.min()),
                "median": _round(series.median()),
                "max": _round(series.max()),
            }
        )
    for column in categorical_columns:
        series = frame[column].dropna()
        top = series.value_counts()
        rows.append(
            {
                "column": column,
                "type": "categorical",
                "count": int(series.count()),
                "unique_values": int(series.nunique()),
                "most_common": str(top.index[0]) if len(top) else None,
                "most_common_count": int(top.iloc[0]) if len(top) else 0,
            }
        )
    return rows


def _correlations(frame, numeric_columns, notes) -> dict:
    if len(numeric_columns) < 2:
        notes.append("Correlation skipped: fewer than two numeric columns.")
        return {"pairs": []}
    corr = frame[numeric_columns].corr(numeric_only=True)
    pairs = []
    for i, col_a in enumerate(numeric_columns):
        for col_b in numeric_columns[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.isna(value):
                continue
            pairs.append(
                {
                    "column_a": col_a,
                    "column_b": col_b,
                    "correlation": round(float(value), 4),
                }
            )
    pairs.sort(key=lambda pair: abs(pair["correlation"]), reverse=True)
    return {"pairs": pairs}


def _group_summaries(frame, numeric_columns, categorical_columns, max_categories, notes) -> list:
    summaries = []
    for group_column in categorical_columns:
        groups = frame[group_column].dropna().unique()
        if len(groups) < 2 or len(groups) > max_categories:
            continue
        for value_column in numeric_columns:
            grouped = frame.groupby(group_column)[value_column].mean(numeric_only=True)
            if len(grouped) < 2:
                continue
            best_group = str(grouped.idxmax())
            worst_group = str(grouped.idxmin())
            summaries.append(
                {
                    "group_column": group_column,
                    "value_column": value_column,
                    "group_means": {str(k): _round(v) for k, v in grouped.items()},
                    "highest_group": best_group,
                    "highest_mean": _round(grouped.max()),
                    "lowest_group": worst_group,
                    "lowest_mean": _round(grouped.min()),
                    "gap": _round(grouped.max() - grouped.min()),
                }
            )
    if not summaries:
        notes.append(
            "Group comparison skipped: no categorical column with 2-"
            f"{max_categories} groups and no numeric column to compare."
        )
    return summaries


def _trends(frame, numeric_columns, datetime_columns, notes) -> dict:
    if not datetime_columns:
        notes.append("Trend analysis skipped: no date column found.")
        return {"series": [], "changes": []}
    if not numeric_columns:
        notes.append("Trend analysis skipped: no numeric columns to track.")
        return {"series": [], "changes": []}
    date_column = datetime_columns[0]
    work = frame[[date_column] + numeric_columns].dropna(subset=[date_column]).copy()
    work = work.sort_values(date_column).set_index(date_column)
    if len(work) < 2:
        notes.append("Trend analysis skipped: fewer than two dated rows.")
        return {"series": [], "changes": []}
    span_days = (work.index.max() - work.index.min()).days
    rule = "ME" if span_days > 120 else "W" if span_days > 21 else "D"
    monthly = work.resample(rule).mean(numeric_only=True)
    series = [
        {"period": idx.strftime("%Y-%m-%d"), **{c: _round(v) for c, v in row.items()}}
        for idx, row in monthly.iterrows()
    ]
    changes = []
    for column in numeric_columns:
        first = monthly[column].dropna()
        if len(first) < 2 or first.iloc[0] == 0:
            continue
        start, end = float(first.iloc[0]), float(first.iloc[-1])
        changes.append(
            {
                "column": column,
                "first_period": monthly.index[0].strftime("%Y-%m-%d"),
                "last_period": monthly.index[-1].strftime("%Y-%m-%d"),
                "first_value": _round(start),
                "last_value": _round(end),
                "percent_change": round((end - start) / abs(start) * 100, 2),
            }
        )
    return {"series": series, "changes": changes}


def _round(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), 4)
