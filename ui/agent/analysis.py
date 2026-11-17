"""Descriptive analysis: statistics, correlations, groups, and trends."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _empty_trends() -> dict:
    """The empty result _trends returns when no trend can be computed."""
    return {"series": [], "changes": []}


def analyze_table(frame: pd.DataFrame, profile: dict, max_group_categories: int = 10) -> dict:
    """Run the fixed analysis set on the cleaned frame."""
    notes = []
    numeric_columns = columns_of_type(profile, frame, "numeric")
    categorical_columns = columns_of_type(profile, frame, "categorical")
    datetime_columns = columns_of_type(profile, frame, "datetime")

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


def columns_of_type(
    profile: dict, frame: pd.DataFrame, kind: str, exclude_identifiers: bool = False
) -> list:
    """Names of the profile columns of one inferred type still in the frame.

    With exclude_identifiers, identifier columns are left out: charts
    should show measured values, not row ids.
    """
    return [
        c["name"]
        for c in profile["columns"]
        if c["inferred_type"] == kind
        and (not exclude_identifiers or not c.get("is_identifier"))
        and c["name"] in frame.columns
    ]


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
        # len(value_counts) is the distinct count: the same series, no
        # mutation between the two calls, so no second nunique() pass.
        rows.append(
            {
                "column": column,
                "type": "categorical",
                "count": int(series.count()),
                "unique_values": int(len(top)),
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
                    "correlation": _round(value),
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
        # One aggregation for every measure column, instead of one groupby
        # pass per column: the same means, computed in a single sweep.
        all_means = frame.groupby(group_column)[numeric_columns].mean(
            numeric_only=True
        )
        for value_column in numeric_columns:
            grouped = all_means[value_column]
            if len(grouped) < 2 or grouped.notna().sum() == 0:
                continue
            best_group = str(grouped.idxmax())
            worst_group = str(grouped.idxmin())
            # max/min computed once: the highest and lowest means and the
            # gap between them all come from the same two values.
            highest_mean = grouped.max()
            lowest_mean = grouped.min()
            summaries.append(
                {
                    "group_column": group_column,
                    "value_column": value_column,
                    "group_means": {str(k): _round(v) for k, v in grouped.items()},
                    "highest_group": best_group,
                    "highest_mean": _round(highest_mean),
                    "lowest_group": worst_group,
                    "lowest_mean": _round(lowest_mean),
                    "gap": _round(highest_mean - lowest_mean),
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
        return _empty_trends()
    if not numeric_columns:
        notes.append("Trend analysis skipped: no numeric columns to track.")
        return _empty_trends()
    date_column = datetime_columns[0]
    work = frame[[date_column] + numeric_columns].dropna(subset=[date_column]).copy()
    work = work.sort_values(date_column).set_index(date_column)
    if len(work) < 2:
        notes.append("Trend analysis skipped: fewer than two dated rows.")
        return _empty_trends()
    span_days = (work.index.max() - work.index.min()).days
    rule = "ME" if span_days > 120 else "W" if span_days > 21 else "D"
    monthly = work.resample(rule).mean(numeric_only=True)
    series = [
        {"period": idx.strftime("%Y-%m-%d"), **{c: _round(v) for c, v in row.items()}}
        for idx, row in monthly.iterrows()
    ]
    changes = []
    for column in numeric_columns:
        values = monthly[column].dropna()
        if len(values) < 2 or values.iloc[0] == 0:
            continue
        start, end = float(values.iloc[0]), float(values.iloc[-1])
        percent_change = round((end - start) / abs(start) * 100, 2)
        if abs(percent_change) > 1000:
            # A near-zero starting value makes the percentage meaningless
            # (a rise from 0.3 to 80 is not a useful "24,900%"); the series
            # itself is still reported, just without a headline percentage.
            continue
        changes.append(
            {
                "column": column,
                "first_period": monthly.index[0].strftime("%Y-%m-%d"),
                "last_period": monthly.index[-1].strftime("%Y-%m-%d"),
                "first_value": _round(start),
                "last_value": _round(end),
                "percent_change": percent_change,
            }
        )
    return {"series": series, "changes": changes}


def _round(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), 4)
