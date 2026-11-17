"""Describe each column before cleaning."""
from __future__ import annotations

import pandas as pd

from agent.analysis import _round as _safe_float
# One rounding rule for the whole package: round to 4 decimals, None when
# the value is missing. analysis.py already defines it; profiling uses the
# same rule for the profile's numeric summaries.


def infer_column_type(series: pd.Series, unique_count: int | None = None) -> str:
    """Infer one of numeric / datetime / boolean / categorical / text.

    `unique_count`, when given, is the series' distinct non-missing count
    (series.nunique(dropna=True)) already computed by the caller, so the
    categorical check below does not count the distinct values twice.
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    non_missing = series.dropna()
    if len(non_missing) == 0:
        return "text"

    as_text = non_missing.astype(str)
    numeric_like = pd.to_numeric(as_text, errors="coerce").notna().mean()
    if numeric_like >= 0.8:
        return "numeric"
    datetime_like = pd.to_datetime(
        as_text, errors="coerce", format="mixed"
    ).notna().mean()
    if datetime_like >= 0.8:
        return "datetime"
    if unique_count is None:
        unique_count = non_missing.nunique()
    if unique_count <= 20:
        return "categorical"
    return "text"


def profile_table(frame: pd.DataFrame) -> dict:
    """Build a JSON-friendly profile of the table."""
    column_profiles = []
    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        entry = {
            "name": str(column),
            "inferred_type": infer_column_type(series, unique_count),
            "missing_count": missing,
            "missing_rate": _safe_float(missing / len(frame)) if len(frame) else 0.0,
            "unique_count": unique_count,
            "is_identifier": _looks_like_identifier(
                str(column), unique_count, len(frame) - missing
            ),
        }
        if entry["inferred_type"] == "numeric":
            numeric_series = pd.to_numeric(series, errors="coerce")
            entry.update(
                {
                    "mean": _safe_float(numeric_series.mean()),
                    "std": _safe_float(numeric_series.std()),
                    "min": _safe_float(numeric_series.min()),
                    "max": _safe_float(numeric_series.max()),
                }
            )
        elif entry["inferred_type"] == "categorical":
            top_values = series.value_counts(dropna=True).head(5)
            entry["top_values"] = [
                {"value": str(value), "count": int(count)}
                for value, count in top_values.items()
            ]
        column_profiles.append(entry)

    return {
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "columns": column_profiles,
    }


def _looks_like_identifier(name: str, unique_count: int, non_missing_count: int) -> bool:
    """Return True when a column looks like a row ID, not a measurement."""
    lowered = name.strip().lower()
    if lowered != "id" and not lowered.endswith("_id"):
        return False
    if non_missing_count == 0:
        return False
    return unique_count / non_missing_count >= 0.98
