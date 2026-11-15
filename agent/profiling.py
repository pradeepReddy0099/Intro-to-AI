"""Profiling: describe the raw table before anything is changed.

The profile drives every later decision (which cleaning steps run, which
charts are drawn, which models are eligible), and it is saved with the run.
"""
from __future__ import annotations

import pandas as pd


def infer_column_type(series: pd.Series) -> str:
    """Infer one of numeric / datetime / boolean / categorical / text."""
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
    if non_missing.nunique() <= 20:
        return "categorical"
    return "text"


def profile_table(frame: pd.DataFrame) -> dict:
    """Build a JSON-friendly profile of the table."""
    column_profiles = []
    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        entry = {
            "name": str(column),
            "inferred_type": infer_column_type(series),
            "missing_count": missing,
            "missing_rate": round(missing / len(frame), 4) if len(frame) else 0.0,
            "unique_count": int(series.nunique(dropna=True)),
            "is_identifier": _looks_like_identifier(str(column), series),
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


def _looks_like_identifier(name: str, series: pd.Series) -> bool:
    """True for ID-style columns: named like an ID and (nearly) all-unique.

    An identifier labels rows rather than measuring anything, so charts and
    correlations skip it. Both conditions must hold — the name alone could
    be a real measurement, and uniqueness alone could be a precise value
    like revenue.
    """
    lowered = name.strip().lower()
    if lowered != "id" and not lowered.endswith("_id"):
        return False
    non_missing = series.dropna()
    if len(non_missing) == 0:
        return False
    return non_missing.nunique() / len(non_missing) >= 0.98


def _safe_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)
