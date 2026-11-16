"""Cleaning:

The original file is never modified. Every change is appended to the
transformation log with the rule applied and the number of rows or cells
affected, so the cleaning can be inspected and replayed.
"""
from __future__ import annotations

import pandas as pd


def clean_table(frame: pd.DataFrame, profile: dict) -> tuple[pd.DataFrame, list]:
    """Apply the fixed cleaning steps and return (cleaned frame, log)."""
    clean_frame = frame.copy()
    log = []

    # 1. Drop exact duplicate rows.
    before = len(clean_frame)
    clean_frame = clean_frame.drop_duplicates().reset_index(drop=True)
    removed = before - len(clean_frame)
    if removed:
        log.append(
            {
                "step": "drop_duplicate_rows",
                "detail": f"Removed {removed} exact duplicate row(s).",
                "rows_affected": removed,
            }
        )

    # 2. Strip surrounding whitespace from text cells.
    text_columns = [
        c["name"]
        for c in profile["columns"]
        if c["inferred_type"] in ("categorical", "text")
    ]
    changed_cells = 0
    for column in text_columns:
        if column not in clean_frame.columns:
            continue
        stripped = clean_frame[column].map(
            lambda v: v.strip() if isinstance(v, str) else v
        )
        changed_cells += int((stripped != clean_frame[column]).sum())
        clean_frame[column] = stripped
    if changed_cells:
        log.append(
            {
                "step": "strip_whitespace",
                "detail": f"Trimmed whitespace in {changed_cells} cell(s).",
                "rows_affected": changed_cells,
            }
        )

    # 3. Unify case variants of the same category (" North", "NORTH", "North"
    # all become the most frequent spelling in the group).
    for column in text_columns:
        if column not in clean_frame.columns:
            continue
        unified, groups_fixed = _unify_case_variants(clean_frame[column])
        clean_frame[column] = unified
        if groups_fixed:
            log.append(
                {
                    "step": "unify_category_spelling",
                    "column": column,
                    "detail": f"Unified {groups_fixed} spelling variant group(s).",
                    "rows_affected": groups_fixed,
                }
            )

    # 4. Coerce columns the profiler flagged as numeric or datetime.
    for column_info in profile["columns"]:
        column = column_info["name"]
        if column not in clean_frame.columns:
            continue
        if column_info["inferred_type"] == "numeric" and not pd.api.types.is_numeric_dtype(
            clean_frame[column]
        ):
            clean_frame[column] = pd.to_numeric(clean_frame[column], errors="coerce")
            log.append(
                {
                    "step": "coerce_to_numeric",
                    "column": column,
                    "detail": "Converted text that looks like numbers to numeric values.",
                    "rows_affected": int(clean_frame[column].notna().sum()),
                }
            )
        elif column_info["inferred_type"] == "datetime" and not pd.api.types.is_datetime64_any_dtype(
            clean_frame[column]
        ):
            clean_frame[column] = pd.to_datetime(
                clean_frame[column], errors="coerce", format="mixed"
            )
            log.append(
                {
                    "step": "coerce_to_datetime",
                    "column": column,
                    "detail": "Parsed date-like text into dates.",
                    "rows_affected": int(clean_frame[column].notna().sum()),
                }
            )

    # 5. Fill remaining missing values: median for numeric, mode for categorical.
    for column_info in profile["columns"]:
        column = column_info["name"]
        if column not in clean_frame.columns:
            continue
        missing = int(clean_frame[column].isna().sum())
        if missing == 0:
            continue
        if column_info["inferred_type"] == "numeric":
            fill_value = clean_frame[column].median()
            clean_frame[column] = clean_frame[column].fillna(fill_value)
            log.append(
                {
                    "step": "fill_missing_median",
                    "column": column,
                    "detail": f"Filled {missing} missing value(s) with the median ({fill_value:.4g}).",
                    "rows_affected": missing,
                }
            )
        elif column_info["inferred_type"] in ("categorical", "boolean"):
            modes = clean_frame[column].mode(dropna=True)
            if len(modes) > 0:
                fill_value = modes.iloc[0]
                clean_frame[column] = clean_frame[column].fillna(fill_value)
                log.append(
                    {
                        "step": "fill_missing_mode",
                        "column": column,
                        "detail": f"Filled {missing} missing value(s) with the most common value ({fill_value!r}).",
                        "rows_affected": missing,
                    }
                )
        else:
            log.append(
                {
                    "step": "missing_left_unfilled",
                    "column": column,
                    "detail": f"Left {missing} missing value(s) unfilled (no safe default for this type).",
                    "rows_affected": missing,
                }
            )

    return clean_frame, log


def _unify_case_variants(series: pd.Series) -> tuple[pd.Series, int]:
    """Map spelling variants of one category to the most frequent spelling."""
    groups: dict[str, dict[str, int]] = {}
    for value in series.dropna():
        if not isinstance(value, str):
            continue
        key = value.strip().lower()
        groups.setdefault(key, {})
        groups[key][value] = groups[key].get(value, 0) + 1
    mapping = {}
    fixed = 0
    for key, spellings in groups.items():
        if len(spellings) > 1:
            most_common = max(spellings.items(), key=lambda item: item[1])[0]
            for spelling in spellings:
                if spelling != most_common:
                    mapping[spelling] = most_common
            fixed += 1
    if not mapping:
        return series, 0
    return series.map(lambda v: mapping.get(v, v)), fixed
