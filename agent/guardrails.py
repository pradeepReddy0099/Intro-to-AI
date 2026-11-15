"""Guardrails: input validation, assumption checks, explicit refusals.

The agent never returns a plausible-looking but unsupported result. When an
assumption fails, the pipeline stops that stage (or the whole run) and
records a plain-language reason instead of improvising.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}


class RefusalError(Exception):
    """Raised when the agent refuses a file, a stage, or a request."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def validate_input_file(path: str | Path) -> Path:
    """Check that the input file exists and has a supported extension."""
    input_path = Path(path)
    if not input_path.exists():
        raise RefusalError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise RefusalError(f"Input path is not a file: {input_path}")
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise RefusalError(
            f"Unsupported file type '{input_path.suffix}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    return input_path


def validate_frame(frame: pd.DataFrame, source_name: str) -> None:
    """Check that a loaded table is usable at all."""
    if frame is None or frame.shape[0] == 0:
        raise RefusalError(f"{source_name} contains no data rows.")
    if frame.shape[1] == 0:
        raise RefusalError(f"{source_name} contains no columns.")
    if frame.columns.duplicated().any():
        raise RefusalError(
            f"{source_name} has duplicate column names. "
            "Rename the columns and try again."
        )


def check_prediction_ready(frame: pd.DataFrame, target: str, config) -> str:
    """Check that prediction is defensible for the given target.

    Returns "classification" or "regression" when the checks pass, and
    raises RefusalError with a plain-language reason otherwise.
    """
    if target not in frame.columns:
        raise RefusalError(
            f"Target column '{target}' is not in the dataset. "
            f"Available columns: {', '.join(str(c) for c in frame.columns)}."
        )
    if len(frame) < config.min_rows_for_prediction:
        raise RefusalError(
            f"Prediction needs at least {config.min_rows_for_prediction} rows, "
            f"but the dataset has only {len(frame)} rows after cleaning."
        )
    target_series = frame[target]
    usable = target_series.dropna()
    if len(usable) == 0:
        raise RefusalError(f"Target column '{target}' is entirely missing.")
    if usable.nunique() < 2:
        raise RefusalError(
            f"Target column '{target}' has only one distinct value "
            f"({usable.iloc[0]!r}). Nothing to predict."
        )

    is_numeric = pd.api.types.is_numeric_dtype(target_series)
    if is_numeric and usable.nunique() > 10:
        if usable.var() == 0:
            raise RefusalError(f"Target column '{target}' has zero variance.")
        return "regression"

    class_counts = usable.value_counts()
    small_classes = class_counts[class_counts < config.min_class_samples]
    if len(small_classes) > 0:
        raise RefusalError(
            f"Classification target '{target}' has classes with fewer than "
            f"{config.min_class_samples} examples: "
            + ", ".join(f"{name} ({count})" for name, count in small_classes.items())
            + ". Add more data or drop the rare classes."
        )
    return "classification"
