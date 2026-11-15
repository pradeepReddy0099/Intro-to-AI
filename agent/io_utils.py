"""Loading for CSV, TSV, and Excel inputs. No network access is used anywhere."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from agent.guardrails import RefusalError, validate_input_file


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV, TSV, or Excel file into a DataFrame.

    Raises RefusalError with a plain-language reason for unreadable,
    malformed, or unsupported inputs.
    """
    input_path = validate_input_file(path)
    suffix = input_path.suffix.lower()
    try:
        if suffix == ".csv":
            # sep=None lets pandas work out the delimiter (comma, semicolon,
            # tab, ...) instead of assuming commas.
            frame = pd.read_csv(
                input_path, sep=None, engine="python", encoding="utf-8-sig"
            )
        elif suffix == ".tsv":
            frame = pd.read_csv(input_path, sep="\t", encoding="utf-8-sig")
        else:
            frame = pd.read_excel(input_path, engine="openpyxl")
    except Exception as exc:  # unreadable or malformed file
        raise RefusalError(f"Could not read {input_path}: {exc}") from exc
    return frame
