"""Tests for cleaning, on several different tables."""
from pathlib import Path

import pandas as pd

from agent.cleaning import clean_table
from agent.io_utils import load_table
from agent.profiling import profile_table

HERE = Path(__file__).resolve().parent
SALES = HERE.parent / "data" / "sample_sales.csv"
CHURN = HERE.parent / "data" / "sample_churn.csv"


def _steps(log):
    return {entry["step"] for entry in log}


def test_cleaning_steps_and_log():
    frame = pd.DataFrame(
        {
            "region": [" North", "NORTH", "North", "South", "South"],
            "sales": [10.0, None, 30.0, 40.0, 40.0],
        }
    )
    profile = profile_table(frame)
    clean_frame, log = clean_table(frame, profile)

    # The two (South, 40.0) rows are exact duplicates, so one is dropped
    # first; the median of the remaining sales {10, 30, 40} is 30.
    assert len(clean_frame) == 4
    assert set(clean_frame["region"].unique()) == {"North", "South"}
    assert clean_frame["sales"].isna().sum() == 0
    # The missing value in the second (North) row is filled with the median 30.
    assert clean_frame.loc[clean_frame["region"] == "North", "sales"].tolist() == [
        10.0,
        30.0,
        30.0,
    ]
    steps = _steps(log)
    assert "drop_duplicate_rows" in steps
    assert "unify_category_spelling" in steps
    assert "fill_missing_median" in steps
    assert all("detail" in entry for entry in log)


def test_duplicate_rows_removed():
    frame = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    profile = profile_table(frame)
    clean_frame, log = clean_table(frame, profile)
    assert len(clean_frame) == 2
    assert any(entry["step"] == "drop_duplicate_rows" for entry in log)


def test_cleaning_on_weather_table():
    frame = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-03", "2025-01-04"],
            "city": [" Boston", "BOSTON", "Boston", "Boston", "Denver"],
            "rain_mm": [10.0, None, 30.0, 30.0, 50.0],
        }
    )
    clean_frame, log = clean_table(frame, profile_table(frame))
    assert len(clean_frame) == 4  # one exact duplicate row dropped
    assert set(clean_frame["city"].unique()) == {"Boston", "Denver"}
    # median of the remaining rain values {10, 30, 50} fills the gap
    assert clean_frame["rain_mm"].isna().sum() == 0
    assert 30.0 in clean_frame["rain_mm"].tolist()
    # the date text was parsed into real dates
    assert pd.api.types.is_datetime64_any_dtype(clean_frame["date"])
    steps = _steps(log)
    assert "coerce_to_datetime" in steps
    assert "fill_missing_median" in steps


def test_cleaning_fills_text_with_mode():
    frame = pd.DataFrame(
        {
            "plan": ["Basic", None, "Pro", "Basic"],
            "months": [1, 2, 3, 4],
        }
    )
    clean_frame, log = clean_table(frame, profile_table(frame))
    assert clean_frame["plan"].isna().sum() == 0
    assert clean_frame["plan"].tolist() == ["Basic", "Basic", "Pro", "Basic"]
    assert "fill_missing_mode" in _steps(log)


def test_cleaning_coerces_number_text():
    frame = pd.DataFrame(
        {
            "price": ["10", "20", "30", "40"],
            "qty": [1, 2, 3, 4],
        }
    )
    clean_frame, log = clean_table(frame, profile_table(frame))
    assert pd.api.types.is_numeric_dtype(clean_frame["price"])
    assert clean_frame["price"].sum() == 100
    assert "coerce_to_numeric" in _steps(log)


def test_cleaning_leaves_clean_table_alone():
    frame = pd.DataFrame({"city": ["Boston", "Denver"], "rain_mm": [1.0, 2.0]})
    clean_frame, log = clean_table(frame, profile_table(frame))
    assert log == []
    assert clean_frame.equals(frame)


def test_cleaning_on_sample_files():
    for path, expected_rows in ((SALES, 240), (CHURN, 300)):
        raw = load_table(path)
        clean_frame, log = clean_table(raw, profile_table(raw))
        assert len(clean_frame) == expected_rows
        assert int(clean_frame.duplicated().sum()) == 0
        assert "drop_duplicate_rows" in _steps(log)
    # every churn column is a fillable type, so nothing stays missing
    raw = load_table(CHURN)
    clean_frame, _ = clean_table(raw, profile_table(raw))
    assert int(clean_frame.isna().sum().sum()) == 0
