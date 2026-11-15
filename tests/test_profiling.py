"""Tests for profiling, on several different tables."""
from pathlib import Path

import pandas as pd

from agent.io_utils import load_table
from agent.profiling import infer_column_type, profile_table

HERE = Path(__file__).resolve().parent
SALES = HERE.parent / "data" / "sample_sales.csv"
CHURN = HERE.parent / "data" / "sample_churn.csv"


def _sample_frame():
    return pd.DataFrame(
        {
            "sales": [10.0, 20.0, None, 40.0],
            "region": ["North", "South", "North", "South"],
            "order_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "notes": ["a", "b", "c", "d"],
        }
    )


def test_infer_column_type():
    frame = _sample_frame()
    assert infer_column_type(frame["sales"]) == "numeric"
    assert infer_column_type(frame["region"]) == "categorical"
    assert infer_column_type(frame["order_date"]) == "datetime"
    assert infer_column_type(frame["notes"]) == "categorical"  # 4 unique, <= 20


def test_profile_counts():
    frame = _sample_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)  # one duplicate
    profile = profile_table(frame)
    assert profile["row_count"] == 5
    assert profile["duplicate_rows"] == 1
    by_name = {c["name"]: c for c in profile["columns"]}
    assert by_name["sales"]["missing_count"] == 1
    assert by_name["sales"]["mean"] == 20.0  # (10 + 20 + 40 + 10) / 4 non-missing
    assert by_name["region"]["unique_count"] == 2


def test_types_on_other_tables():
    frame = pd.DataFrame(
        {
            "when": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "city": ["Boston", "Denver", "Miami"],
            "storm": [0, 1, 0],
        }
    )
    assert infer_column_type(frame["when"]) == "datetime"
    assert infer_column_type(frame["city"]) == "categorical"
    assert infer_column_type(frame["storm"]) == "numeric"
    # more than 20 different long strings are free text, not a category
    long_frame = pd.DataFrame({"comment": [f"note {i}" for i in range(30)]})
    assert infer_column_type(long_frame["comment"]) == "text"


def test_identifier_flag():
    frame = pd.DataFrame(
        {
            "student_id": range(100, 150),
            "group_id": [1, 2] * 25,
            "score": range(50),
        }
    )
    by_name = {c["name"]: c for c in profile_table(frame)["columns"]}
    assert by_name["student_id"]["is_identifier"] is True
    assert by_name["group_id"]["is_identifier"] is False  # values repeat
    assert by_name["score"]["is_identifier"] is False  # name is not an ID name


def test_missing_rate():
    frame = pd.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": ["x", "y", "z", "w"]})
    by_name = {c["name"]: c for c in profile_table(frame)["columns"]}
    assert by_name["a"]["missing_count"] == 1
    assert by_name["a"]["missing_rate"] == 0.25
    assert by_name["b"]["missing_rate"] == 0.0


def test_profile_on_sample_files():
    sales = profile_table(load_table(SALES))
    assert sales["row_count"] == 242
    assert sales["column_count"] == 11
    assert sales["duplicate_rows"] == 2
    sales_by_name = {c["name"]: c for c in sales["columns"]}
    assert sales_by_name["order_date"]["inferred_type"] == "datetime"
    assert sales_by_name["order_id"]["is_identifier"] is True

    churn = profile_table(load_table(CHURN))
    assert churn["row_count"] == 302
    assert churn["column_count"] == 10
    assert churn["duplicate_rows"] == 2
    churn_by_name = {c["name"]: c for c in churn["columns"]}
    assert churn_by_name["plan_type"]["inferred_type"] == "categorical"
    assert churn_by_name["customer_id"]["is_identifier"] is True
