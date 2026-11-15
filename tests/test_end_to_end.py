"""End-to-end tests: full pipeline, run on several different datasets."""
from pathlib import Path

import pandas as pd

from agent.pipeline import run_pipeline

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd() / "tests"
SALES = HERE.parent / "data" / "sample_sales.csv"
CHURN = HERE.parent / "data" / "sample_churn.csv"


def weather_frame():
    """A small weather table: dates, a city, numbers, a yes/no storm flag."""
    rows = []
    for i in range(60):
        rain = (i * 7) % 30
        rows.append(
            {
                "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                "city": ["Boston", "Denver", "Miami"][i % 3],
                "temp_c": 5 + (i % 20),
                "rainfall_mm": rain,
                "storm": 1 if rain > 20 else 0,
            }
        )
    return pd.DataFrame(rows)


def student_frame():
    """A small school table: an ID column and a numeric score to predict."""
    rows = []
    for i in range(50):
        hours = (i % 10) + 1
        rows.append(
            {
                "student_id": 1000 + i,
                "subject": ["Math", "Art"][i % 2],
                "study_hours": hours,
                "score": 40 + hours * 5 + (i % 3),
            }
        )
    return pd.DataFrame(rows)


def save(frame, folder, name):
    path = folder / name
    frame.to_csv(path, index=False)
    return path


def test_full_report_run(tmp_path):
    results = run_pipeline(SALES, out_dir=tmp_path / "full")
    assert results["status"] == "complete"
    assert results["prediction"] is None  # no target given
    assert (tmp_path / "full" / "report.html").exists()
    assert (tmp_path / "full" / "report.md").exists()
    assert (tmp_path / "full" / "config.json").exists()
    assert (tmp_path / "full" / "transformation_log.json").exists()
    charts = list((tmp_path / "full" / "charts").glob("*.png"))
    assert len(charts) >= 3  # histograms, bars, heatmap, trend
    assert results["profile"]["duplicate_rows"] == 2  # built into the sample
    assert results["profile"]["row_count"] == 242


def test_question_and_prediction_run(tmp_path):
    results = run_pipeline(
        SALES,
        question="predict churn",
        target="churn",
        out_dir=tmp_path / "churn",
    )
    assert results["status"] == "complete"
    assert results["question_info"]["intent"] == "predict"
    prediction = results["prediction"]
    assert prediction is not None
    assert prediction["task_type"] == "classification"
    assert prediction["best_model"] in {m["model"] for m in prediction["models"]}
    assert (tmp_path / "churn" / "report.html").exists()


def test_churn_sample_run(tmp_path):
    results = run_pipeline(
        CHURN,
        question="predict churn",
        target="churn",
        out_dir=tmp_path / "churn_sample",
    )
    assert results["status"] == "complete"
    assert results["profile"]["row_count"] == 302
    prediction = results["prediction"]
    assert prediction is not None
    assert prediction["task_type"] == "classification"
    assert (tmp_path / "churn_sample" / "report.md").exists()


def test_weather_table_run(tmp_path):
    path = save(weather_frame(), tmp_path, "weather.csv")
    results = run_pipeline(
        path, question="predict storm", target="storm", out_dir=tmp_path / "weather"
    )
    assert results["status"] == "complete"
    prediction = results["prediction"]
    assert prediction is not None
    assert prediction["task_type"] == "classification"
    assert (tmp_path / "weather" / "report.html").exists()


def test_student_scores_are_regression(tmp_path):
    path = save(student_frame(), tmp_path, "students.csv")
    results = run_pipeline(
        path, question="predict score", target="score", out_dir=tmp_path / "students"
    )
    assert results["status"] == "complete"
    prediction = results["prediction"]
    assert prediction is not None
    assert prediction["task_type"] == "regression"
    assert prediction["metric"] == "r2"
    # the ID column is a row label, so it gets no chart of its own
    assert not list((tmp_path / "students" / "charts").glob("*student_id*"))


def test_semicolon_csv_loads(tmp_path):
    path = tmp_path / "semi.csv"
    student_frame().to_csv(path, index=False, sep=";")
    results = run_pipeline(path, out_dir=tmp_path / "semi")
    assert results["status"] == "complete"
    assert len(results["profile"]["columns"]) == 4  # split into real columns


def test_repeat_runs_are_identical(tmp_path):
    first = run_pipeline(SALES, out_dir=tmp_path / "run1")
    second = run_pipeline(SALES, out_dir=tmp_path / "run2")
    assert first["status"] == "complete"
    assert second["status"] == "complete"
    first_html = (tmp_path / "run1" / "report.html").read_bytes()
    second_html = (tmp_path / "run2" / "report.html").read_bytes()
    assert first_html == second_html
    assert (tmp_path / "run1" / "report.md").read_bytes() == (
        tmp_path / "run2" / "report.md"
    ).read_bytes()
    assert first["prediction"] is None and second["prediction"] is None
