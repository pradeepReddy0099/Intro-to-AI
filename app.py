"""Browser user interface for the data analysis agent.

Run it with:  streamlit run app.py

Upload a dataset (or pick one from data/), choose a question and an
optional target column, and run the same pipeline the command line runs.
Every run is kept under output/ui_runs/, so previous results can be
opened again from the history list.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

from agent.guardrails import RefusalError
from agent.io_utils import load_table
from agent.pipeline import run_pipeline
from agent.reporting import direct_answer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
USER_DATA_DIR = ROOT / "tests" / "user_data"
RUNS_DIR = ROOT / "output" / "ui_runs"
EXTENSIONS = ["csv", "tsv", "xlsx", "xls"]

# The basic features the program offers, as (dropdown label, question).
# Picking one sets the question the pipeline answers; an empty question
# means the full report with no specific question.
FEATURES = [
    ("Full report (profile, cleaning, statistics, correlations, groups, trends, charts)", ""),
    ("Summarize the data", "summarize the data"),
    ("Compare groups", "compare the groups"),
    ("Find correlations between columns", "what correlates with the other columns?"),
    ("Show the trend over time", "how does it trend over time?"),
    ("Predict a target column", "predict the target"),
]


def place_file(name: str, data: bytes, folder: Path) -> Path:
    """Save uploaded bytes into a folder, never overwriting a different file."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    if target.exists():
        if target.read_bytes() == data:
            return target
        stem, suffix = target.stem, target.suffix
        count = 2
        while (folder / f"{stem}_{count}{suffix}").exists():
            count += 1
        target = folder / f"{stem}_{count}{suffix}"
    target.write_bytes(data)
    return target


def save_upload(uploaded) -> Path:
    """Copy an uploaded file into data/ and tests/user_data/; return its path."""
    data = uploaded.getvalue()
    saved = place_file(uploaded.name, data, DATA_DIR)
    place_file(uploaded.name, data, USER_DATA_DIR)
    return saved


def data_files() -> list[Path]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(
        p for p in DATA_DIR.iterdir() if p.suffix.lower() in {".csv", ".tsv", ".xlsx", ".xls"}
    )


def columns_of(path: Path) -> list[str]:
    try:
        return [str(c) for c in load_table(path).columns]
    except RefusalError:
        return []


def save_summary(folder: Path, results: dict, dataset: str, question, target) -> dict:
    prediction = results.get("prediction")
    summary = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": dataset,
        "question": question or "",
        "target": target or "",
        "status": results["status"],
        "reason": results.get("reason", ""),
        "answer": direct_answer(results) if results["status"] == "complete" else "",
        "cleaned_rows": results.get("cleaned_rows"),
        "prediction": (
            {
                "task_type": prediction["task_type"],
                "target": prediction["target"],
                "metric": prediction["metric"],
                "best_model": prediction["best_model"],
                "best_test_score": prediction["best_test_score"],
                "models": prediction["models"],
            }
            if prediction
            else None
        ),
        "prediction_refusal": results.get("prediction_refusal", ""),
        "charts": [chart["file"] for chart in results.get("charts", [])],
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def show_results(folder: Path, summary: dict) -> None:
    st.subheader(f"Results — {summary['dataset']}")
    st.caption(f"Run at {summary['created']}")
    if summary["status"] == "refused":
        st.error(f"Refused: {summary['reason']}")
        return
    st.success(summary["answer"])
    st.write(f"Rows after cleaning: {summary['cleaned_rows']}")
    prediction = summary.get("prediction")
    if prediction:
        st.write(
            f"**Prediction:** {prediction['task_type']} on `{prediction['target']}` — "
            f"best model **{prediction['best_model']}** "
            f"(test {prediction['metric']} {prediction['best_test_score']:.3f})"
        )
        st.dataframe(prediction["models"], hide_index=True)
    elif summary.get("prediction_refusal"):
        st.info(summary["prediction_refusal"])
    charts = sorted((folder / "charts").glob("*.png"))
    if charts:
        st.write("**Charts**")
        columns = st.columns(2)
        for i, chart in enumerate(charts):
            columns[i % 2].image(str(chart), caption=chart.stem.replace("_", " "))
    report_html = folder / "report.html"
    report_md = folder / "report.md"
    if report_html.exists():
        st.download_button(
            "Download the HTML report", report_html.read_bytes(), "report.html", "text/html"
        )
    if report_md.exists():
        with st.expander("Read the text report"):
            st.markdown(report_md.read_text(encoding="utf-8"))


def past_runs() -> list[Path]:
    if not RUNS_DIR.is_dir():
        return []
    folders = [p for p in RUNS_DIR.iterdir() if (p / "summary.json").exists()]
    return sorted(folders, key=lambda p: p.name, reverse=True)


st.set_page_config(page_title="AI-Powered Local Analysis Agent", layout="wide")
st.title("AI-Powered Local Analysis Agent")
st.write(
    "Upload a dataset, ask a question, and get a report with charts and a "
    "model comparison. Everything runs on this machine."
)

st.header("1. Your dataset")
uploaded = st.file_uploader(
    "Upload a data file", type=EXTENSIONS, help="CSV, TSV, or Excel."
)
dataset_path = None
if uploaded is not None:
    marker = (uploaded.name, uploaded.size)
    if st.session_state.get("upload_marker") != marker:
        st.session_state["upload_marker"] = marker
        st.session_state["upload_path"] = str(save_upload(uploaded))
    dataset_path = Path(st.session_state["upload_path"])
    st.info(
        f"Using **{dataset_path.name}** — a copy was saved into data/ and "
        "tests/user_data/, so the tests will check it too."
    )
else:
    choices = data_files()
    if choices:
        picked = st.selectbox(
            "…or pick a dataset already in data/",
            choices,
            format_func=lambda p: p.name,
        )
        dataset_path = picked

st.header("2. Choose a feature and run")
feature = st.selectbox("What do you want to do?", [label for label, _ in FEATURES])
feature_question = dict(FEATURES)[feature]
typed = st.text_input(
    "…or type your own question instead",
    placeholder="e.g. what correlates with revenue?",
)
question = typed or feature_question
wants_predict = feature.startswith("Predict")
target = None
if dataset_path is not None:
    columns = columns_of(dataset_path)
    target_label = "Target column to predict" + (
        " — needed for this feature" if wants_predict else " (optional)"
    )
    target_choice = st.selectbox(target_label, ["(none)"] + columns)
    if target_choice != "(none)":
        target = target_choice

run_clicked = st.button(
    "Run the analysis", type="primary", disabled=dataset_path is None
)
if run_clicked and wants_predict and target is None:
    st.warning("This feature needs a target: pick the column to predict above.")
    run_clicked = False
if run_clicked:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = RUNS_DIR / f"{stamp}_{dataset_path.stem}"
    count = 2
    while folder.exists():
        folder = RUNS_DIR / f"{stamp}_{dataset_path.stem}_{count}"
        count += 1
    with st.spinner("Running…"):
        results = run_pipeline(
            dataset_path, question=question or None, target=target, out_dir=folder
        )
        summary = save_summary(folder, results, dataset_path.name, question, target)
    st.session_state["current"] = (str(folder), summary)

if "current" in st.session_state:
    folder, summary = st.session_state["current"]
    show_results(Path(folder), summary)

st.header("3. Previous results")
runs = past_runs()
if not runs:
    st.write("No runs yet — your runs will be listed here.")
else:
    def label(folder: Path) -> str:
        info = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        return f"{info['created']} — {info['dataset']} ({info['status']})"

    chosen = st.selectbox("Open an earlier run", runs, format_func=label)
    if st.button("Open this run"):
        info = json.loads((chosen / "summary.json").read_text(encoding="utf-8"))
        st.session_state["current"] = (str(chosen), info)
        st.rerun()

st.header("4. Tests")
st.write("Run the project's test suite, including any dataset you uploaded.")
if st.button("Run the tests"):
    with st.spinner("Testing… this takes about half a minute."):
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    tail = "\n".join((done.stdout + done.stderr).splitlines()[-8:])
    if done.returncode == 0:
        st.success("All tests passed.")
    else:
        st.error("Some tests failed — the last lines are below.")
    st.code(tail)
