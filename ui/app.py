"""Browser user interface for the data analysis agent.

Run it with:  streamlit run app.py

Upload a dataset (or pick one from data/), choose a question and an
optional target column, and run the same pipeline the command line runs.
Every run is kept under output/ui_runs/, so previous results can be
opened again from the history list in the sidebar.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from agent.guardrails import RefusalError
from agent.insights import build_insights
from agent.io_utils import load_table
from agent.pipeline import run_pipeline
from agent.reporting import direct_answer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
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
    """Save an uploaded file into data/; return its path."""
    return place_file(uploaded.name, uploaded.getvalue(), DATA_DIR)


def data_files() -> list[Path]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(
        p for p in DATA_DIR.iterdir() if p.suffix.lower() in {".csv", ".tsv", ".xlsx", ".xls"}
    )


@st.cache_data(show_spinner=False)
def preview_of(path_text: str, stamp: float):
    """First rows, shape and column names of a dataset file (for the preview)."""
    try:
        frame = load_table(Path(path_text))
    except RefusalError:
        return None
    return frame.head(6), len(frame), len(frame.columns), [str(c) for c in frame.columns]


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
        "insights": build_insights(results),
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
    st.subheader(f"✅ Results — {summary['dataset']}")
    st.caption(f"Run at {summary['created']}")
    if summary["status"] == "refused":
        st.error(f"Refused: {summary['reason']}")
        return
    tab_overview, tab_charts, tab_report = st.tabs(
        ["🎯 Overview", "📊 Charts", "📄 Report"]
    )
    with tab_overview:
        st.markdown(
            f'<div class="answer">🎯 {summary["answer"]}</div>',
            unsafe_allow_html=True,
        )
        insights = summary.get("insights") or {}
        if insights.get("insights") or insights.get("decisions"):
            with st.container(border=True):
                # Both blocks are the same shape — a bold heading, then one
                # bullet per item — so one loop renders them both.
                for heading, items in [
                    ("🧭 Key insights — what the data says", insights.get("insights")),
                    ("✅ What to do next", insights.get("decisions")),
                ]:
                    if items:
                        st.markdown(f"**{heading}**")
                        for item in items:
                            st.markdown(f"- {item}")
        prediction = summary.get("prediction")
        cards = st.columns(3 if prediction else 1)
        cards[0].metric("Rows after cleaning", summary["cleaned_rows"])
        if prediction:
            models = prediction["models"]
            best_row = next(m for m in models if m["model"] == prediction["best_model"])
            base_row = models[0]
            cards[1].metric(
                "Best model",
                prediction["best_model"],
                delta=f"CV {best_row['cv_mean']:.3f}",
            )
            cards[2].metric(
                f"Test {prediction['metric']}",
                f"{prediction['best_test_score']:.3f}",
                delta=(
                    f"{prediction['best_test_score'] - base_row['test_score']:+.3f} "
                    f"vs {base_row['model']}"
                ),
            )
            st.caption(
                f"Prediction: {prediction['task_type']} on `{prediction['target']}` "
                "— the winning row is highlighted below."
            )
            table = pd.DataFrame(models)
            styled = table.style.apply(
                lambda row: [
                    "background-color: #EDE9FE; font-weight: 600"
                    if row["model"] == prediction["best_model"]
                    else ""
                    for _ in row
                ],
                axis=1,
            ).hide(axis="index")
            st.dataframe(styled)
        elif summary.get("prediction_refusal"):
            st.info(summary["prediction_refusal"])
    with tab_charts:
        charts = sorted((folder / "charts").glob("*.png"))
        if charts:
            columns = st.columns(2)
            for i, chart in enumerate(charts):
                with columns[i % 2].container(border=True):
                    st.markdown(f"**{chart.stem.replace('_', ' ').capitalize()}**")
                    st.image(str(chart))
        else:
            st.info("No charts were drawn for this run.")
    with tab_report:
        report_html = folder / "report.html"
        report_md = folder / "report.md"
        buttons = st.columns(2)
        if report_html.exists():
            buttons[0].download_button(
                "⬇️ Download the HTML report",
                report_html.read_bytes(),
                "report.html",
                "text/html",
                type="primary",
            )
        if report_md.exists():
            buttons[1].download_button(
                "⬇️ Download the text report (.md)",
                report_md.read_bytes(),
                "report.md",
                "text/markdown",
            )
            with st.expander("Read the text report"):
                st.markdown(report_md.read_text(encoding="utf-8"))


def load_summary(folder: Path) -> dict:
    """Read back a run's summary.json (the same file save_summary wrote)."""
    return json.loads((folder / "summary.json").read_text(encoding="utf-8"))


def past_runs() -> list[Path]:
    if not RUNS_DIR.is_dir():
        return []
    folders = [p for p in RUNS_DIR.iterdir() if (p / "summary.json").exists()]
    return sorted(folders, key=lambda p: p.name, reverse=True)


st.set_page_config(
    page_title="AI-Powered Local Analysis Agent", page_icon="📊", layout="wide"
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #EEF0FF 0%, #F7F7FC 240px); }
    #MainMenu, footer, .stAppDeployButton { display: none !important; }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1100px; padding-top: 2rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EEF0FF 0%, #F7F7FC 320px);
        border-right: 1px solid #E4E2F5;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes popIn {
        from { opacity: 0; transform: scale(0.96); }
        to { opacity: 1; transform: scale(1); }
    }
    .hero {
        background: linear-gradient(120deg, #312E81, #6D28D9, #2563EB, #6D28D9, #312E81);
        background-size: 220% 220%;
        animation: gradientShift 14s ease infinite, fadeUp 0.7s ease both;
        color: white; border-radius: 18px; padding: 30px 36px;
        box-shadow: 0 12px 30px rgba(49, 46, 129, 0.28); margin-bottom: 8px;
    }
    .hero h1 { color: white; font-size: 2.1rem; margin: 0 0 6px 0; padding: 0; }
    .hero p { color: #DDD6FE; font-size: 1.02rem; margin: 0; }
    .answer {
        background: linear-gradient(120deg, #EDE9FE, #DBEAFE);
        border-left: 6px solid #6D28D9; border-radius: 12px;
        padding: 16px 20px; font-size: 1.05rem; font-weight: 600;
        color: #312E81; animation: popIn 0.6s ease both; margin-bottom: 12px;
    }
    h2 {
        color: #312E81; border-bottom: 3px solid #C4B5FD;
        padding-bottom: 6px; margin-top: 2rem; animation: fadeUp 0.6s ease both;
    }
    h3 { animation: fadeUp 0.6s ease both; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px; font-weight: 600; padding: 0.45rem 1.2rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px); }
    button[kind="primary"] {
        background: linear-gradient(120deg, #6D28D9, #2563EB);
        border: none; box-shadow: 0 6px 16px rgba(109, 40, 217, 0.35);
    }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #E4E2F5; border-radius: 14px;
        padding: 14px 18px; box-shadow: 0 3px 10px rgba(49, 46, 129, 0.06);
        animation: fadeUp 0.6s ease both;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px); box-shadow: 0 10px 22px rgba(49, 46, 129, 0.14);
    }
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; animation: fadeUp 0.6s ease both; }
    [data-testid="stAlert"] { border-radius: 12px; animation: fadeUp 0.5s ease both; }
    [data-testid="stImage"] img { animation: popIn 0.7s ease both; border-radius: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0; padding: 8px 20px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { color: #6D28D9; }
    </style>
    <div class="hero">
        <h1>📊 AI-Powered Local Analysis Agent</h1>
        <p>Upload a dataset, ask a question, and get a report with charts and a
        model comparison — everything runs on this machine. No LLM, no network,
        no cost.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🧭 How it works")
    st.markdown(
        "1. Pick or upload a dataset\n"
        "2. Choose a feature — and a target column if you want a prediction\n"
        "3. Run: you get key insights, charts and a model comparison"
    )
    st.divider()
    st.markdown("### 🕘 Previous results")
    runs = past_runs()
    if not runs:
        st.write("No runs yet — your runs will be listed here.")
    else:
        def run_label(folder: Path) -> str:
            info = load_summary(folder)
            return f"{info['created']} — {info['dataset']} ({info['status']})"

        chosen = st.selectbox("Open an earlier run", runs, format_func=run_label)
        if st.button("Open this run"):
            info = load_summary(chosen)
            st.session_state["current"] = (str(chosen), info)
            st.rerun()
    st.divider()
    st.caption(
        "Runs 100% locally · No LLM · No network · "
        "Same input, same report (seed 42)"
    )

st.header("📂 1. Your dataset")
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
    st.info(f"Using **{dataset_path.name}** — a copy was saved into data/.")
else:
    choices = data_files()
    if choices:
        picked = st.selectbox(
            "…or pick a dataset already in data/",
            choices,
            format_func=lambda p: p.name,
        )
        dataset_path = picked

if dataset_path is not None:
    preview = preview_of(str(dataset_path), dataset_path.stat().st_mtime)
    if preview is not None:
        head, row_count, col_count, col_names = preview
        with st.container(border=True):
            st.markdown("**👀 Preview — the first rows of your data**")
            chips = st.columns(3)
            chips[0].metric("Rows", row_count)
            chips[1].metric("Columns", col_count)
            chips[2].metric("File size", f"{dataset_path.stat().st_size / 1024:.1f} KB")
            st.dataframe(head, hide_index=True)
            st.caption("Columns: " + ", ".join(col_names))

st.header("🎯 2. Choose a feature and run")
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
    # The preview above already loaded this file (cached), so reuse its
    # column list instead of loading the table a second time per rerun.
    columns = preview[3] if preview is not None else []
    target_label = "Target column to predict" + (
        " — needed for this feature" if wants_predict else " (optional)"
    )
    target_choice = st.selectbox(target_label, ["(none)"] + columns)
    if target_choice != "(none)":
        target = target_choice

if dataset_path is not None:
    st.caption(
        f"Ready when you are: **{dataset_path.name}** · {feature} · "
        f"Target: **{target or '—'}**"
    )
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
    with st.status("Running the analysis…", expanded=True) as status:
        st.write("📂 Profiling the table…")
        st.write("🧹 Cleaning and logging every change…")
        st.write("📈 Computing statistics, correlations, groups and trends…")
        st.write("🎨 Drawing the charts…")
        if target:
            st.write("🤖 Comparing the four models…")
        results = run_pipeline(
            dataset_path, question=question or None, target=target, out_dir=folder
        )
        summary = save_summary(folder, results, dataset_path.name, question, target)
        status.update(label="✅ Analysis complete", state="complete", expanded=False)
    st.session_state["current"] = (str(folder), summary)
    if summary["status"] == "complete":
        st.balloons()

if "current" in st.session_state:
    folder, summary = st.session_state["current"]
    show_results(Path(folder), summary)
