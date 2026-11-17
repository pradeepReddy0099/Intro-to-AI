"""Run the stages in order: profile, clean, analyze, chart, predict, report."""
from __future__ import annotations

from pathlib import Path

from agent.analysis import analyze_table
from agent.cleaning import clean_table
from agent.config import AgentConfig
from agent.guardrails import RefusalError, validate_frame
from agent.io_utils import load_table
from agent.prediction import run_prediction
from agent.profiling import profile_table
from agent.questions import parse_question
from agent.reporting import build_refusal_report, build_report
from agent.visualization import make_charts


def run_pipeline(
    input_path: str | Path,
    question: str | None = None,
    target: str | None = None,
    out_dir: str | Path = "output",
    seed: int = 42,
) -> dict:
    """Run the full pipeline and return a results dictionary."""
    config = AgentConfig(random_seed=seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        raw_frame = load_table(input_path)
        validate_frame(raw_frame, str(input_path))
        profile = profile_table(raw_frame)
        clean_frame, clean_log = clean_table(raw_frame, profile)
        analysis = analyze_table(clean_frame, profile, config.max_categories_for_grouping)
        charts = make_charts(
            clean_frame,
            profile,
            analysis,
            out_path / "charts",
            config.max_histogram_columns,
            config.max_bar_columns,
        )
        question_info = parse_question(question, list(clean_frame.columns))
    except RefusalError as exc:
        build_refusal_report(exc.reason, out_path)
        return {
            "status": "refused",
            "input_file": str(input_path),
            "reason": exc.reason,
            "config": config.to_dict(),
            "report": {"html": str(out_path / "report.html")},
        }

    prediction = None
    prediction_refusal = ""
    prediction_target = target or question_info.get("target_column")
    if prediction_target:
        try:
            prediction = run_prediction(clean_frame, prediction_target, config)
        except RefusalError as exc:
            prediction_refusal = exc.reason
    elif question_info["intent"] == "predict":
        prediction_refusal = (
            "Prediction was requested but no target column was named. "
            "Re-run with --target <column>."
        )
    else:
        prediction_refusal = "No target column was given, so prediction was skipped."

    results = {
        "status": "complete",
        "input_file": str(input_path),
        "config": config.to_dict(),
        "question_info": question_info,
        "profile": profile,
        "clean_log": clean_log,
        "analysis": analysis,
        "charts": charts,
        "prediction": prediction,
        "prediction_refusal": prediction_refusal,
        "cleaned_rows": int(len(clean_frame)),
    }
    results["report"] = build_report(results, out_path)
    return results
