"""Tests for guardrails: refusals for bad input and unsafe prediction."""
import pandas as pd
import pytest

from agent.config import AgentConfig
from agent.guardrails import RefusalError, check_prediction_ready, validate_frame
from agent.io_utils import load_table
from agent.pipeline import run_pipeline


def test_empty_frame_refused():
    with pytest.raises(RefusalError, match="no data rows"):
        validate_frame(pd.DataFrame(), "empty.csv")


def test_unsupported_file_refused(tmp_path):
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("hello")
    with pytest.raises(RefusalError, match="Unsupported file type"):
        load_table(bad_file)


def test_missing_target_refused():
    frame = pd.DataFrame({"a": range(40), "b": [0, 1] * 20})
    with pytest.raises(RefusalError, match="not in the dataset"):
        check_prediction_ready(frame, "nope", AgentConfig())


def test_too_few_rows_refused():
    frame = pd.DataFrame({"a": range(10), "b": [0, 1] * 5})
    with pytest.raises(RefusalError, match="at least 30 rows"):
        check_prediction_ready(frame, "b", AgentConfig())


def test_constant_target_refused():
    frame = pd.DataFrame({"a": range(40), "b": [1] * 40})
    with pytest.raises(RefusalError, match="only one distinct value"):
        check_prediction_ready(frame, "b", AgentConfig())


def test_rare_class_refused():
    frame = pd.DataFrame({"a": range(40), "b": [0] * 38 + [1] * 2})
    with pytest.raises(RefusalError, match="fewer than 5 examples"):
        check_prediction_ready(frame, "b", AgentConfig())


def test_pipeline_refusal_writes_report(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("a,b\n")
    results = run_pipeline(empty_file, out_dir=tmp_path / "out")
    assert results["status"] == "refused"
    assert (tmp_path / "out" / "report.html").exists()
    assert "no data rows" in results["reason"]
