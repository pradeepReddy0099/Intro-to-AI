"""Universal tests: the same checks run on ANY dataset, not a fixed one.

Datasets come from three places:

1. Files you drop in tests/user_data/ (any .csv, .tsv, .xlsx, or .xls).
2. The AGENT_TEST_DATA environment variable: a file, or a folder of files,
   e.g.  AGENT_TEST_DATA=/path/to/mydata python -m pytest tests
3. Random tables generated on the fly, so these checks always run even
   when no user files are present.

Only properties that must hold for every valid table are checked — never
values that belong to one particular dataset.
"""
import os
import random
from pathlib import Path

import pandas as pd
import pytest

from agent.cleaning import clean_table
from agent.guardrails import RefusalError
from agent.io_utils import load_table
from agent.pipeline import run_pipeline
from agent.profiling import profile_table

HERE = Path(__file__).resolve().parent
USER_DATA_DIR = HERE / "user_data"
EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}


def user_files():
    """Collect dataset files the user provided, from the folder or env var."""
    files = []
    if USER_DATA_DIR.is_dir():
        files.extend(
            p for p in sorted(USER_DATA_DIR.iterdir()) if p.suffix.lower() in EXTENSIONS
        )
    env_path = os.environ.get("AGENT_TEST_DATA")
    if env_path:
        path = Path(env_path)
        if path.is_dir():
            files.extend(
                p for p in sorted(path.iterdir()) if p.suffix.lower() in EXTENSIONS
            )
        elif path.is_file():
            files.append(path)
    return files


def random_frame(seed):
    """Build a mixed-type table whose exact values nobody chose in advance."""
    rng = random.Random(seed)
    rows = []
    for i in range(60):
        rows.append(
            {
                "row_id": i + 1,
                "day": f"2025-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
                "group": rng.choice(["Alpha", " alpha ", "ALPHA", "Beta", "beta"]),
                "value": None if rng.random() < 0.1 else round(rng.uniform(0, 100), 2),
                "target": 1 if i % 3 else 0,
            }
        )
    rows.append(dict(rows[5]))  # one exact duplicate row
    return pd.DataFrame(rows)


GENERATED = [random_frame(seed) for seed in (7, 21, 99)]


def load_user_frames():
    frames = []
    for path in user_files():
        try:
            frames.append((path.name, load_table(path)))
        except RefusalError:
            continue  # an unreadable file is skipped, not a failure
    return frames


USER_FRAMES = load_user_frames()
ALL_FRAMES = [(f"generated {i}", frame) for i, frame in enumerate(GENERATED)] + USER_FRAMES


@pytest.mark.parametrize("name,frame", ALL_FRAMES, ids=[name for name, _ in ALL_FRAMES])
def test_profile_matches_the_table(name, frame):
    profile = profile_table(frame)
    assert profile["row_count"] == len(frame)
    assert profile["column_count"] == frame.shape[1]
    assert profile["duplicate_rows"] == int(frame.duplicated().sum())
    for entry in profile["columns"]:
        series = frame[entry["name"]]
        assert entry["missing_count"] == int(series.isna().sum())
        assert entry["unique_count"] == int(series.nunique(dropna=True))


@pytest.mark.parametrize("name,frame", ALL_FRAMES, ids=[name for name, _ in ALL_FRAMES])
def test_cleaning_invariants(name, frame):
    profile = profile_table(frame)
    clean_frame, log = clean_table(frame, profile)
    # duplicates are gone and no other rows were lost
    assert int(clean_frame.duplicated().sum()) == 0
    assert len(clean_frame) == len(frame) - profile["duplicate_rows"]
    # numeric columns end up with no missing values (median fill)
    for entry in profile["columns"]:
        column = entry["name"]
        if entry["inferred_type"] == "numeric" and frame[column].notna().any():
            assert int(clean_frame[column].isna().sum()) == 0
    # every change the cleaner claims is written in the log
    assert all("step" in entry and "detail" in entry for entry in log)


@pytest.mark.parametrize("index", range(len(GENERATED)))
def test_pipeline_on_generated_tables(tmp_path, index):
    frame = GENERATED[index]
    path = tmp_path / f"table{index}.csv"
    frame.to_csv(path, index=False)
    results = run_pipeline(
        path, question="predict target", target="target", out_dir=tmp_path / "run"
    )
    assert results["status"] == "complete"
    assert results["prediction"] is not None
    assert results["prediction"]["task_type"] == "classification"
    assert (tmp_path / "run" / "report.md").exists()
    assert (tmp_path / "run" / "report.html").exists()
    # running it again with the same inputs gives the identical report
    run_pipeline(
        path, question="predict target", target="target", out_dir=tmp_path / "again"
    )
    assert (tmp_path / "run" / "report.md").read_bytes() == (
        tmp_path / "again" / "report.md"
    ).read_bytes()


def test_pipeline_on_user_files(tmp_path):
    files = user_files()
    if not files:
        pytest.skip("no user dataset files in tests/user_data/ or AGENT_TEST_DATA")
    for i, path in enumerate(files):
        results = run_pipeline(path, out_dir=tmp_path / f"user{i}")
        assert results["status"] == "complete", path.name
        assert (tmp_path / f"user{i}" / "report.html").exists(), path.name
