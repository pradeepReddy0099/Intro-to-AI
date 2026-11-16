"""Head-to-head evaluation harness for the data analysis agent.

Implements the evaluation designed in the revised proposal:

- An independent reference implementation computes
  ground truth for a fixed task set on the two sample datasets.
- The agent is run 5 times per dataset. Every task value is extracted
  from the agent's structured results and compared with ground truth, across runs, with wall-clock runtime
  recorded per run.
- Prediction tasks compare the agent's per-model test scores against an
  independently coded run of the same model set under the same protocol
  (80/20 stratified split, seed 42).

Outputs: eval/results/eval_results.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPS = 5
SEED = 42

from agent.pipeline import run_pipeline  # noqa: E402


# ---------------------------------------------------------------- reference
def reference_clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Independent implementation of the documented cleaning rules:
    drop exact duplicates; strip text whitespace; unify case variants of
    a category to its most frequent spelling; 
    fill missing numeric with the median and categorical with the mode.
    Date time gaps are left unfilled."""
    df = frame.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    def is_text(series):
        return (not pd.api.types.is_numeric_dtype(series)
                and not pd.api.types.is_datetime64_any_dtype(series))

    text_cols = [c for c in df.columns if is_text(df[c])]
    for col in text_cols:
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    for col in text_cols:
        lowered = df[col].map(lambda v: v.lower() if isinstance(v, str) else v)
        for key, group in df.groupby(lowered, dropna=False):
            if not isinstance(key, str):
                continue
            spellings = group[col].dropna()
            if spellings.empty:
                continue
            df.loc[group.index, col] = spellings.mode().iloc[0]

    for col in df.columns:
        if not is_text(df[col]):
            continue
        as_num = pd.to_numeric(df[col], errors="coerce")
        if as_num.notna().mean() > 0.8:
            df[col] = as_num
            continue
        as_date = pd.to_datetime(df[col], errors="coerce", format="mixed")
        if as_date.notna().mean() > 0.8:
            df[col] = as_date

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif is_text(df[col]):
            modes = df[col].mode(dropna=True)
            if len(modes):
                df[col] = df[col].fillna(modes.iloc[0])
    return df


def reference_prediction(frame: pd.DataFrame, target: str) -> dict:
    """Same protocol as the agent, coded independently: drop datetime and
    >50-category features, median/mode imputation, one-hot encoding,
    scaling for logistic regression, 80/20 stratified split, seed 42."""
    y = frame[target]
    X = frame.drop(columns=[target])
    keep = []
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            continue
        if (not pd.api.types.is_numeric_dtype(X[col])
                and X[col].nunique(dropna=True) > 50):
            continue
        keep.append(col)
    X = X[keep]
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    def build(model, scale):
        num_steps = [("fill", SimpleImputer(strategy="median"))]
        if scale:
            num_steps.append(("scale", StandardScaler()))
        prep = ColumnTransformer([
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline([
                ("fill", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ])
        return Pipeline([("prep", prep), ("model", model)])

    models = {
        "Dummy (baseline)": build(DummyClassifier(strategy="most_frequent"), False),
        "Logistic Regression": build(
            LogisticRegression(max_iter=1000, random_state=SEED), True),
        "Random Forest": build(
            RandomForestClassifier(n_estimators=300, random_state=SEED), False),
        "Gradient Boosting": build(
            GradientBoostingClassifier(random_state=SEED), False),
    }
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    scores = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        scores[name] = round(float(pipe.score(X_test, y_test)), 4)
    return scores


# ---------------------------------------------------------------- extraction
def desc_entry(analysis, column):
    for entry in analysis["descriptive"]:
        if entry["column"] == column:
            return entry
    return None


def corr_value(analysis, col_a, col_b):
    for pair in analysis["correlations"]["pairs"]:
        if {pair["column_a"], pair["column_b"]} == {col_a, col_b}:
            return pair["correlation"]
    return None


def group_entry(analysis, group_col, value_col):
    for entry in analysis["group_summaries"]:
        if entry["group_column"] == group_col and entry["value_column"] == value_col:
            return entry
    return None


def trend_value(analysis, period, column):
    for row in analysis["trends"]["series"]:
        if row["period"] == period:
            return row[column]
    return None


def month_end(day: pd.Timestamp) -> str:
    return str((day + pd.offsets.MonthEnd(0)).date())


# ---------------------------------------------------------------- tasks
def build_tasks():
    """Each task: id, dataset key, description, tolerance, and functions to
    pull the value from (agent results) and from (reference frame)."""
    sales_ref = reference_clean(pd.read_csv(BASE / "data/sample_sales.csv"))
    churn_ref = reference_clean(pd.read_csv(BASE / "data/sample_churn.csv"))
    refs = {"sales": sales_ref, "churn": churn_ref}

    sales_first_period = month_end(sales_ref["order_date"].min())
    sales_rev_by_region = sales_ref.groupby("region")["revenue"].mean()
    churn_by_plan = churn_ref.groupby("plan_type")["churn"].mean()
    sales_monthly = sales_ref.set_index("order_date").resample("ME")["revenue"].mean()

    tasks = [
        dict(id="S1", dataset="sales", tol=0,
             desc="Cleaned row count (sales)",
             agent=lambda r: desc_entry(r["analysis"], "revenue")["count"],
             ref=lambda f: len(f)),
        dict(id="S2", dataset="sales", tol=0.01,
             desc="Mean revenue (sales)",
             agent=lambda r: desc_entry(r["analysis"], "revenue")["mean"],
             ref=lambda f: round(float(f["revenue"].mean()), 4)),
        dict(id="S3", dataset="sales", tol=0.01,
             desc="Mean units sold (sales)",
             agent=lambda r: desc_entry(r["analysis"], "units_sold")["mean"],
             ref=lambda f: round(float(f["units_sold"].mean()), 4)),
        dict(id="S4", dataset="sales", tol=0.001,
             desc="Correlation units_sold vs revenue (sales)",
             agent=lambda r: corr_value(r["analysis"], "units_sold", "revenue"),
             ref=lambda f: round(float(f["units_sold"].corr(f["revenue"])), 4)),
        dict(id="S5", dataset="sales", tol=0.01,
             desc="Region with highest mean revenue (sales)",
             agent=lambda r: (lambda e: (e["highest_group"], e["highest_mean"]))(
                 group_entry(r["analysis"], "region", "revenue")),
             ref=lambda f: (str(sales_rev_by_region.idxmax()),
                            round(float(sales_rev_by_region.max()), 4))),
        dict(id="S6", dataset="sales", tol=0.01,
             desc=f"Mean revenue in first month {sales_first_period} (sales trend)",
             agent=lambda r: trend_value(r["analysis"], sales_first_period, "revenue"),
             ref=lambda f: round(float(sales_monthly.iloc[0]), 4)),
        dict(id="S7", dataset="sales", tol=0,
             desc="Most common product category (sales)",
             agent=lambda r: desc_entry(r["analysis"], "product_category")["most_common"],
             ref=lambda f: str(f["product_category"].mode().iloc[0])),
        dict(id="C1", dataset="churn", tol=0,
             desc="Cleaned row count (churn)",
             agent=lambda r: desc_entry(r["analysis"], "churn")["count"],
             ref=lambda f: len(f)),
        dict(id="C2", dataset="churn", tol=0.001,
             desc="Overall churn rate (churn)",
             agent=lambda r: desc_entry(r["analysis"], "churn")["mean"],
             ref=lambda f: round(float(f["churn"].mean()), 4)),
        dict(id="C3", dataset="churn", tol=0.01,
             desc="Mean tenure months (churn)",
             agent=lambda r: desc_entry(r["analysis"], "tenure_months")["mean"],
             ref=lambda f: round(float(f["tenure_months"].mean()), 4)),
        dict(id="C4", dataset="churn", tol=0.001,
             desc="Correlation monthly_spend vs total_spend (churn)",
             agent=lambda r: corr_value(r["analysis"], "monthly_spend", "total_spend"),
             ref=lambda f: round(float(f["monthly_spend"].corr(f["total_spend"])), 4)),
        dict(id="C5", dataset="churn", tol=0.001,
             desc="Plan type with highest churn rate (churn)",
             agent=lambda r: (lambda e: (e["highest_group"], e["highest_mean"]))(
                 group_entry(r["analysis"], "plan_type", "churn")),
             ref=lambda f: (str(churn_by_plan.idxmax()),
                            round(float(churn_by_plan.max()), 4))),
    ]
    return tasks, refs


def values_match(agent_value, ref_value, tol):
    if isinstance(ref_value, tuple):
        return (agent_value[0] == ref_value[0]
                and abs(agent_value[1] - ref_value[1]) <= max(tol, 0.011))
    if isinstance(ref_value, str):
        return agent_value == ref_value
    return abs(agent_value - ref_value) <= max(tol, 0.00011)


# ---------------------------------------------------------------- main
def main():
    tasks, refs = build_tasks()
    ref_predictions = {
        name: reference_prediction(frame, "churn") for name, frame in refs.items()
    }
    ground_truth = {}
    for task in tasks:
        ground_truth[task["id"]] = task["ref"](refs[task["dataset"]])

    runs = {"sales": [], "churn": []}
    report_hashes = {"sales": [], "churn": []}
    runtimes = {"sales": [], "churn": []}
    file_for = {"sales": "data/sample_sales.csv", "churn": "data/sample_churn.csv"}

    for rep in range(REPS):
        for name in ("sales", "churn"):
            out_dir = RESULTS_DIR / "runs" / name / f"rep{rep + 1}"
            start = time.perf_counter()
            result = run_pipeline(BASE / file_for[name], target="churn",
                                  out_dir=out_dir, seed=SEED)
            runtimes[name].append(round(time.perf_counter() - start, 2))
            report_hashes[name].append(
                hashlib.md5((out_dir / "report.md").read_bytes()).hexdigest())
            runs[name].append(result)
        print(f"rep {rep + 1}/{REPS} done", flush=True)

    task_rows = []
    for task in tasks:
        seen = [task["agent"](runs[task["dataset"]][rep]) for rep in range(REPS)]
        truth = ground_truth[task["id"]]
        task_rows.append({
            "id": task["id"], "dataset": task["dataset"], "desc": task["desc"],
            "ground_truth": truth,
            "agent_values": seen,
            "reproducible": len({json.dumps(v, default=str) for v in seen}) == 1,
            "correct": all(values_match(v, truth, task["tol"]) for v in seen),
        })

    pred_rows = []
    for name in ("sales", "churn"):
        agent_models = runs[name][0]["prediction"]["models"]
        row = {"dataset": name,
               "agent_best_model": runs[name][0]["prediction"]["best_model"],
               "agent_best_test": runs[name][0]["prediction"]["best_test_score"],
               "per_model": []}
        for model in agent_models:
            ref_score = ref_predictions[name][model["model"]]
            row["per_model"].append({
                "model": model["model"],
                "agent_test": model["test_score"],
                "reference_test": ref_score,
                "match": abs(model["test_score"] - ref_score) <= 0.021,
            })
        row["reproducible"] = len({
            json.dumps(r["prediction"]["models"], default=str) for r in runs[name]
        }) == 1
        pred_rows.append(row)

    summary = {
        "reps": REPS,
        "tasks": task_rows,
        "predictions": pred_rows,
        "report_md5_identical": {
            name: len(set(hashes)) == 1 for name, hashes in report_hashes.items()
        },
        "mean_runtime_seconds": {
            name: round(float(np.mean(times)), 2) for name, times in runtimes.items()
        },
        "runtimes": runtimes,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "eval_results.json").write_text(json.dumps(summary, indent=2, default=str))

    ok = sum(t["correct"] for t in task_rows)
    repro = sum(t["reproducible"] for t in task_rows)
    print(f"\nDescriptive tasks: {ok}/{len(task_rows)} correct, "
          f"{repro}/{len(task_rows)} identical across {REPS} runs")
    for row in pred_rows:
        matched = sum(m["match"] for m in row["per_model"])
        print(f"Prediction ({row['dataset']}): best={row['agent_best_model']} "
              f"test={row['agent_best_test']}, per-model matches {matched}/"
              f"{len(row['per_model'])}, reproducible={row['reproducible']}")
    print("Report byte-identical across runs:", summary["report_md5_identical"])
    print("Mean runtime per full run:", summary["mean_runtime_seconds"])


if __name__ == "__main__":
    main()
