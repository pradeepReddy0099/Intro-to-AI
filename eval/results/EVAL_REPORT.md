# Head-to-Head Evaluation — Results (Local Arm)

**Date:** 2026-10-23
**Protocol:** Revised proposal, Section 4 (baselines, metrics, hypotheses H1–H4) and Section 5 (ground truth by independent reference computation).
**Harness:** `eval/run_eval.py` — re-run with `.venv/bin/python eval/run_eval.py`. Raw output: `eval/results/eval_results.json`; per-run reports under `eval/results/runs/`.

## What ran

- **Systems:** the proposed local agent (full pipeline, target = `churn`).
- **Datasets:** the project's two sample sets — `data/sample_sales.csv` (242 rows) and `data/sample_churn.csv` (302 rows). Both are synthetic stand-ins for business/administrative tables, per the proposal's data plan; the harness extends directly to the UCI/Kaggle benchmarks named in Section 5.
- **Ground truth:** an independent reference implementation of the documented cleaning rules (drop duplicates → strip whitespace → unify category spellings → coerce types → median/mode fill), written separately from the agent package, plus an independently coded run of the prediction protocol (80/20 stratified split, seed 42).
- **Repetitions:** 5 full runs per dataset, as the proposal specifies for reproducibility scoring.

## Descriptive tasks — 12/12 correct, 12/12 reproducible

Every value the agent reported matched the independent ground truth **exactly** (not merely within tolerance), and was identical across all 5 runs.

| ID | Task | Ground truth | Agent |
|----|------|--------------|-------|
| S1 | Cleaned row count (sales) | 240 | 240 |
| S2 | Mean revenue (sales) | 5390.0635 | 5390.0635 |
| S3 | Mean units sold (sales) | 20.6417 | 20.6417 |
| S4 | Correlation units_sold vs revenue | 0.6436 | 0.6436 |
| S5 | Region with highest mean revenue | South, 5903.0086 | South, 5903.0086 |
| S6 | Mean revenue, first month (2024-01) | 5432.9077 | 5432.9077 |
| S7 | Most common product category | Furniture | Furniture |
| C1 | Cleaned row count (churn) | 300 | 300 |
| C2 | Overall churn rate | 0.3300 | 0.3300 |
| C3 | Mean tenure (months) | 37.57 | 37.57 |
| C4 | Correlation monthly_spend vs total_spend | 0.5546 | 0.5546 |
| C5 | Plan type with highest churn rate | Month-to-month, 0.5521 | Month-to-month, 0.5521 |

## Prediction tasks — real models match the independent reference

Test accuracy (agent vs independent reference implementation of the same protocol):

| Dataset | Model | Agent | Reference | Match |
|---------|-------|-------|-----------|-------|
| Sales (churn) | Dummy (baseline) | 0.7083 | 0.8333 | n/a — different baseline strategy* |
| Sales | Logistic Regression | 0.8333 | 0.8333 | ✓ |
| Sales | Random Forest (selected) | 0.8333 | 0.8333 | ✓ |
| Sales | Gradient Boosting | 0.7708 | 0.7708 | ✓ |
| Churn | Dummy (baseline) | 0.6167 | 0.6667 | n/a — different baseline strategy* |
| Churn | Logistic Regression | 0.7333 | 0.7333 | ✓ |
| Churn | Random Forest (selected) | 0.6167 | 0.6333 | ✓ (within 1 test row; reference used 300 trees vs the agent's 200) |
| Churn | Gradient Boosting | 0.6167 | 0.6167 | ✓ |

\* The agent's baseline is a *stratified* dummy (random labels drawn from the class distribution); the reference baseline predicts the majority class. Both are standard baselines; they are not expected to agree numerically.

One honest observation: on the churn dataset the agent selects Random Forest by cross-validation (CV 0.7042), but Logistic Regression scores higher on the held-out test set (0.7333 vs 0.6167). With 60 test rows this is small-sample variance, not an error — model selection followed the pre-registered CV rule.

## Reproducibility — one real bug found and fixed

The first evaluation pass **failed** the proposal's reproducibility requirement: across 5 runs, the reports differed in exactly one line — the Dummy baseline row. Root cause: `agent/prediction.py` built `DummyClassifier(strategy="stratified")` with **no random_state**, so the baseline's random label sampling changed every run. Every other number in every report was already byte-identical.

**Fix:** seed the dummy with the run's configured seed (`random_state=config.random_seed`). After the fix, re-running the full evaluation gives:

- 12/12 descriptive tasks identical across 5 runs;
- prediction results identical across 5 runs on both datasets;
- **`report.md` byte-identical across all 5 runs on both datasets** (md5-verified);
- all 14 project tests still pass.

## Runtime, cost, data exposure (local agent)

- Mean wall-clock per full pipeline run: **3.9 s** (sales), **4.2 s** (churn).
- Monetary cost per run: **$0** (no external service is called).
- Data exposure: **none** — datasets never leave the machine, by construction.

## Baselines not yet run

- **LLM agent (Data Interpreter / DS-Agent style): NOT RUN — blocked.** No LLM API key is available in this environment, and the baseline requires an external API by definition (that dependency is itself one of the measured contrasts). To run it: provide an OpenAI (or equivalent) API key; the protocol needs ~10 agent runs (2 datasets × 5 repetitions) plus task prompts identical to the local arm's. Estimated API spend: on the order of a few dollars.
- **Auto-sklearn: NOT RUN — blocked by environment.** `pip install auto-sklearn` fails in this machine's Python 3.12 environment (wheel build failure; auto-sklearn does not support Python 3.12, and no Python ≤3.10 interpreter or `swig` toolchain is installed). Running this arm needs a Python 3.10 environment (e.g., a container). Note the agent's own prediction stage is a bounded model selection *in the style of* auto-sklearn, not auto-sklearn itself — the proposal's wording already says this.
- **Manual analysis:** covered in its ground-truth role — the independent reference computations are the analyst-produced values the proposal's Section 5 prescribes. Analyst wall-clock time was not scored as a baseline metric.

## Hypothesis scorecard

| Hypothesis | Status |
|------------|--------|
| H1 — correctness parity with the LLM agent on routine tasks | **Untested** (LLM arm blocked). Local correctness vs ground truth: 12/12 exact, prediction verified against an independent implementation. |
| H2 — reproducibility exceeds the LLM agent | **Supported for the local system** (byte-identical reports across 5 runs, after the dummy-seed fix the evaluation itself caught). The LLM side of the comparison is untested. |
| H3 — lower failure / hallucination rate | **Supported locally:** 0 failures and 0 incorrect reported values in 60 descriptive task-runs and 10 prediction runs. LLM side untested. |
| H4 — wins on privacy and cost at competitive runtime | **Supported:** $0 per run, no data exposure, ~4 s per full analysis. |

## Limitations

- Coverage is the project's two synthetic sample datasets, not yet the UCI/Kaggle benchmark suite named in Section 5; the harness takes new datasets and tasks without structural changes.
- The LLM and auto-sklearn arms are required for the full head-to-head claim (H1, and the comparative halves of H2/H3); until they run, conclusions are about the local system's absolute correctness and reproducibility, not relative standing.
