# AI-Powered Local Analysis Agent — UI version

Upload a table, ask a question, get a report with charts and a model
comparison. Everything runs on your own machine — no language model,
no network calls, no per-run cost. The same input always gives the same
report (fixed seed 42).

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

A page opens in your browser. Upload a dataset (CSV, TSV, or Excel) or
pick one of the two samples in `data/`, choose what you want to do,
pick a target column if you want a prediction, and press
**Run the analysis**. Every run is kept under `output/ui_runs/` and can
be reopened from **Previous results** on the page.

## Run the website on GitHub (Codespaces)

No install needed — GitHub runs it for you:

1. Open the repository on GitHub, click **Code** → **Codespaces** →
   **Create codespace on main**.
2. Wait for the setup to finish (it installs the requirements once).
   The website opens by itself in a browser tab.
3. If it doesn't open, go to the **Ports** tab and open port **8501**.

GitHub includes a free monthly allowance for Codespaces on personal
accounts. The website stops when the codespace stops — start a new
codespace any time to run it again.

## What is in here

- `app.py` — the Streamlit page (the only way to run it)
- `agent/` — the pipeline the page runs: profile, clean, analyse,
  chart, predict, report (one file per step)
- `data/` — two synthetic sample tables to try it with
