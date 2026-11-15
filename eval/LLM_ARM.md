# LLM-agent arm — status and how to run it

**Status (2026-10-24): harness built and dry-run verified; arm NOT yet run.**
The blocker is unchanged: no LLM API key is available in this environment.
No scores exist for this arm, and none are claimed.

## What the harness does

`eval/llm_eval.py` implements the revised proposal's Section 4 protocol for
the LLM baseline:

- Same datasets as the local arm: `data/sample_sales.csv` (242 rows) and
  `data/sample_churn.csv` (302 rows).
- Same task prompts as the local arm: S1–S7 (sales) and C1–C5 (churn),
  with the documented cleaning rules stated in the prompt so the LLM
  cleans the data the same way before computing values.
- Same scoring: values are compared against the independent ground truth
  computed by `eval/run_eval.py` (via `build_tasks`), using its tolerances.
- Repetitions: 5 per dataset by default (`--reps`), as the proposal
  requires at least five.
- Metrics recorded per rep: correct count, runtime, prompt/completion
  tokens, dollar cost (only when `LLM_INPUT_PRICE_PER_1K_USD` and
  `LLM_OUTPUT_PRICE_PER_1K_USD` are set — no price is invented),
  failures (missing keys / unparseable replies), and reproducibility
  (whether all 5 reps return identical answers).
- Data exposure: `yes` — the CSV text is sent to the provider's API.
  That is a measured contrast with the local agent, and the reason this
  arm runs only on the synthetic sample datasets unless the user
  explicitly approves other data.

## Honest scope limit

This is a **direct-answer** LLM baseline in the style of Data Interpreter /
DS-Agent, not a replica of either system: it does not generate and
execute code. Prediction-model training is therefore out of scope for
this arm; the scored set is the 12 descriptive tasks. The prediction
comparison stays with the local arm and the auto-sklearn arm.

## How to run it

From the project root, with the project venv active:

```bash
# Free check — validates prompts, ground truth, and scoring, no API call:
.venv/bin/python eval/llm_eval.py --dry-run

# Real run — needs a key, ~10 calls (2 datasets x 5 reps):
OPENAI_API_KEY=... .venv/bin/python eval/llm_eval.py --reps 5
```

Dry-run output on 2026-10-24 (verified):

```
sales: 7 tasks, 19423 prompt chars
churn: 5 tasks, 20444 prompt chars
Ground truth tasks: 12 (expected 12: S1-S7, C1-C5)
```

Without a key the script exits with code 2 and writes no scores.

## Getting a key

Add an OpenAI (or equivalent) API key through the Secure Vault
API-access flow — never in chat. The key is read from the
`OPENAI_API_KEY` environment variable only; it is never printed, logged,
or written to output files. Model is selectable with `LLM_MODEL`
(default `gpt-4o-mini`); the model used is recorded in the output JSON.

## Output

`eval/results/llm_results.json` — per-rep task rows (agent value vs
ground truth, correct/incorrect/missing), plus a summary with
correct-per-rep, mean runtime, reproducibility, failures, tokens, and
cost. Once this arm runs, its summary belongs in `EVAL_REPORT.md`
alongside the local arm, and H1 (correctness parity) can finally be
judged instead of left untested.

## Auto-sklearn arm (separate)

Also in progress on 2026-10-24: a Python 3.10.15 environment was
installed at `workspace/data-analysis-agent/.py310/` (standalone build,
since apt on Ubuntu 24.04 has no Python 3.10 and there is no Docker or
conda here), and an `auto-sklearn` install into `.venv310` was started.
That arm is prediction-only, per the proposal. Its result will be
reported separately; until it completes, the auto-sklearn arm is also
NOT RUN.
