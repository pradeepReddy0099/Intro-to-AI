"""LLM-agent arm of the head-to-head evaluation (revised proposal, Section 4).

This is the LLM comparison arm of that evaluation: the same datasets
and the same task prompts as the local arm, run at least five times, and
scored on correctness, reproducibility, failure / hallucination rate,
runtime, monetary cost, and data exposure.

Design notes (read before claiming more than this arm measures):

- Mode: direct answer. The LLM receives the raw CSV text plus the exact
  task definitions and returns one JSON object with one value per task.
  This is an LLM baseline in the style of Data Interpreter / DS-Agent, not
  a replica of either system: it does not generate and execute code, so
  prediction-model training is out of scope here. The 12 descriptive
  tasks (S1-S7, C1-C5) are the scored set, matching the local arm's
  descriptive table. Prediction tasks stay with the local arm and the
  auto-sklearn arm.
- Data exposure is Yes by construction: the CSV text is sent to the
  provider's API. That is itself one of the measured contrasts with the
  local agent, and it is why this arm only ever runs on the synthetic
  sample datasets (or other data the user has explicitly approved).
- No key, no run. If OPENAI_API_KEY is not set, the script exits with a
  clear blocked message and writes no scores. --dry-run validates prompt
  construction, ground truth, and scoring without calling any API, so
  the harness can be checked for free.
- Cost: the script records prompt / completion tokens from the API
  response. Dollar cost is computed only when LLM_PRICE_PER_1K env vars
  are set (LLM_INPUT_PRICE_PER_1K_USD, LLM_OUTPUT_PRICE_PER_1K_USD);
  otherwise cost is reported as tokens, and the user reads dollars off
  the provider's billing page. No price is invented for a model.
- The API key is read from the environment only. It is never printed,
  logged, or written to any output file.

Usage (from the project root, with the project venv active):
    python eval/llm_eval.py --dry-run
    OPENAI_API_KEY=... python eval/llm_eval.py --reps 5
    OPENAI_API_KEY=... LLM_MODEL=gpt-4o-mini python eval/llm_eval.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from eval.run_eval import build_tasks, values_match  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CLEANING_RULES = (
    "Cleaning rules (apply these before computing any value): drop exact "
    "duplicate rows; strip whitespace from text cells; unify case variants "
    "of a category to its most frequent spelling; coerce numeric and "
    "datetime columns; fill missing numeric values with the median and "
    "missing categorical values with the mode; leave datetime gaps unfilled."
)
SYSTEM_PROMPT = (
    "You are a data analysis agent. Compute every requested value from "
    "the CSV data provided, after applying the stated cleaning rules. "
    "Return a single JSON object only, with one key per task id, and no "
    "explanation, markdown, or extra text. Numeric values must be numbers, "
    "not strings. For a 'highest group' task, return a two-element array "
    "[group_name, mean_value]."
)


def build_prompts():
    """One prompt per dataset, with the local arm's task wording verbatim."""
    tasks, refs = build_tasks()
    prompts = {}
    for dataset in ("sales", "churn"):
        dataset_tasks = [t for t in tasks if t["dataset"] == dataset]
        csv_path = BASE / f"data/sample_{dataset}.csv"
        csv_text = csv_path.read_text()
        task_lines = "\n".join(
            f"- {t['id']}: {t['desc']}" for t in dataset_tasks
        )
        user_prompt = (
            f"Dataset: {dataset}\n\n"
            f"{CLEANING_RULES}\n\n"
            f"CSV data:\n```csv\n{csv_text}```\n\n"
            f"Tasks (return one JSON key per task id):\n{task_lines}"
        )
        prompts[dataset] = {
            "tasks": [t["id"] for t in dataset_tasks],
            "prompt": user_prompt,
            "prompt_sha256": hashlib.sha256(
                user_prompt.encode()).hexdigest(),
            "prompt_chars": len(user_prompt),
        }
    ground_truth = {t["id"]: t["ref"](refs[t["dataset"]]) for t in tasks}
    task_meta = {t["id"]: {"dataset": t["dataset"], "tol": t["tol"],
                           "desc": t["desc"]} for t in tasks}
    return prompts, ground_truth, task_meta


def call_llm(model, system_prompt, user_prompt, api_key, timeout=300):
    """One chat-completions call. Returns (parsed_json_or_None, usage, raw)."""
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }).encode()
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        return None, {}, f"HTTP {exc.code}: {detail}"
    except Exception as exc:  # network / timeout: a failed run, not a score
        return None, {}, f"{type(exc).__name__}: {exc}"
    text = payload["choices"][0]["message"]["content"]
    usage = payload.get("usage", {})
    try:
        return json.loads(text), usage, text
    except json.JSONDecodeError:
        return None, usage, text


def score_answers(answers, ground_truth, task_meta):
    """Score one rep: correct count, and which reported values hallucinate.

    A reported value that parses but does not match ground truth counts
    as an incorrect value (the proposal's hallucination measure for
    numeric claims). A missing key or unparseable reply counts as a
    failure, not as a hallucination.
    """
    rows = []
    for task_id, truth in ground_truth.items():
        if answers is None or task_id not in answers:
            rows.append({"id": task_id, "status": "missing",
                         "agent": None, "ground_truth": truth})
            continue
        value = answers[task_id]
        # JSON turns tuples into lists; normalise before comparing.
        if isinstance(value, list):
            value = tuple(value)
        tol = task_meta[task_id]["tol"]
        ok = values_match(value, truth, tol)
        rows.append({"id": task_id,
                     "status": "correct" if ok else "incorrect",
                     "agent": value, "ground_truth": truth})
    correct = sum(r["status"] == "correct" for r in rows)
    return rows, correct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=5,
                        help="repetitions per dataset (proposal: at least 5)")
    parser.add_argument("--model", default=None,
                        help="model name (default: $LLM_MODEL or gpt-4o-mini)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate prompts and scoring without an API call")
    parser.add_argument("--out", default=None,
                        help="output JSON path (default: eval/results/llm_results.json)")
    args = parser.parse_args()

    prompts, ground_truth, task_meta = build_prompts()
    model = args.model or os.environ.get("LLM_MODEL", "gpt-4o-mini")

    if args.dry_run:
        for dataset, info in prompts.items():
            print(f"{dataset}: {len(info['tasks'])} tasks, "
                  f"{info['prompt_chars']} prompt chars, "
                  f"sha256 {info['prompt_sha256'][:12]}")
        print(f"Ground truth tasks: {len(ground_truth)} "
              f"(expected 12: S1-S7, C1-C5)")
        print("Dry run OK: prompts, ground truth, and scoring load. "
              "No API call made. Set OPENAI_API_KEY to run for real.")
        return 0

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("BLOCKED: OPENAI_API_KEY is not set. No scores were produced. "
              "Add the key through the Secure Vault API-access flow, then "
              "re-run this script; estimated spend is a few dollars for "
              "2 datasets x 5 repetitions.", file=sys.stderr)
        return 2

    input_price = os.environ.get("LLM_INPUT_PRICE_PER_1K_USD")
    output_price = os.environ.get("LLM_OUTPUT_PRICE_PER_1K_USD")
    out_path = Path(args.out) if args.out else RESULTS_DIR / "llm_results.json"

    runs = {}
    for rep in range(args.reps):
        for dataset, info in prompts.items():
            start = time.perf_counter()
            answers, usage, raw = call_llm(
                model, SYSTEM_PROMPT, info["prompt"], api_key)
            runtime = round(time.perf_counter() - start, 2)
            rows, correct = score_answers(answers, {
                tid: ground_truth[tid] for tid in info["tasks"]}, task_meta)
            cost = None
            if input_price and output_price:
                cost = round(
                    usage.get("prompt_tokens", 0) / 1000 * float(input_price)
                    + usage.get("completion_tokens", 0) / 1000
                    * float(output_price), 4)
            runs.setdefault(dataset, []).append({
                "rep": rep + 1, "runtime_seconds": runtime,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cost_usd": cost, "correct": correct,
                "total": len(info["tasks"]),
                "failed": answers is None,
                "rows": rows,
            })
            print(f"rep {rep + 1}/{args.reps} {dataset}: "
                  f"{correct}/{len(info['tasks'])} correct, "
                  f"{runtime}s, failed={answers is None}", flush=True)

    summary = {"model": model, "reps": args.reps, "mode": "direct_answer",
               "data_exposure": "yes (CSV sent to provider API)",
               "cost_note": ("USD only when LLM_*_PRICE_PER_1K_USD are set; "
                             "otherwise tokens are recorded and dollars "
                             "come from provider billing."),
               "datasets": {}}
    for dataset, reps in runs.items():
        all_answers = [json.dumps({r["id"]: r["agent"] for r in rep["rows"]},
                                  default=str, sort_keys=True)
                       for rep in reps]
        summary["datasets"][dataset] = {
            "correct_per_rep": [r["correct"] for r in reps],
            "mean_runtime_seconds": round(
                sum(r["runtime_seconds"] for r in reps) / len(reps), 2),
            "reproducible": len(set(all_answers)) == 1,
            "failures": sum(r["failed"] for r in reps),
            "total_prompt_tokens": sum(
                r["prompt_tokens"] or 0 for r in reps),
            "total_completion_tokens": sum(
                r["completion_tokens"] or 0 for r in reps),
            "total_cost_usd": (round(sum(r["cost_usd"] or 0 for r in reps), 4)
                               if all(r["cost_usd"] is not None for r in reps)
                               else None),
        }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"summary": summary, "runs": runs}, indent=2, default=str))
    print(f"\nWrote {out_path}")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
