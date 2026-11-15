"""Command-line interface.

Usage:
    python -m agent analyze data.csv --out output/
    python -m agent analyze data.csv --question "what drives revenue?" --out output/
    python -m agent analyze data.csv --target churn --out output/
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from agent.pipeline import run_pipeline
from agent.reporting import direct_answer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Local, deterministic data analysis agent for tabular data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser(
        "analyze", help="Analyze a CSV or Excel file and write a report."
    )
    analyze.add_argument("input", help="Path to a .csv, .tsv, .xlsx, or .xls file.")
    analyze.add_argument("--question", default=None, help="Question in plain language.")
    analyze.add_argument("--target", default=None, help="Target column for prediction.")
    analyze.add_argument("--out", default="output", help="Output directory.")
    analyze.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    return parser


def _save_upload(input_path: Path) -> None:
    """Copy the user's data file into data/ and tests/user_data/.

    The file becomes one of the project's own datasets: it sits in data/
    with the samples, and the universal tests pick it up from
    tests/user_data/ on the next test run. A file that is already saved
    (same name and same contents) is left as is; a different file with
    the same name is saved with a number added, never overwritten.
    """
    root = Path(__file__).resolve().parent.parent
    if input_path.resolve().parent == (root / "data").resolve():
        return  # already one of the project's data files
    for folder in (root / "data", root / "tests" / "user_data"):
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / input_path.name
        if target.exists():
            if target.read_bytes() == input_path.read_bytes():
                continue  # this exact file is already saved here
            stem, suffix = input_path.stem, input_path.suffix
            count = 2
            while (folder / f"{stem}_{count}{suffix}").exists():
                count += 1
            target = folder / f"{stem}_{count}{suffix}"
        shutil.copyfile(input_path, target)
        print(f"Saved a copy of your data file to {target.relative_to(root)}")


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        results = run_pipeline(
            args.input,
            question=args.question,
            target=args.target,
            out_dir=args.out,
            seed=args.seed,
        )
        if results["status"] == "refused":
            print(f"Refused: {results['reason']}")
            print(f"Refusal report: {results['report']['html']}")
            return 1
        _save_upload(Path(args.input))
        print(f"Rows after cleaning: {results['cleaned_rows']}")
        print(f"Answer: {direct_answer(results)}")
        print(f"Report: {results['report']['html']}")
        print(f"Charts: {len(results['charts'])} PNG files in {args.out}/charts/")
        if results.get("prediction"):
            prediction = results["prediction"]
            print(
                f"Prediction: best model {prediction['best_model']} "
                f"(test {prediction['metric']} {prediction['best_test_score']:.3f})"
            )
        elif results.get("prediction_refusal"):
            print(f"Prediction: {results['prediction_refusal']}")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
