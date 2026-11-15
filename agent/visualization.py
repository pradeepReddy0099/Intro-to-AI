"""Visualization: rule-based chart selection, rendered locally to PNG.

Chart types are chosen from variable types, never from free-form reasoning:
histogram for a numeric column, bars for a categorical column, heatmap for
the correlation matrix, line chart for a trend over time.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_charts(frame: pd.DataFrame, profile: dict, analysis: dict, charts_dir) -> list:
    """Save charts to `charts_dir` and return a list of chart metadata."""
    charts_path = Path(charts_dir)
    charts_path.mkdir(parents=True, exist_ok=True)
    charts = []

    numeric_columns = [
        c["name"]
        for c in profile["columns"]
        if c["inferred_type"] == "numeric"
        and not c.get("is_identifier")
        and c["name"] in frame.columns
    ]
    categorical_columns = [
        c["name"]
        for c in profile["columns"]
        if c["inferred_type"] == "categorical"
        and not c.get("is_identifier")
        and c["name"] in frame.columns
    ]

    for column in numeric_columns[:4]:
        filename = f"hist_{_slug(column)}.png"
        _save_histogram(frame[column], column, charts_path / filename)
        charts.append(
            {
                "file": filename,
                "title": f"Distribution of {column}",
                "description": "Histogram of the cleaned values.",
            }
        )

    for column in categorical_columns[:3]:
        filename = f"bar_{_slug(column)}.png"
        _save_bar_chart(frame[column], column, charts_path / filename)
        charts.append(
            {
                "file": filename,
                "title": f"Counts by {column}",
                "description": "Bar chart of the ten most frequent values.",
            }
        )

    if len(numeric_columns) >= 2:
        filename = "correlation_heatmap.png"
        _save_heatmap(frame[numeric_columns], charts_path / filename)
        charts.append(
            {
                "file": filename,
                "title": "Correlation heatmap",
                "description": "Pearson correlation between numeric columns.",
            }
        )

    if analysis["trends"]["series"] and numeric_columns:
        filename = "trend.png"
        _save_trend(analysis["trends"]["series"], numeric_columns, charts_path / filename)
        charts.append(
            {
                "file": filename,
                "title": "Trend over time",
                "description": "Average value per period for numeric columns.",
            }
        )

    return charts


def _save_histogram(series: pd.Series, column: str, path: Path) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=20, edgecolor="white")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _save_bar_chart(series: pd.Series, column: str, path: Path) -> None:
    counts = series.value_counts(dropna=True).head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(label) for label in counts.index], counts.values)
    ax.set_title(f"Counts by {column}")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _save_heatmap(numeric_frame: pd.DataFrame, path: Path) -> None:
    corr = numeric_frame.corr(numeric_only=True).values
    labels = list(numeric_frame.columns)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, f"{corr[row, col]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlation heatmap")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _save_trend(series_rows: list, numeric_columns: list, path: Path) -> None:
    periods = [row["period"] for row in series_rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    for column in numeric_columns[:3]:
        values = [row.get(column) for row in series_rows]
        ax.plot(periods, values, marker="o", label=column)
    ax.set_title("Trend over time")
    ax.set_xlabel("Period")
    ax.set_ylabel("Average value")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_").lower() or "column"
