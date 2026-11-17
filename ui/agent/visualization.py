"""Make simple charts from the cleaned table."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from agent.analysis import columns_of_type


def make_charts(
    frame: pd.DataFrame,
    profile: dict,
    analysis: dict,
    charts_dir,
    max_histograms: int = 4,
    max_bar_charts: int = 3,
) -> list:
    """Save charts to `charts_dir` and return a list of chart metadata."""
    charts_path = Path(charts_dir)
    charts_path.mkdir(parents=True, exist_ok=True)
    charts = []

    numeric_columns = columns_of_type(profile, frame, "numeric", exclude_identifiers=True)
    categorical_columns = columns_of_type(profile, frame, "categorical", exclude_identifiers=True)

    for column in numeric_columns[:max_histograms]:
        filename = f"hist_{_slug(column)}.png"
        _save_histogram(frame[column], column, charts_path / filename)
        charts.append(
            {
                "file": filename,
                "title": f"Distribution of {column}",
                "description": "Histogram of the cleaned values.",
            }
        )

    for column in categorical_columns[:max_bar_charts]:
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


def _new_figure(width: float, height: float):
    """A blank figure and axes of the given size, for one chart."""
    return plt.subplots(figsize=(width, height))


def _finish_figure(fig, path: Path) -> None:
    """Lay out, save, and close a finished chart figure."""
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _save_histogram(series: pd.Series, column: str, path: Path) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = _new_figure(6, 4)
    ax.hist(values, bins=20, edgecolor="white")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    _finish_figure(fig, path)


def _save_bar_chart(series: pd.Series, column: str, path: Path) -> None:
    counts = series.value_counts(dropna=True).head(10)
    fig, ax = _new_figure(6, 4)
    ax.bar([str(label) for label in counts.index], counts.values)
    ax.set_title(f"Counts by {column}")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    _finish_figure(fig, path)


def _save_heatmap(numeric_frame: pd.DataFrame, path: Path) -> None:
    corr = numeric_frame.corr(numeric_only=True).values
    labels = list(numeric_frame.columns)
    fig, ax = _new_figure(6, 5)
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
    _finish_figure(fig, path)


def _save_trend(series_rows: list, numeric_columns: list, path: Path) -> None:
    periods = [row["period"] for row in series_rows]
    fig, ax = _new_figure(7, 4)
    for column in numeric_columns[:3]:
        values = [row.get(column) for row in series_rows]
        ax.plot(periods, values, marker="o", label=column)
    ax.set_title("Trend over time")
    ax.set_xlabel("Period")
    ax.set_ylabel("Average value")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    _finish_figure(fig, path)


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_").lower() or "column"
