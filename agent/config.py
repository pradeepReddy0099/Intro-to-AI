"""Versioned pipeline configuration.

Every run saves its configuration alongside the report, so a result can
always be traced back to the exact settings that produced it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

PIPELINE_VERSION = "1.0.0"


@dataclass
class AgentConfig:
    """Settings that control one pipeline run.

    The defaults are deliberately fixed: changing them is a conscious,
    recorded decision, which is what makes runs reproducible.
    """

    version: str = PIPELINE_VERSION
    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    min_rows_for_prediction: int = 30
    min_class_samples: int = 5
    max_histogram_columns: int = 4
    max_bar_columns: int = 3
    max_categories_for_grouping: int = 10

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
