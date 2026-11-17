"""Fixed settings for one pipeline run."""
from __future__ import annotations

from dataclasses import asdict, dataclass

PIPELINE_VERSION = "1.0.0"


@dataclass
class AgentConfig:
    """Fixed settings for one run."""

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
