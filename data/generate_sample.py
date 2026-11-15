"""Generate the synthetic sample dataset used by the README examples.

The data is fake (seeded, so regenerating produces the same file). It
deliberately contains the mess real spreadsheets have: a few missing
values, inconsistent region spellings (" North", "NORTH"), and two exact
duplicate rows, so the cleaning stage has something to log.

Usage:
    python data/generate_sample.py
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running in an environment without __file__
    HERE = Path.cwd() / "data"


def main() -> None:
    rng = random.Random(7)
    numpy_rng = np.random.default_rng(7)

    regions = ["North", "South", "East", "West"]
    categories = ["Electronics", "Furniture", "Office Supplies"]
    channels = ["Online", "Retail", "Partner"]
    segments = ["Small Business", "Consumer", "Enterprise"]

    rows = []
    start = pd.Timestamp("2024-01-01")
    for order_id in range(1, 241):
        region = rng.choice(regions)
        # A few inconsistent spellings, on purpose.
        if order_id % 53 == 0:
            region = " " + region
        elif order_id % 71 == 0:
            region = region.upper()
        category = rng.choice(categories)
        channel = rng.choice(channels)
        segment = rng.choice(segments)
        units = int(numpy_rng.integers(1, 40))
        unit_price = round(float(numpy_rng.uniform(10, 500)), 2)
        revenue = round(units * unit_price, 2)
        satisfaction = rng.choice([1, 2, 3, 4, 5, None]) if order_id % 19 == 0 else rng.choice([1, 2, 3, 4, 5])
        # Churn is more likely for Consumer + Retail rows (a learnable pattern).
        churn_probability = 0.35 if (segment == "Consumer" and channel == "Retail") else 0.12
        churn = 1 if rng.random() < churn_probability else 0
        rows.append(
            {
                "order_id": order_id,
                "order_date": start + pd.Timedelta(days=order_id % 365),
                "region": region,
                "product_category": category,
                "channel": channel,
                "customer_segment": segment,
                "units_sold": None if order_id % 23 == 0 else units,
                "unit_price": unit_price,
                "revenue": revenue,
                "satisfaction": satisfaction,
                "churn": churn,
            }
        )

    frame = pd.DataFrame(rows)
    # Two exact duplicate rows, on purpose.
    frame = pd.concat([frame, frame.iloc[[5, 120]]], ignore_index=True)
    frame = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    frame.to_csv(HERE / "sample_sales.csv", index=False)
    print(f"Wrote {len(frame)} rows to {HERE / 'sample_sales.csv'}")


if __name__ == "__main__":
    main()
