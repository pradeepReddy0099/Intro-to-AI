"""Generate a second synthetic sample dataset (customer churn) for the agent.

The data is fake (seeded, so regenerating produces the same file). Like
real spreadsheets it has a few missing values, inconsistent region
spellings (" North", "NORTH"), and two exact duplicate rows, so the
cleaning stage has something to log. The churn column is a usable
prediction target: churn is more likely for month-to-month customers
with high support calls and low tenure.

Usage:
    python data/generate_churn_sample.py
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
    rng = random.Random(42)
    numpy_rng = np.random.default_rng(42)

    regions = ["North", "South", "East", "West"]
    plans = ["Month-to-month", "One year", "Two year"]
    payment_methods = ["Credit card", "Bank transfer", "Electronic check", "Mailed check"]

    rows = []
    start = pd.Timestamp("2023-01-01")
    for customer_id in range(1, 301):
        region = rng.choice(regions)
        # A few inconsistent spellings, on purpose.
        if customer_id % 47 == 0:
            region = " " + region
        elif customer_id % 83 == 0:
            region = region.upper()
        plan = rng.choice(plans)
        tenure = int(numpy_rng.integers(1, 73))
        monthly_spend = round(float(numpy_rng.uniform(20, 120)), 2)
        support_calls = int(numpy_rng.integers(0, 10))
        payment = rng.choice(payment_methods)
        total_spend = round(monthly_spend * tenure, 2)
        # Churn is more likely for month-to-month plans, short tenure,
        # and many support calls (a learnable pattern).
        churn_probability = 0.15
        if plan == "Month-to-month":
            churn_probability += 0.30
        if tenure < 12:
            churn_probability += 0.15
        if support_calls >= 6:
            churn_probability += 0.15
        churn = 1 if rng.random() < min(churn_probability, 0.9) else 0
        rows.append(
            {
                "customer_id": customer_id,
                "signup_date": start + pd.Timedelta(days=int(numpy_rng.integers(0, 700))),
                "region": region,
                "plan_type": plan,
                "payment_method": payment,
                "tenure_months": tenure,
                "monthly_spend": None if customer_id % 29 == 0 else monthly_spend,
                "support_calls": support_calls,
                "total_spend": total_spend,
                "churn": churn,
            }
        )

    frame = pd.DataFrame(rows)
    # Two exact duplicate rows, on purpose.
    frame = pd.concat([frame, frame.iloc[[10, 200]]], ignore_index=True)
    frame = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
    frame.to_csv(HERE / "sample_churn.csv", index=False)
    print(f"Wrote {len(frame)} rows to {HERE / 'sample_churn.csv'}")


if __name__ == "__main__":
    main()
