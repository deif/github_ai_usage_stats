#!/usr/bin/env python3
"""
Generate a synthetic "AI Usage Report" CSV for demos and screenshots.

The output matches the column schema expected by ``generate_dashboard.py`` and
contains realistic-looking (but entirely
fake) data: a spread of users, models, weekday patterns and costs across a few
months, so every dashboard panel has something interesting to show.

Usage
-----
    python demo/generate_demo_data.py                  # -> demo/data/ai_usage_report.csv
    python demo/generate_demo_data.py -o reports --seed 7
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from datetime import date, datetime, time, timedelta
from typing import List

# Output column order (matches what load_data() reads).
COLUMNS = [
    "date",
    "username",
    "product",
    "sku",
    "model",
    "organization",
    "repository",
    "cost_center_name",
    "quantity",
    "applied_cost_per_quantity",
    "gross_amount",
    "discount_amount",
    "net_amount",
    "total_monthly_quota",
    "aic_quantity",
    "aic_gross_amount",
]

# Fictional users with a relative activity weight (higher = more records).
USERS = [
    ("ada.lovelace", 10), ("alan.turing", 9), ("grace.hopper", 8),
    ("linus.torvalds", 8), ("margaret.hamilton", 7), ("ken.thompson", 6),
    ("dennis.ritchie", 6), ("barbara.liskov", 5), ("donald.knuth", 5),
    ("tim.berners-lee", 4), ("guido.van-rossum", 4), ("bjarne.stroustrup", 3),
    ("james.gosling", 3), ("john.carmack", 2), ("brian.kernighan", 2),
]

# (model name, relative weight, credits-per-request multiplier).
MODELS = [
    ("Claude Sonnet 4", 30, 1.0),
    ("Auto: Claude Sonnet 4", 16, 1.0),
    ("GPT-5-Codex", 14, 1.0),
    ("Claude Sonnet 3.7", 12, 1.0),
    ("GPT-4.1", 12, 1.0),
    ("Claude Opus 4.1", 8, 10.0),
    ("GPT-5 mini", 7, 0.0),
    ("Claude Haiku 3.5", 6, 0.33),
    ("Code Review", 5, 1.0),
]

REPOS = [
    "platform-api", "web-frontend", "mobile-app", "data-pipeline",
    "infra-terraform", "ml-models", "docs-site", "billing-service",
]
COST_CENTERS = ["Engineering", "Data Science", "Platform", "Mobile"]

PRODUCT = "Copilot"
SKU = "Copilot Premium Request"
ORG = "acme-corp"
PRICE_PER_QUANTITY = 0.04  # USD per credit
MONTHLY_QUOTA = 1000


def _weighted_choice(rng: random.Random, items):
    population = [v for v, *_ in items]
    weights = [w for _, w, *_ in items]
    return rng.choices(population, weights=weights, k=1)[0]


def generate_rows(rng: random.Random, start: date, end: date) -> List[dict]:
    model_lookup = {name: mult for name, _, mult in MODELS}
    rows: List[dict] = []
    day = start
    while day <= end:
        # Weekends are much quieter than weekdays.
        weekday = day.weekday()
        if weekday < 5:
            active_count = rng.randint(9, len(USERS))
        else:
            active_count = rng.randint(1, 4)

        active_users = rng.sample(
            [u for u, _ in USERS],
            k=min(active_count, len(USERS)),
        )
        for username in active_users:
            for _ in range(rng.randint(3, 10)):  # several records per active user
                model = _weighted_choice(rng, MODELS)
                multiplier = model_lookup[model]
                requests = rng.randint(2, 30)
                quantity = round(requests * multiplier, 2)

                gross = round(quantity * PRICE_PER_QUANTITY, 4)
                # Occasional discount to make the gross/net split visible.
                discount = round(gross * rng.choice([0, 0, 0, 0.1, 0.2]), 4)
                net = round(gross - discount, 4)

                ts = datetime.combine(
                    day, time(rng.randint(7, 20), rng.randint(0, 59))
                )
                rows.append({
                    "date": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "username": username,
                    "product": PRODUCT,
                    "sku": SKU,
                    "model": model,
                    "organization": ORG,
                    "repository": rng.choice(REPOS),
                    "cost_center_name": rng.choice(COST_CENTERS),
                    "quantity": quantity,
                    "applied_cost_per_quantity": PRICE_PER_QUANTITY,
                    "gross_amount": gross,
                    "discount_amount": discount,
                    "net_amount": net,
                    "total_monthly_quota": MONTHLY_QUOTA,
                    "aic_quantity": 0,
                    "aic_gross_amount": 0,
                })
        day += timedelta(days=1)
    return rows


def parse_args(argv=None) -> argparse.Namespace:
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    p = argparse.ArgumentParser(description="Generate a synthetic AI usage report CSV.")
    p.add_argument("-o", "--output-dir", default=default_dir,
                   help="Directory to write the demo CSV into (default: demo/data).")
    p.add_argument("-n", "--filename", default="ai_usage_report.csv",
                   help="CSV file name written inside the output dir "
                        "(default: ai_usage_report.csv).")
    p.add_argument("--start", default="2026-04-01", help="First day (YYYY-MM-DD).")
    p.add_argument("--end", default="2026-06-18", help="Last day (YYYY-MM-DD).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    rows = generate_rows(rng, start, end)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.filename)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} rows to {output_path} "
          f"({start} -> {end}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
