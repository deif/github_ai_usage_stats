#!/usr/bin/env python3
"""
AI Usage Dashboard generator (Bokeh).

Reads one or more GitHub Copilot "AI Usage Report" CSV files, de-duplicates
overlapping/identical rows, and produces a single self-contained, interactive
HTML dashboard (pan / zoom / hover / legend) describing AI model usage.

Usage examples
--------------
    # Use every *.csv in the current folder, write dashboard.html
    python generate_dashboard.py

    # Point at a directory of reports, custom output file
    python generate_dashboard.py --input ./reports --output dashboard.html

    # Anonymized, top 15 items in "top N" charts
    python generate_dashboard.py --anonymize --top 15
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from math import pi
from typing import List

import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    CustomJS,
    Div,
    FactorRange,
    HoverTool,
    Legend,
    LinearAxis,
    LinearColorMapper,
    NumeralTickFormatter,
    Range1d,
    RangeSlider,
)
from bokeh.palettes import Category10, Category20, Category20b
from bokeh.plotting import figure, save
from bokeh.resources import INLINE
from bokeh.transform import cumsum, dodge

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Columns that together uniquely identify a single usage record. Two rows that
# match on all of these are considered the same observation and only counted
# once, even if they appear in several overlapping report files.
KEY_COLUMNS = [
    "date",
    "username",
    "product",
    "sku",
    "model",
    "organization",
    "repository",
    "cost_center_name",
]

NUMERIC_COLUMNS = [
    "quantity",
    "applied_cost_per_quantity",
    "gross_amount",
    "discount_amount",
    "net_amount",
    "total_monthly_quota",
    "aic_quantity",
    "aic_gross_amount",
]

DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# Flat base plan fee added on top of metered net spend in the projection.
BASE_MONTHLY_COST = 1900.0


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def find_csv_files(input_path: str) -> List[str]:
    """Return a sorted list of CSV files from a file or directory path."""
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.csv")))
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        # Allow glob patterns such as "reports/*.csv".
        files = sorted(glob.glob(input_path))
    return files


def load_data(files: List[str]) -> pd.DataFrame:
    """Load and concatenate all CSV files, then de-duplicate."""
    frames = []
    for path in files:
        try:
            df = pd.read_csv(path, dtype=str)
            df["__source_file"] = os.path.basename(path)
            frames.append(df)
            print(f"  loaded {len(df):>6} rows from {os.path.basename(path)}")
        except Exception as exc:  # noqa: BLE001 - surface any read problem
            print(f"  WARNING: could not read {path}: {exc}", file=sys.stderr)

    if not frames:
        raise SystemExit("No readable CSV data found.")

    raw = pd.concat(frames, ignore_index=True)
    total_before = len(raw)

    # 1) Drop fully identical rows (same data appearing in multiple files).
    #    Ignore the helper source-file column when comparing.
    compare_cols = [c for c in raw.columns if c != "__source_file"]
    deduped = raw.drop_duplicates(subset=compare_cols, keep="first")
    exact_dropped = total_before - len(deduped)

    # 2) Collapse remaining overlaps that share the same natural key. These can
    #    occur when reports overlap in time; keep the row with the largest
    #    quantity (the most complete observation for that key/day).
    key_present = [c for c in KEY_COLUMNS if c in deduped.columns]
    if key_present:
        deduped = deduped.copy()
        deduped["__q"] = pd.to_numeric(
            deduped.get("quantity"), errors="coerce"
        ).fillna(0)
        deduped = (
            deduped.sort_values("__q")
            .drop_duplicates(subset=key_present, keep="last")
            .drop(columns="__q")
        )
    key_dropped = total_before - exact_dropped - len(deduped)

    print(
        f"  de-duplicated: {total_before} -> {len(deduped)} rows "
        f"({exact_dropped} identical, {key_dropped} overlapping removed)"
    )

    return _coerce_types(deduped)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and numeric columns; add helper time columns."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["day_of_week"] = df["date"].dt.day_name()
    df["day"] = df["date"].dt.date

    # Simplify model names into a "family" for higher-level grouping.
    df["model_family"] = df["model"].apply(_model_family)
    df["is_auto"] = df["model"].str.startswith("Auto:", na=False)
    return df


def _model_family(model: str) -> str:
    """Bucket a model name into a coarse family for summary charts."""
    if not isinstance(model, str):
        return "Other"
    m = model.replace("Auto:", "").strip().lower()
    if "opus" in m:
        return "Claude Opus"
    if "sonnet" in m:
        return "Claude Sonnet"
    if "haiku" in m:
        return "Claude Haiku"
    if "codex" in m:
        return "GPT Codex"
    if "gpt" in m:
        return "GPT"
    if "review" in m:
        return "Code Review"
    return "Other"


def anonymize_users(df: pd.DataFrame, outdir: str) -> pd.DataFrame:
    """Replace usernames with stable pseudonyms (User 01, User 02, ...).

    The mapping is deterministic: users are ranked by total credits consumed,
    so the same person always gets the same pseudonym across every chart and
    can be traced between graphs without revealing their identity. A key file
    mapping pseudonym -> real username is written next to the charts so the
    mapping can be reversed by an authorized owner if ever needed.
    """
    order = (
        df.groupby("username")["quantity"].sum().sort_values(ascending=False).index
    )
    width = max(2, len(str(len(order))))
    mapping = {
        name: f"User {str(i + 1).zfill(width)}" for i, name in enumerate(order)
    }

    df = df.copy()
    df["username"] = df["username"].map(mapping)

    key_path = os.path.join(outdir, "anonymization_key.csv")
    os.makedirs(outdir, exist_ok=True)
    (
        pd.DataFrame(
            {"pseudonym": list(mapping.values()), "username": list(mapping.keys())}
        ).to_csv(key_path, index=False)
    )
    print(f"  anonymized {len(mapping)} users; key written to {key_path}")
    return df


def monthly_spend_projection(
    df: pd.DataFrame,
    amount_col: str = "net_amount",
    fixed_monthly_cost: float = 0.0,
) -> dict:
    """Build cumulative spend for the latest month plus a month-end projection.

    Uses ``net_amount`` (the amount actually billed) by default. The daily
    run-rate is measured only over days that had real billed spend: if the
    start of the month shows $0 (e.g. a license/plan change), those leading
    zero days are excluded from the rate so the projection is not dragged
    down. The rate is then extended linearly to the end of the month.

    ``fixed_monthly_cost`` is a flat amount (e.g. the base plan fee) added on
    top of the metered spend. It is added to every displayed cumulative and
    projected value but is excluded from the run-rate calculation, so it does
    not distort the per-day slope.
    """
    import calendar

    latest = df["date"].max()
    year, month = latest.year, latest.month
    last_day = latest.day
    days_in_month = calendar.monthrange(year, month)[1]

    cur = df[(df["date"].dt.year == year) & (df["date"].dt.month == month)]
    daily = (
        cur.groupby(cur["date"].dt.day)[amount_col]
        .sum()
        .reindex(range(1, last_day + 1), fill_value=0.0)
    )
    cumulative = daily.cumsum()
    metered_to_date = float(cumulative.iloc[-1])

    # Measure the run-rate only from the first day that had real billed spend,
    # so leading $0 days (e.g. a mid-month plan change) don't deflate it.
    nonzero_days = daily.index[daily > 0]
    first_spend_day = int(nonzero_days.min()) if len(nonzero_days) else last_day
    rate_days = max(last_day - first_spend_day + 1, 1)
    daily_rate = metered_to_date / rate_days
    remaining_days = days_in_month - last_day

    # Add the flat monthly cost on top of the metered spend for display.
    cumulative = cumulative + fixed_monthly_cost
    spent_to_date = metered_to_date + fixed_monthly_cost
    projected_total = spent_to_date + daily_rate * remaining_days

    # Projection line: from today's cumulative value out to month-end.
    proj_days = list(range(last_day, days_in_month + 1))
    proj_values = [
        spent_to_date + daily_rate * (d - last_day) for d in proj_days
    ]

    return {
        "month_label": latest.strftime("%B %Y"),
        "days": list(cumulative.index),
        "cumulative": list(cumulative.values),
        "proj_days": proj_days,
        "proj_values": proj_values,
        "days_in_month": days_in_month,
        "last_day": last_day,
        "spent_to_date": spent_to_date,
        "daily_rate": float(daily_rate),
        "first_spend_day": first_spend_day,
        "fixed_monthly_cost": float(fixed_monthly_cost),
        "projected_total": float(projected_total),
    }


FULL_W = 1320
HALF_W = 650
THIRD_W = 432
ROW_H = 360

# "Top N" bar panels default to this many rows; a slider can reveal up to each
# panel's full item count (embedded into the HTML for client-side slicing).
DEFAULT_TOP = 10


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def distinct_colors(n: int) -> List[str]:
    """Return n visually distinct hex colors (up to 40 via Category20+20b)."""
    palette = list(Category20[20]) + list(Category20b[20])
    if n <= 2:
        return list(Category10[3])[:n]
    if n <= 20:
        return list(Category20[max(3, n)])[:n]
    return palette[:n]


def _attach_top_slider(slider, full_src, view_src, y_range, label_field):
    """Wire a RangeSlider to show the ranked interval [lo, hi] of ``full_src``
    (sorted largest-first) in ``view_src`` and update the categorical
    ``y_range`` factors."""
    cb = CustomJS(
        args=dict(full=full_src, view=view_src, yr=y_range, slider=slider),
        code=f"""
        const f = full.data;
        const total = f['{label_field}'].length;
        let lo = Math.round(slider.value[0]) - 1;   // 1-based -> 0-based
        let hi = Math.round(slider.value[1]);
        lo = Math.max(0, lo);
        hi = Math.min(total, hi);
        if (hi <= lo) hi = lo + 1;
        const nd = {{}};
        for (const key in f) {{ nd[key] = f[key].slice(lo, hi); }}
        view.data = nd;
        yr.factors = nd['{label_field}'].slice().reverse();
        """,
    )
    slider.js_on_change("value", cb)


# ---------------------------------------------------------------------------
# Individual panels
# ---------------------------------------------------------------------------

def panel_models_by_credits(df, slider, max_n=None, default_n=DEFAULT_TOP):
    usage = (
        df.groupby("model")["quantity"].sum().sort_values(ascending=False)
    )
    if max_n is not None:
        usage = usage.head(max_n)
    models = list(usage.index)
    values = list(usage.values)
    full = ColumnDataSource(dict(model=models, value=values))
    d = min(default_n, len(models))
    view = ColumnDataSource(dict(model=models[:d], value=values[:d]))
    yr = FactorRange(*models[:d][::-1])
    p = figure(
        y_range=yr, height=ROW_H, width=HALF_W,
        title="Most popular models (by credits)",
        toolbar_location="above", tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    p.hbar(y="model", right="value", height=0.7, source=view, color="#4C72B0")
    p.add_tools(HoverTool(tooltips=[("Model", "@model"), ("Credits", "@value{0,0}")]))
    p.xaxis.formatter = NumeralTickFormatter(format="0,0")
    p.xaxis.axis_label = "Credits"
    _attach_top_slider(slider, full, view, yr, "model")
    return p


def panel_models_by_records(df, slider, max_n=None, default_n=DEFAULT_TOP):
    rec = df.groupby("model").size().sort_values(ascending=False)
    if max_n is not None:
        rec = rec.head(max_n)
    models = list(rec.index)
    values = list(rec.values)
    full = ColumnDataSource(dict(model=models, value=values))
    d = min(default_n, len(models))
    view = ColumnDataSource(dict(model=models[:d], value=values[:d]))
    yr = FactorRange(*models[:d][::-1])
    p = figure(
        y_range=yr, height=ROW_H, width=HALF_W,
        title="Most popular models (by usage records)",
        toolbar_location="above", tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    p.hbar(y="model", right="value", height=0.7, source=view, color="#55A868")
    p.add_tools(HoverTool(tooltips=[("Model", "@model"), ("Records", "@value{0,0}")]))
    p.xaxis.axis_label = "Usage records"
    _attach_top_slider(slider, full, view, yr, "model")
    return p


def panel_active_weekdays(df):
    by_day = df.groupby("day_of_week")["quantity"].sum().reindex(DAY_ORDER).fillna(0)
    colors = ["#4C72B0"] * 5 + ["#DD8452"] * 2
    src = ColumnDataSource(dict(day=DAY_ORDER, value=list(by_day.values), color=colors))
    p = figure(
        x_range=DAY_ORDER, height=ROW_H, width=HALF_W,
        title="Most active days of the week",
        toolbar_location="above", tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    p.vbar(x="day", top="value", width=0.8, source=src, color="color")
    p.add_tools(HoverTool(tooltips=[("Day", "@day"), ("Credits", "@value{0,0}")]))
    p.yaxis.formatter = NumeralTickFormatter(format="0,0")
    p.yaxis.axis_label = "Total credits"
    p.xaxis.major_label_orientation = 0.5
    return p


def panel_credits_histogram(df):
    data = df[df["quantity"] > 0]["quantity"].values
    hist, edges = np.histogram(data, bins=40)
    src = ColumnDataSource(dict(top=hist, left=edges[:-1], right=edges[1:]))
    p = figure(
        height=ROW_H, width=HALF_W,
        title="Distribution of credits per usage record",
        toolbar_location="above", tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    p.quad(top="top", bottom=0, left="left", right="right", source=src,
           fill_color="#C44E52", line_color="white")
    p.add_tools(HoverTool(tooltips=[
        ("Range", "@left{0,0} – @right{0,0}"), ("Records", "@top")
    ]))
    p.xaxis.axis_label = "Credits per record"
    p.yaxis.axis_label = "Number of records"
    return p


def panel_usage_over_time(df, x_range=None):
    daily = df.groupby("day").agg(
        credits=("quantity", "sum"), active_users=("username", "nunique")
    )
    daily.index = pd.to_datetime(daily.index)
    src = ColumnDataSource(dict(
        day=list(daily.index), credits=list(daily["credits"]),
        users=list(daily["active_users"]),
    ))
    fig_kwargs = dict(
        x_axis_type="datetime", height=ROW_H, width=FULL_W,
        title="Daily usage over time (credits & active users)",
        toolbar_location="above", tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    if x_range is not None:
        fig_kwargs["x_range"] = x_range
    p = figure(**fig_kwargs)
    r1 = p.line("day", "credits", source=src, color="#4C72B0", line_width=2,
                legend_label="Credits")
    p.scatter("day", "credits", source=src, color="#4C72B0", size=5)
    p.yaxis.axis_label = "Credits consumed"
    p.yaxis.formatter = NumeralTickFormatter(format="0,0")

    # Second y-axis for active users.
    from bokeh.models import LinearAxis, Range1d
    p.extra_y_ranges = {"users": Range1d(start=0, end=max(daily["active_users"]) * 1.2)}
    p.add_layout(LinearAxis(y_range_name="users", axis_label="Active users"), "right")
    r2 = p.line("day", "users", source=src, color="#DD8452", line_width=2,
                y_range_name="users", legend_label="Active users")
    p.scatter("day", "users", source=src, color="#DD8452", size=5,
              y_range_name="users")

    p.add_tools(HoverTool(tooltips=[
        ("Date", "@day{%F}"), ("Credits", "@credits{0,0}"),
        ("Active users", "@users"),
    ], formatters={"@day": "datetime"}, mode="vline", renderers=[r1]))
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    return p


def _stacked_area_over_time(df, column_name, title, top=None, x_range=None):
    """Build a stacked-area figure of daily usage records per category."""
    if top is not None:
        keep = df.groupby(column_name).size().sort_values(ascending=False).head(top).index
        cats = df[column_name].where(df[column_name].isin(keep), "Other")
    else:
        cats = df[column_name]
    work = df.assign(_cat=cats)
    pivot = work.pivot_table(
        index="day", columns="_cat", values="username", aggfunc="size", fill_value=0
    )
    order = pivot.sum().sort_values(ascending=False).index.tolist()
    pivot = pivot[order]
    pivot.index = pd.to_datetime(pivot.index)

    data = {"day": list(pivot.index)}
    for c in order:
        data[c] = list(pivot[c].values)
    src = ColumnDataSource(data)
    colors = distinct_colors(len(order))

    fig_kwargs = dict(
        x_axis_type="datetime", height=ROW_H + 40, width=FULL_W, title=title,
        toolbar_location="above", tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    if x_range is not None:
        fig_kwargs["x_range"] = x_range
    p = figure(**fig_kwargs)
    renderers = p.varea_stack(
        stackers=order, x="day", color=colors, source=src,
    )
    # Build a legend that sits outside the plot to avoid covering data.
    legend = Legend(
        items=[(cat, [r]) for cat, r in zip(order, renderers)],
        location="center",
    )
    p.add_layout(legend, "right")
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "8pt"
    p.yaxis.axis_label = "Usage records per day"
    return p


def panels_top_users(df, slider, max_n=None, default_n=DEFAULT_TOP):
    """Build the credits + cost user panels sharing one categorical y-axis,
    both driven by the slider with a single callback."""
    credits = (
        df.groupby("username")["quantity"].sum().sort_values(ascending=False)
    )
    if max_n is not None:
        credits = credits.head(max_n)
    users = list(credits.index)
    cred_vals = list(credits.values)
    cost = (
        df.groupby("username")[["gross_amount", "net_amount"]]
        .sum().reindex(users).fillna(0.0)
    )
    gross = list(cost["gross_amount"])
    net = list(cost["net_amount"])
    d = min(default_n, len(users))

    yr = FactorRange(*users[:d][::-1])  # shared by both panels

    cred_full = ColumnDataSource(dict(user=users, value=cred_vals))
    cred_view = ColumnDataSource(dict(user=users[:d], value=cred_vals[:d]))
    p1 = figure(
        y_range=yr, height=ROW_H, width=HALF_W,
        title="Top users by credits",
        toolbar_location="above", tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    p1.hbar(y="user", right="value", height=0.7, source=cred_view, color="#8172B3")
    p1.add_tools(HoverTool(tooltips=[("User", "@user"), ("Credits", "@value{0,0}")]))
    p1.xaxis.formatter = NumeralTickFormatter(format="0,0")
    p1.xaxis.axis_label = "Credits"

    cost_full = ColumnDataSource(dict(user=users, gross=gross, net=net))
    cost_view = ColumnDataSource(dict(user=users[:d], gross=gross[:d], net=net[:d]))
    p2 = figure(
        y_range=yr, height=ROW_H, width=HALF_W,
        title="Top users by cost",
        toolbar_location="above", tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    p2.hbar(y=dodge("user", 0.18, range=yr), right="gross", height=0.32,
            source=cost_view, color="#A0AEC0", legend_label="Gross")
    p2.hbar(y=dodge("user", -0.18, range=yr), right="net", height=0.32,
            source=cost_view, color="#C44E52", legend_label="Net")
    p2.add_tools(HoverTool(tooltips=[
        ("User", "@user"), ("Gross", "@gross{0,0.0}"), ("Net", "@net{0,0.0}")
    ]))
    p2.xaxis.axis_label = "Cost"
    p2.legend.location = "bottom_right"

    cb = CustomJS(
        args=dict(cf=cred_full, cv=cred_view, kf=cost_full, kv=cost_view,
                  yr=yr, slider=slider),
        code="""
        const total = cf.data['user'].length;
        let lo = Math.round(slider.value[0]) - 1;   // 1-based -> 0-based
        let hi = Math.round(slider.value[1]);
        lo = Math.max(0, lo);
        hi = Math.min(total, hi);
        if (hi <= lo) hi = lo + 1;
        const nc = {}; for (const key in cf.data) nc[key] = cf.data[key].slice(lo, hi);
        cv.data = nc;
        const nk = {}; for (const key in kf.data) nk[key] = kf.data[key].slice(lo, hi);
        kv.data = nk;
        yr.factors = cf.data['user'].slice(lo, hi).reverse();
        """,
    )
    slider.js_on_change("value", cb)
    return p1, p2


def panel_top_records(df, slider, max_n=None, default_n=DEFAULT_TOP):
    if max_n is None:
        max_n = len(df)
    recs = df.nlargest(max_n, "quantity")[["date", "username", "model", "quantity"]]
    # Rank prefix keeps every category label unique (a user can have several
    # records with the same model/date), which a Bokeh categorical axis requires.
    labels = [
        f"#{i} · {r.username} • {r.model} • {r.date.date()}"
        for i, r in enumerate(recs.itertuples(), start=1)
    ]
    values = list(recs["quantity"])
    user = list(recs["username"])
    model = list(recs["model"])
    date = [str(d.date()) for d in recs["date"]]
    full = ColumnDataSource(dict(
        label=labels, value=values, user=user, model=model, date=date,
    ))
    d = min(default_n, len(labels))
    view = ColumnDataSource(dict(
        label=labels[:d], value=values[:d], user=user[:d],
        model=model[:d], date=date[:d],
    ))
    yr = FactorRange(*labels[:d][::-1])
    p = figure(
        y_range=yr, height=ROW_H, width=FULL_W,
        title="Top usage records by credits",
        toolbar_location="above", tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    p.hbar(y="label", right="value", height=0.7, source=view, color="#C44E52")
    p.add_tools(HoverTool(tooltips=[
        ("User", "@user"), ("Model", "@model"),
        ("Date", "@date"), ("Credits", "@value{0,0}"),
    ]))
    p.xaxis.formatter = NumeralTickFormatter(format="0,0")
    p.xaxis.axis_label = "Credits (single record)"
    _attach_top_slider(slider, full, view, yr, "label")
    return p


def panel_weekday_model_heatmap(df, top):
    top_models = (
        df.groupby("model")["quantity"].sum().sort_values(ascending=False).head(top).index
    )
    pivot = (
        df[df["model"].isin(top_models)]
        .pivot_table(index="model", columns="day_of_week", values="quantity",
                     aggfunc="sum", fill_value=0)
        .reindex(columns=DAY_ORDER).fillna(0)
    )
    models = pivot.sum(axis=1).sort_values().index.tolist()
    rows = []
    for m in models:
        for d in DAY_ORDER:
            rows.append((m, d, float(pivot.loc[m, d])))
    src = ColumnDataSource(dict(
        model=[r[0] for r in rows], day=[r[1] for r in rows],
        value=[r[2] for r in rows],
    ))
    mapper = LinearColorMapper(
        palette="Viridis256", low=0, high=max(r[2] for r in rows) or 1
    )
    p = figure(
        x_range=DAY_ORDER, y_range=models, height=ROW_H + 60, width=FULL_W,
        title="Model usage by day of week (credits)",
        toolbar_location="above", tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    p.rect(x="day", y="model", width=1, height=1, source=src,
           fill_color={"field": "value", "transform": mapper}, line_color="white")
    p.add_tools(HoverTool(tooltips=[
        ("Model", "@model"), ("Day", "@day"), ("Credits", "@value{0,0}")
    ]))
    p.add_layout(ColorBar(color_mapper=mapper, title="Credits"), "right")
    return p


def _pie(series, title, palette=None, width=THIRD_W, color_map=None):
    data = series.reset_index()
    data.columns = ["label", "value"]
    data["angle"] = data["value"] / data["value"].sum() * 2 * pi
    n = len(data)
    if color_map is not None:
        data["color"] = [color_map[label] for label in data["label"]]
    else:
        if palette is None:
            palette = distinct_colors(n)
        data["color"] = palette[:n]
    data["pct"] = data["value"] / data["value"].sum() * 100
    src = ColumnDataSource(data)
    p = figure(
        height=ROW_H, width=width, title=title, toolbar_location="above",
        tools="save", x_range=(-1.1, 1.6),
    )
    p.wedge(
        x=0, y=0, radius=0.85,
        start_angle=cumsum("angle", include_zero=True), end_angle=cumsum("angle"),
        line_color="white", fill_color="color", legend_field="label", source=src,
    )
    p.add_tools(HoverTool(tooltips=[
        ("", "@label"), ("Value", "@value{0,0}"), ("Share", "@pct{0.0}%")
    ]))
    p.axis.visible = False
    p.grid.grid_line_color = None
    p.legend.label_text_font_size = "8pt"
    p.legend.location = "center_right"
    return p


def _family_color_map(df):
    """Stable model-family -> color map shared by the family pie charts."""
    families = (
        df.groupby("model_family")["quantity"].sum().sort_values(ascending=False).index
    )
    colors = distinct_colors(len(families))
    return {fam: colors[i] for i, fam in enumerate(families)}


def panel_family_credit_pie(df, color_map=None):
    fam = df.groupby("model_family")["quantity"].sum().sort_values(ascending=False)
    return _pie(fam, "Credit share by model family", color_map=color_map)


def panel_family_record_pie(df, color_map=None):
    fam = df.groupby("model_family").size().sort_values(ascending=False)
    return _pie(fam, "Usage-record share by model family", color_map=color_map)


def panel_auto_vs_manual(df):
    split = df.assign(
        selection=df["is_auto"].map({True: "Auto-selected", False: "Manually chosen"})
    ).groupby("selection")["quantity"].sum().sort_values(ascending=False)
    return _pie(split, "Auto-selected vs. manually chosen", palette=["#64B5CD", "#CCB974"])


def panel_spend_projection(df):
    proj = monthly_spend_projection(df, fixed_monthly_cost=BASE_MONTHLY_COST)
    actual = ColumnDataSource(dict(day=proj["days"], value=proj["cumulative"]))
    projected = ColumnDataSource(dict(day=proj["proj_days"], value=proj["proj_values"]))
    endpoint = ColumnDataSource(dict(
        day=[proj["days_in_month"]], value=[proj["projected_total"]],
        label=[f"Projected month-end: {proj['projected_total']:,.0f}"],
    ))

    p = figure(
        height=ROW_H, width=FULL_W,
        title=(
            f"Monthly net-spend projection — {proj['month_label']} "
            f"(day {proj['last_day']} of {proj['days_in_month']}, "
            f"run-rate {proj['daily_rate']:,.0f}/day)"
        ),
        toolbar_location="above", tools="pan,box_zoom,wheel_zoom,reset,save",
        x_range=(1, proj["days_in_month"]),
    )
    r1 = p.line("day", "value", source=actual, color="#4C72B0", line_width=3,
                legend_label="Cumulative net spend to date (incl. base fee)")
    p.scatter("day", "value", source=actual, color="#4C72B0", size=6)
    r2 = p.line("day", "value", source=projected, color="#DD8452", line_width=3,
                line_dash="dashed", legend_label="Projected (run-rate)")
    p.scatter("day", "value", source=endpoint, color="#DD8452", size=11,
              marker="diamond")
    from bokeh.models import LabelSet
    p.add_layout(LabelSet(
        x="day", y="value", text="label", source=endpoint,
        x_offset=-8, y_offset=8, text_align="right",
        text_font_style="bold", text_color="#B5562E", text_font_size="11pt",
    ))
    p.add_tools(HoverTool(tooltips=[("Day", "@day"), ("Cumulative", "@value{0,0}")],
                          renderers=[r1, r2], mode="vline"))
    p.xaxis.axis_label = "Day of month"
    p.yaxis.axis_label = "Cumulative net spend (USD)"
    p.yaxis.formatter = NumeralTickFormatter(format="0,0")
    p.legend.location = "top_left"

    notes = []
    if proj.get("fixed_monthly_cost", 0):
        notes.append(
            f"Includes flat base plan fee of {proj['fixed_monthly_cost']:,.0f} "
            f"USD on top of metered net spend."
        )
    if proj.get("first_spend_day", 1) > 1:
        notes.append(
            f"Run-rate based on net spend from day {proj['first_spend_day']} "
            f"onward (earlier days had no billed spend)."
        )
    if notes:
        caption = Div(
            text="<p style='font-family:sans-serif;color:#555;margin:0 0 8px 4px'>"
            + " ".join(notes) + "</p>",
            width=FULL_W,
        )
        return column(p, caption)
    return p


def panel_credits_net_gross_projection(df):
    """Cumulative credits, net spend and gross spend for the current month,
    each with its own month-end run-rate projection. Credits (a count) share
    the left axis; net/gross spend (both USD) share the right axis.
    """
    proj_credits = monthly_spend_projection(df, amount_col="quantity")
    proj_net = monthly_spend_projection(df, amount_col="net_amount")
    proj_gross = monthly_spend_projection(df, amount_col="gross_amount")

    credits_actual = ColumnDataSource(dict(day=proj_credits["days"], value=proj_credits["cumulative"]))
    credits_proj = ColumnDataSource(dict(day=proj_credits["proj_days"], value=proj_credits["proj_values"]))
    net_actual = ColumnDataSource(dict(day=proj_net["days"], value=proj_net["cumulative"]))
    net_proj = ColumnDataSource(dict(day=proj_net["proj_days"], value=proj_net["proj_values"]))
    gross_actual = ColumnDataSource(dict(day=proj_gross["days"], value=proj_gross["cumulative"]))
    gross_proj = ColumnDataSource(dict(day=proj_gross["proj_days"], value=proj_gross["proj_values"]))

    p = figure(
        height=ROW_H, width=FULL_W,
        title=(
            f"Monthly credits & spend projection — {proj_credits['month_label']} "
            f"(day {proj_credits['last_day']} of {proj_credits['days_in_month']})"
        ),
        toolbar_location="above", tools="pan,box_zoom,wheel_zoom,reset,save",
        x_range=(1, proj_credits["days_in_month"]),
    )

    # Right-hand axis (USD) for net/gross spend; left (default) axis is credits.
    max_usd = max(proj_net["projected_total"], proj_gross["projected_total"]) or 1
    p.extra_y_ranges = {"usd": Range1d(start=0, end=max_usd * 1.15)}
    p.add_layout(LinearAxis(y_range_name="usd", axis_label="Cumulative spend (USD)"), "right")

    r1 = p.line("day", "value", source=credits_actual, color="#4C72B0", line_width=3,
                legend_label="Credits used to date")
    p.scatter("day", "value", source=credits_actual, color="#4C72B0", size=6)
    r2 = p.line("day", "value", source=credits_proj, color="#4C72B0", line_width=3,
                line_dash="dashed", legend_label="Credits projected")

    r3 = p.line("day", "value", source=net_actual, color="#55A868", line_width=3,
                legend_label="Net spend to date", y_range_name="usd")
    p.scatter("day", "value", source=net_actual, color="#55A868", size=6, y_range_name="usd")
    r4 = p.line("day", "value", source=net_proj, color="#55A868", line_width=3,
                line_dash="dashed", legend_label="Net spend projected", y_range_name="usd")

    r5 = p.line("day", "value", source=gross_actual, color="#C44E52", line_width=3,
                legend_label="Gross spend to date", y_range_name="usd")
    p.scatter("day", "value", source=gross_actual, color="#C44E52", size=6, y_range_name="usd")
    r6 = p.line("day", "value", source=gross_proj, color="#C44E52", line_width=3,
                line_dash="dashed", legend_label="Gross spend projected", y_range_name="usd")

    p.add_tools(HoverTool(
        tooltips=[("Day", "@day"), ("Value", "@value{0,0}")],
        renderers=[r1, r2, r3, r4, r5, r6], mode="vline",
    ))
    p.xaxis.axis_label = "Day of month"
    p.yaxis[0].axis_label = "Cumulative credits used"
    p.yaxis.formatter = NumeralTickFormatter(format="0,0")
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    caption = Div(
        text=(
            "<p style='font-family:sans-serif;color:#555;margin:0 0 8px 4px'>"
            f"Projected month-end — Credits: {proj_credits['projected_total']:,.0f} &middot; "
            f"Net: {proj_net['projected_total']:,.0f} USD &middot; "
            f"Gross: {proj_gross['projected_total']:,.0f} USD</p>"
        ),
        width=FULL_W,
    )
    return column(p, caption)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_dashboard(df, output_path, top):
    n_users = df["username"].nunique()
    per_user_spend = df.groupby("username")["net_amount"].sum()
    avg_spend_per_user = per_user_spend.mean() if n_users else 0.0
    median_spend_per_user = per_user_spend.median() if n_users else 0.0
    max_spend_per_user = per_user_spend.max() if n_users else 0.0
    min_spend_per_user = per_user_spend.min() if n_users else 0.0
    header = Div(
        text="<h1 style='font-family:sans-serif;margin:0'>AI Usage Dashboard</h1>"
        f"<p style='font-family:sans-serif;color:#555'>"
        f"{len(df):,} records &middot; "
        f"{df['date'].min().date()} → {df['date'].max().date()} &middot; "
        f"{n_users} users &middot; "
        f"{df['model'].nunique()} models &middot; "
        f"Avg net spend/user: {avg_spend_per_user:,.2f} USD &middot; "
        f"Median: {median_spend_per_user:,.2f} USD &middot; "
        f"Max: {max_spend_per_user:,.2f} USD &middot; "
        f"Min: {min_spend_per_user:,.2f} USD</p>",
        width=FULL_W,
    )

    # A single range slider selects which slice of the ranking every "Top N"
    # bar panel shows, e.g. ranks 1-10 (default) or 90-100. Its maximum is the
    # largest categorical dimension (distinct users/models) so those charts can
    # be expanded to any window up to every item.
    slider_max = max(
        df["model"].nunique(),
        df["username"].nunique(),
        1,
    )
    default_hi = min(DEFAULT_TOP, slider_max)
    top_slider = RangeSlider(
        start=1, end=slider_max, value=(1, default_hi), step=1,
        title="Rank interval shown in ranked bar charts", width=FULL_W,
    )

    # Time-series panels share a common datetime x-axis so zooming/panning
    # one of them moves the others in lockstep.
    usage_ts = panel_usage_over_time(df)
    models_ts = _stacked_area_over_time(
        df, "model", "Model usage over time (by usage records)", top,
        x_range=usage_ts.x_range,
    )
    family_ts = _stacked_area_over_time(
        df, "model_family",
        "Model family usage over time (by usage records)",
        x_range=usage_ts.x_range,
    )

    # Top-user panels share the categorical y-axis (same users, same order)
    # so vertical scroll/zoom stays aligned, and both follow the slider.
    users_credits, users_cost = panels_top_users(df, top_slider)

    # Shared color map so each model family keeps one color across both pies.
    family_colors = _family_color_map(df)

    grid = column(
        top_slider,
        row(panel_models_by_credits(df, top_slider),
            panel_models_by_records(df, top_slider)),
        row(panel_active_weekdays(df), panel_credits_histogram(df)),
        usage_ts,
        models_ts,
        family_ts,
        panel_spend_projection(df),
        panel_credits_net_gross_projection(df),
        row(users_credits, users_cost),
        panel_top_records(df, top_slider, max_n=slider_max),
        panel_weekday_model_heatmap(df, top),
        row(
            panel_family_credit_pie(df, color_map=family_colors),
            panel_family_record_pie(df, color_map=family_colors),
            panel_auto_vs_manual(df),
        ),
    )

    layout = column(header, grid)
    save(layout, filename=output_path, resources=INLINE, title="AI Usage Dashboard")
    print(f"  wrote {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an interactive Bokeh AI usage dashboard from CSV reports.",
    )
    parser.add_argument("--input", "-i", default=".",
                        help="CSV file, directory, or glob pattern (default: current dir).")
    parser.add_argument("--output", "-o", default="dashboard.html",
                        help="Output HTML file (default: dashboard.html).")
    parser.add_argument("--top", "-t", type=int, default=10,
                        help="How many items to show in 'top N' charts (default: 10).")
    parser.add_argument("--anonymize", "-a", action="store_true",
                        help="Replace usernames with stable pseudonyms.")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    files = find_csv_files(args.input)
    if not files:
        print(f"No CSV files found at '{args.input}'.", file=sys.stderr)
        return 1

    print(f"Found {len(files)} CSV file(s):")
    df = load_data(files)

    if args.anonymize:
        outdir = os.path.dirname(os.path.abspath(args.output))
        df = anonymize_users(df, outdir)

    print(f"\nBuilding interactive dashboard '{args.output}'...")
    build_dashboard(df, args.output, args.top)
    print("Done. Open the HTML file in a browser.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
