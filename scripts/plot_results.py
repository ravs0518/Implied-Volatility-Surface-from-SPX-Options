#!/usr/bin/env python3
"""Create clear Matplotlib plots from the project CSV outputs.

Examples:
    python3 scripts/plot_results.py
    python3 scripts/plot_results.py --plot-date 2023-08-15 --plot-expire-date 2023-12-29
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter


DTE_BUCKETS = (
    (3.0, 10.0, "3–10"),
    (10.0, 20.0, "10–20"),
    (20.0, 40.0, "20–40"),
    (40.0, 60.0, "40–60"),
    (60.0, 90.0, "60–90"),
    (90.0, float("inf"), "90+") ,
)


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def sample_std(values):
    if len(values) < 2:
        return float("nan")
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def bucket_for_dte(dte):
    for lower, upper, label in DTE_BUCKETS:
        if lower <= dte < upper:
            return lower, upper, label
    return None


def make_dte_bucket_analysis(rows, csv_output: Path, plot_output: Path):
    buckets = defaultdict(lambda: {"options": [], "sr1": [], "difference": [], "r_squared": []})
    for row in rows:
        if not row.get("sr1_rate_pct") or not row.get("difference_bps"):
            continue
        bucket = bucket_for_dte(float(row["dte"]))
        if bucket is None:
            continue
        key = bucket[2]
        buckets[key]["options"].append(float(row["options_rate_pct"]))
        buckets[key]["sr1"].append(float(row["sr1_rate_pct"]))
        buckets[key]["difference"].append(float(row["difference_bps"]))
        if row.get("options_r_squared"):
            buckets[key]["r_squared"].append(float(row["options_r_squared"]))

    summary = []
    for lower, upper, label in DTE_BUCKETS:
        values = buckets[label]
        if not values["options"]:
            continue
        summary.append({
            "dte_bucket": label,
            "dte_min": lower,
            "dte_max": "" if upper == float("inf") else upper,
            "observations": len(values["options"]),
            "options_mean_pct": sum(values["options"]) / len(values["options"]),
            "options_std_pct": sample_std(values["options"]),
            "sr1_mean_pct": sum(values["sr1"]) / len(values["sr1"]),
            "sr1_std_pct": sample_std(values["sr1"]),
            "mean_difference_bps": sum(values["difference"]) / len(values["difference"]),
            "difference_std_bps": sample_std(values["difference"]),
            "mean_r_squared": sum(values["r_squared"]) / len(values["r_squared"]) if values["r_squared"] else "",
        })

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(summary[0]) if summary else ["dte_bucket"]
    with csv_output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)

    if not summary:
        raise ValueError("No matched options/SR1 observations available for DTE buckets")

    labels = [row["dte_bucket"] for row in summary]
    x = list(range(len(labels)))
    options_mean = [row["options_mean_pct"] for row in summary]
    options_std = [row["options_std_pct"] for row in summary]
    sr1_mean = [row["sr1_mean_pct"] for row in summary]
    difference_mean = [row["mean_difference_bps"] for row in summary]
    difference_std = [row["difference_std_bps"] for row in summary]
    counts = [row["observations"] for row in summary]

    fig, (rate_ax, error_ax) = plt.subplots(2, 1, figsize=(11, 9), dpi=150, sharex=True)
    rate_ax.plot(x, options_mean, marker="o", linewidth=2, color="tab:blue", label="Options regression mean")
    rate_ax.fill_between(x,
                         [mean - std for mean, std in zip(options_mean, options_std)],
                         [mean + std for mean, std in zip(options_mean, options_std)],
                         color="tab:blue", alpha=0.16, label="Options ±1 standard deviation")
    rate_ax.plot(x, sr1_mean, marker="o", linewidth=2, color="tab:red", label="SR1 approximation mean")
    rate_ax.set_title("Interest-rate estimates by option time to expiration")
    rate_ax.set_ylabel("Annualized rate (%)")
    rate_ax.legend(frameon=False)

    error_ax.axhline(0, color="#555555", linewidth=0.9)
    error_ax.errorbar(x, difference_mean, yerr=difference_std, fmt="o-", capsize=4,
                      color="tab:purple", linewidth=1.8, label="Mean difference ±1 standard deviation")
    error_ax.set_xlabel("DTE bucket")
    error_ax.set_ylabel("Options − SR1 (basis points)")
    error_ax.set_xticks(x, labels)
    error_ax.legend(frameon=False)
    for xi, count in zip(x, counts):
        error_ax.annotate(f"n={count:,}", (xi, difference_mean[xi]), xytext=(0, 10),
                          textcoords="offset points", ha="center", fontsize=8)

    for ax in (rate_ax, error_ax):
        ax.minorticks_on()
        ax.grid(True, which="major", color="#b0b0b0", linewidth=0.8, alpha=0.8)
        ax.grid(True, which="minor", color="#d9d9d9", linewidth=0.5, linestyle=":", alpha=0.8)
    fig.tight_layout()
    plot_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_output, bbox_inches="tight")
    plt.close(fig)


def make_regression_plot(rows, output: Path, quote_date: str, expire_date: str):
    pairs = {}
    for row in rows:
        if row["quote_date"] != quote_date or row["expire_date"] != expire_date:
            continue
        key = row["strike"]
        pairs.setdefault(key, {})[row["option_type"]] = float(row["mid"])

    points = sorted(
        (float(strike), values["call"] - values["put"])
        for strike, values in pairs.items()
        if "call" in values and "put" in values
        and values["call"] > 0 and values["put"] > 0
    )
    if not points:
        raise ValueError(f"No matched call-put pairs for {quote_date} / {expire_date}")

    x = [point[0] for point in points]
    y = [point[1] for point in points]
    n = len(points)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    slope = sum((xi - mean_x) * (yi - mean_y) for xi, yi in points) / sum((xi - mean_x) ** 2 for xi in x)
    intercept = mean_y - slope * mean_x
    fitted = [intercept + slope * xi for xi in x]
    ss_res = sum((yi - fi) ** 2 for yi, fi in zip(y, fitted))
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    r_squared = 1 - ss_res / ss_tot if ss_tot else float("nan")

    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    ax.scatter(x, y, s=14, alpha=0.65, label="Observed pairs")
    ax.plot(x, fitted, color="crimson", linewidth=2, label="OLS fit")
    ax.set_title(f"SPX put-call parity regression: {quote_date} to {expire_date}")
    ax.set_xlabel("Strike price (K)")
    ax.set_ylabel("Call midpoint − put midpoint")
    ax.minorticks_on()
    ax.grid(True, which="major", color="#b0b0b0", linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", color="#d9d9d9", linewidth=0.5, linestyle=":", alpha=0.8)
    ax.legend(frameon=False)
    ax.text(
        0.01,
        0.99,
        f"C − P = {intercept:.2f} {slope:+.6f}K\nR² = {r_squared:.5f} | matched pairs = {n}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def make_comparison_plot(rows, output: Path, min_dte: float, max_dte: float):
    daily = defaultdict(lambda: {"options": [], "sr1": []})
    for row in rows:
        if not row["sr1_rate_pct"]:
            continue
        dte = float(row["dte"])
        if min_dte <= dte <= max_dte:
            daily[row["quote_date"]]["options"].append(float(row["options_rate_pct"]))
            daily[row["quote_date"]]["sr1"].append(float(row["sr1_rate_pct"]))

    dates = sorted(daily)
    if not dates:
        raise ValueError("No shared option/SR1 observations in the requested DTE range")
    date_values = [datetime.strptime(value, "%Y-%m-%d") for value in dates]
    option_values = [sum(daily[d]["options"]) / len(daily[d]["options"]) for d in dates]
    sr1_values = [sum(daily[d]["sr1"]) / len(daily[d]["sr1"]) for d in dates]

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.plot(date_values, option_values, color="tab:blue", linewidth=1.5, label="Options regression")
    ax.plot(date_values, sr1_values, color="tab:red", linewidth=1.8, label="SR1 spot-rate approximation")
    ax.set_title(f"Options regression versus SR1 SOFR spot-rate approximation\nDaily averages for {min_dte:.0f}–{max_dte:.0f} DTE")
    ax.set_xlabel("Quote date")
    ax.set_ylabel("Annualized rate (%)")
    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    ax.minorticks_on()
    ax.grid(True, which="major", color="#b0b0b0", linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", color="#d9d9d9", linewidth=0.5, linestyle=":", alpha=0.8)
    ax.legend(frameon=False)
    ax.text(
        0.01,
        0.99,
        f"Shared dates = {len(dates)}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def make_bucket_timeseries_plot(rows, output: Path):
    daily = defaultdict(lambda: {label: [] for _, _, label in DTE_BUCKETS} | {"sr1": []})
    for row in rows:
        if not row.get("sr1_rate_pct") or not row.get("options_rate_pct"):
            continue
        bucket = bucket_for_dte(float(row["dte"]))
        if bucket is None:
            continue
        daily[row["quote_date"]][bucket[2]].append(float(row["options_rate_pct"]))
        daily[row["quote_date"]]["sr1"].append(float(row["sr1_rate_pct"]))

    dates = sorted(daily)
    if not dates:
        raise ValueError("No shared option/SR1 observations available for DTE bucket plot")
    date_values = [datetime.strptime(value, "%Y-%m-%d") for value in dates]

    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")
    for (_, _, label), color in zip(DTE_BUCKETS, colors):
        values = []
        visible_dates = []
        for date_text, date_values_by_bucket in sorted(daily.items()):
            if date_values_by_bucket[label]:
                visible_dates.append(datetime.strptime(date_text, "%Y-%m-%d"))
                values.append(sum(date_values_by_bucket[label]) / len(date_values_by_bucket[label]))
        if visible_dates:
            ax.plot(visible_dates, values, linewidth=1.25, alpha=0.9, color=color, label=f"Options {label} DTE")

    sr1_dates = []
    sr1_values = []
    for date_text, date_values_by_bucket in sorted(daily.items()):
        if date_values_by_bucket["sr1"]:
            sr1_dates.append(datetime.strptime(date_text, "%Y-%m-%d"))
            sr1_values.append(sum(date_values_by_bucket["sr1"]) / len(date_values_by_bucket["sr1"]))
    ax.plot(sr1_dates, sr1_values, color="black", linewidth=2.2, label="SR1 spot-rate approximation")
    ax.set_title("Options regression versus SR1 approximation by DTE bucket")
    ax.set_xlabel("Quote date")
    ax.set_ylabel("Annualized rate (%)")
    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    ax.minorticks_on()
    ax.grid(True, which="major", color="#b0b0b0", linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", color="#d9d9d9", linewidth=0.5, linestyle=":", alpha=0.8)
    ax.legend(frameon=False, ncol=2)
    ax.text(
        0.01,
        0.99,
        f"Shared dates = {len(dates)} | Daily averages within each bucket",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path, default=Path("data/cleaned/spx_options_2023_filtered.csv"))
    parser.add_argument("--comparison", type=Path, default=Path("data/cleaned/options_vs_sr1_2023.csv"))
    parser.add_argument("--regression-output", type=Path, default=Path("data/cleaned/implied_rate_regression_q3_matplotlib.png"))
    parser.add_argument("--comparison-output", type=Path, default=Path("data/cleaned/options_vs_sr1_2023_matplotlib.png"))
    parser.add_argument("--bucket-timeseries-output", type=Path, default=Path("data/cleaned/options_vs_sr1_by_dte_2023.png"))
    parser.add_argument("--bucket-summary-output", type=Path, default=Path("data/cleaned/options_vs_sr1_dte_buckets_2023.csv"))
    parser.add_argument("--bucket-plot-output", type=Path, default=Path("data/cleaned/options_vs_sr1_dte_buckets_2023.png"))
    parser.add_argument("--plot-date", default="2023-08-15")
    parser.add_argument("--plot-expire-date", default="2023-12-29")
    parser.add_argument("--min-dte", type=float, default=20.0)
    parser.add_argument("--max-dte", type=float, default=60.0)
    args = parser.parse_args()

    make_regression_plot(read_rows(args.options), args.regression_output, args.plot_date, args.plot_expire_date)
    comparison_rows = read_rows(args.comparison)
    make_comparison_plot(comparison_rows, args.comparison_output, args.min_dte, args.max_dte)
    make_bucket_timeseries_plot(comparison_rows, args.bucket_timeseries_output)
    make_dte_bucket_analysis(comparison_rows, args.bucket_summary_output, args.bucket_plot_output)
    print(f"Wrote {args.regression_output}")
    print(f"Wrote {args.comparison_output}")
    print(f"Wrote {args.bucket_timeseries_output}")
    print(f"Wrote {args.bucket_summary_output}")
    print(f"Wrote {args.bucket_plot_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
