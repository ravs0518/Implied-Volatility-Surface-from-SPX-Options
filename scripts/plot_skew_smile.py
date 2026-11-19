#!/usr/bin/env python3
"""Plot representative SPX volatility skew and smile slices.

The script uses the vendor IV values from the June 30, 2023 snapshot. By
default, it selects a longer-dated slice for skew and a shorter-dated slice
for smile, reflecting the term-structure pattern visible in the surface.

Example:
    python3 scripts/plot_skew_smile.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svi_data import estimate_forward


DEFAULT_INPUT = Path("data/Q2/max_spx_options_filtered_2023-06-30.csv")
DEFAULT_OUTPUT = Path("results/volatility_skew_smile_2023-06-30.png")


def clean_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["iv"] > 0) & (df["iv"] < 2) & (df["bid"] > 0) & (df["ask"] >= df["bid"])].copy()
    return df


def slice_statistics(df: pd.DataFrame, max_dte: float) -> list[dict]:
    statistics = []
    for expiry, group in df[df["dte"] <= max_dte].groupby("expire_date"):
        group = group.copy()
        group["mid"] = (group["bid"] + group["ask"]) / 2
        try:
            forward, _ = estimate_forward(group)
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        if not np.isfinite(forward) or forward <= 0:
            continue
        group["log_moneyness"] = np.log(group["strike"] / forward)
        # Bin first so dense strikes do not dominate the shape estimate.
        bins = np.linspace(-0.25, 0.22, 20)
        binned = group.groupby(
            pd.cut(group["log_moneyness"], bins=bins, include_lowest=True),
            observed=True,
        )["iv"].median().dropna()
        if len(binned) < 6:
            continue
        x = np.array([interval.mid for interval in binned.index], dtype=float)
        y = binned.to_numpy(float)
        linear = np.polyfit(x, y, 1)
        quadratic = np.polyfit(x, y, 2)
        statistics.append({
            "expiry": expiry,
            "dte": float(group["dte"].iloc[0]),
            "group": group,
            "x": x,
            "y": y,
            "slope": float(linear[0]),
            "curvature": float(quadratic[0]),
        })
    return statistics


def plot_slice(ax, selected: dict, title: str, fit_kind: str):
    group = selected["group"]
    calls = group[group["option_type"].str.lower() == "call"]
    puts = group[group["option_type"].str.lower() == "put"]
    ax.scatter(calls["log_moneyness"], 100 * calls["iv"], s=10, alpha=0.18, color="tab:blue", label="Calls")
    ax.scatter(puts["log_moneyness"], 100 * puts["iv"], s=10, alpha=0.18, color="tab:orange", label="Puts")
    ax.plot(selected["x"], 100 * selected["y"], "ko-", markersize=4, linewidth=1.4, label="Binned median IV")

    x_fit = np.linspace(selected["x"].min(), selected["x"].max(), 200)
    if fit_kind == "skew":
        coefficients = np.polyfit(selected["x"], selected["y"], 1)
        y_fit = np.polyval(coefficients, x_fit)
        annotation = f"Linear slope = {coefficients[0]:.3f} IV / log-moneyness"
    else:
        coefficients = np.polyfit(selected["x"], selected["y"], 2)
        y_fit = np.polyval(coefficients, x_fit)
        annotation = f"Quadratic curvature = {coefficients[0]:.3f}"
    ax.plot(x_fit, 100 * y_fit, color="crimson", linewidth=2, label="Shape fit")

    ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_title(f"{title}: {selected['expiry']} ({selected['dte']:.0f} DTE)")
    ax.set_xlabel("Log-moneyness  ln(K / F)")
    ax.set_ylabel("Implied volatility (%)")
    ax.text(
        0.02,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    ax.minorticks_on()
    ax.grid(True, which="major", color="#b0b0b0", linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", color="#d9d9d9", linewidth=0.5, linestyle=":", alpha=0.8)
    ax.legend(frameon=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skew-expiry", help="Override automatically selected skew expiration")
    parser.add_argument("--smile-expiry", help="Override automatically selected smile expiration")
    parser.add_argument("--skew-min-dte", type=float, default=120.0,
                        help="Minimum DTE for the representative skew slice")
    parser.add_argument("--smile-max-dte", type=float, default=20.0,
                        help="Maximum DTE for the representative smile slice")
    args = parser.parse_args()

    df = clean_data(args.input)
    statistics = slice_statistics(df, max_dte=365)
    if not statistics:
        raise ValueError("No expiration had enough valid IV observations")

    skew = next((item for item in statistics if item["expiry"] == args.skew_expiry), None) if args.skew_expiry else None
    smile = next((item for item in statistics if item["expiry"] == args.smile_expiry), None) if args.smile_expiry else None
    if skew is None:
        long_dated = [item for item in statistics if item["dte"] >= args.skew_min_dte]
        skew = min(long_dated or statistics, key=lambda item: item["slope"])
    if smile is None:
        short_dated = [item for item in statistics if item["dte"] <= args.smile_max_dte]
        positive_curvature = [item for item in short_dated if item["curvature"] > 0]
        smile = max(positive_curvature or short_dated or statistics, key=lambda item: item["curvature"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, dpi=150)
    plot_slice(axes[0], skew, "Volatility skew", "skew")
    plot_slice(axes[1], smile, "Volatility smile", "smile")
    fig.suptitle("SPX implied-volatility slices — June 30, 2023", fontsize=15)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Selected skew slice: {skew['expiry']} ({skew['dte']:.0f} DTE)")
    print(f"Selected smile slice: {smile['expiry']} ({smile['dte']:.0f} DTE)")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
