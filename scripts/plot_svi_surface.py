#!/usr/bin/env python3
"""Plot one fitted SVI surface with observed option IV quotes.

The default figure uses the June 30, 2023 snapshot and log-forward moneyness
on the x-axis:

    k = ln(K / F_0,T)

Example:
    python3 scripts/plot_svi_surface.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from svi import fit_surface
from svi_data import SliceCriteria, estimate_forward, prepare_slices


DEFAULT_INPUT = Path("data/Q2/max_spx_options_filtered_2023-06-30.csv")
DEFAULT_OUTPUT = Path("results/svi_surface_log_moneyness_2023-06-30.png")


def quote_points(day: pd.DataFrame, max_iv: float):
    points = []
    for _, group in day.groupby("expire_date"):
        try:
            forward, _ = estimate_forward(group)
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        if not np.isfinite(forward) or forward <= 0:
            continue
        valid = group[
            (group["iv"] > 0)
            & (group["iv"] <= max_iv)
            & (group["bid"] > 0)
            & (group["ask"] >= group["bid"])
        ].copy()
        if valid.empty:
            continue
        points.append(pd.DataFrame({
            "k": np.log(valid["strike"].to_numpy(float) / forward),
            "dte": valid["dte"].to_numpy(float),
            "iv": valid["iv"].to_numpy(float),
        }))
    return pd.concat(points, ignore_index=True) if points else pd.DataFrame(columns=["k", "dte", "iv"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-dte", type=float, default=3.0)
    parser.add_argument("--max-dte", type=float, default=365.0)
    parser.add_argument("--max-spread", type=float, default=0.50)
    parser.add_argument("--min-quotes", type=int, default=8)
    parser.add_argument("--parity-strikes", type=int, default=40)
    parser.add_argument("--quote-iv-max", type=float, default=1.0)
    parser.add_argument("--k-min", type=float, default=-0.25)
    parser.add_argument("--k-max", type=float, default=0.25)
    parser.add_argument("--k-points", type=int, default=100)
    parser.add_argument("--t-points", type=int, default=70)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    day = pd.read_csv(args.input)
    day["mid"] = (day["bid"] + day["ask"]) / 2
    criteria = SliceCriteria(
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        max_relative_spread=args.max_spread,
        min_quotes=args.min_quotes,
        parity_strikes=args.parity_strikes,
    )
    slices = prepare_slices(day, criteria)
    if len(slices) < 2:
        raise ValueError("At least two usable expirations are required")

    spot = float(day["underlying_last"].dropna().iloc[0])
    surface = fit_surface(slices, spot=spot, seed=args.seed)

    k = np.linspace(args.k_min, args.k_max, args.k_points)
    maturities = np.linspace(float(surface.maturities[0]), float(surface.maturities[-1]), args.t_points)
    iv_grid = []
    for T in maturities:
        forward = np.exp(surface.log_forward(float(T)))
        strikes = forward * np.exp(k)
        iv_grid.append(surface.implied_vol(strikes, float(T)))
    iv_grid = np.asarray(iv_grid)
    K, T = np.meshgrid(k, 365.0 * maturities)

    quotes = quote_points(day, args.quote_iv_max)
    fig = plt.figure(figsize=(11, 8), dpi=150)
    ax = fig.add_subplot(projection="3d")
    surface_plot = ax.plot_surface(K, T, iv_grid, cmap="viridis", linewidth=0.15, antialiased=True, alpha=0.92)
    if not quotes.empty:
        ax.scatter(quotes["k"], quotes["dte"], quotes["iv"], c="black", s=2.5, alpha=0.55, label="Observed quotes")
    ax.set_xlabel("log-forward moneyness  ln(K / F₀,ₜ)", labelpad=10)
    ax.set_ylabel("DTE (days)", labelpad=10)
    ax.set_zlabel("Implied volatility", labelpad=8)
    ax.set_title(f"SVI implied-volatility surface — {day['quote_date'].iloc[0]}\n(black = observed quotes)")
    ax.view_init(elev=28, azim=-60)
    ax.set_xlim(args.k_min, args.k_max)
    ax.set_ylim(0, 365.0 * float(surface.maturities[-1]))
    ax.set_zlim(0, max(0.45, float(np.nanmax(iv_grid) * 1.05)))
    fig.colorbar(surface_plot, ax=ax, shrink=0.6, pad=0.1, label="Implied volatility")
    if not quotes.empty:
        ax.legend(loc="upper left")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Fitted {len(surface.fits)} expirations")
    print(f"Overlayed {len(quotes):,} observed quotes")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
