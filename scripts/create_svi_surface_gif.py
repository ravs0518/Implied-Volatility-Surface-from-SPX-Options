#!/usr/bin/env python3
"""Fit and animate SVI implied-volatility surfaces across quote dates.

Examples:
    python3 scripts/create_svi_surface_gif.py
    python3 scripts/create_svi_surface_gif.py \
        data/Q1/max_spx_options_filtered_2023-01-04.csv \
        data/Q2/max_spx_options_filtered_2023-06-30.csv \
        data/Q3/max_spx_options_filtered_2023-08-15.csv \
        data/Q4/max_spx_options_filtered_2023-11-14.csv \
        --output results/svi_surface_quarterly.gif

The default input is the all-year filtered file. SVI calibration is performed
separately for every quote date and then rendered as one GIF frame per date.
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
from matplotlib.animation import FuncAnimation, PillowWriter

# Allow the script to import the project-level SVI modules when run as
# ``python3 scripts/create_svi_surface_gif.py`` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from svi import SVISurface, fit_surface
from svi_data import SliceCriteria, prepare_slices


DEFAULT_INPUT = Path("data/cleaned/spx_options_2023_filtered.csv")
DEFAULT_OUTPUT = Path("results/svi_surface_2023.gif")


def load_date_groups(paths: list[Path]) -> dict[str, pd.DataFrame]:
    grouped: dict[str, list[pd.DataFrame]] = {}
    for path in paths:
        frame = pd.read_csv(path)
        frame["quote_date"] = frame["quote_date"].astype(str).str[:10]
        for quote_date, day in frame.groupby("quote_date", sort=True):
            grouped.setdefault(quote_date, []).append(day.copy())
    return {
        quote_date: pd.concat(frames, ignore_index=True)
        for quote_date, frames in sorted(grouped.items())
    }


def fit_date_surface(day: pd.DataFrame, criteria: SliceCriteria, seed: int) -> SVISurface:
    day = day.copy()
    day["mid"] = (day["bid"] + day["ask"]) / 2
    slices = prepare_slices(day, criteria)
    if len(slices) < 2:
        raise ValueError(f"only {len(slices)} usable expirations")
    spot = float(day["underlying_last"].dropna().iloc[0])
    return fit_surface(slices, spot=spot, seed=seed)


def surface_grid(surface: SVISurface, k_min: float, k_max: float, n_k: int, n_t: int):
    k = np.linspace(k_min, k_max, n_k)
    maturities = np.linspace(float(surface.maturities[0]), float(surface.maturities[-1]), n_t)
    volatility = []
    for T in maturities:
        forward = np.exp(surface.log_forward(float(T)))
        strikes = forward * np.exp(k)
        volatility.append(surface.implied_vol(strikes, float(T)))
    return k, 365.0 * maturities, np.asarray(volatility) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="CSV files containing long-format option data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--min-dte", type=float, default=7.0)
    parser.add_argument("--max-dte", type=float, default=365.0)
    parser.add_argument("--max-spread", type=float, default=0.50)
    parser.add_argument("--min-quotes", type=int, default=8)
    parser.add_argument("--parity-strikes", type=int, default=40)
    parser.add_argument("--max-dates", type=int, help="Keep only the first N dates after sorting")
    parser.add_argument("--date-step", type=int, default=1, help="Use every Nth date; useful for large GIFs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-min", type=float, default=-0.25)
    parser.add_argument("--k-max", type=float, default=0.25)
    parser.add_argument("--k-points", type=int, default=80)
    parser.add_argument("--t-points", type=int, default=40)
    args = parser.parse_args()

    paths = args.inputs or [DEFAULT_INPUT]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))

    date_groups = load_date_groups(paths)
    dates = sorted(date_groups)[:: max(1, args.date_step)]
    if args.max_dates is not None:
        dates = dates[: args.max_dates]
    if not dates:
        raise ValueError("No quote dates found")

    criteria = SliceCriteria(
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        max_relative_spread=args.max_spread,
        min_quotes=args.min_quotes,
        parity_strikes=args.parity_strikes,
    )

    fitted = []
    skipped = []
    for index, quote_date in enumerate(dates, start=1):
        try:
            surface = fit_date_surface(date_groups[quote_date], criteria, args.seed)
            grid = surface_grid(surface, args.k_min, args.k_max, args.k_points, args.t_points)
            fitted.append((quote_date, grid))
            print(f"[{index}/{len(dates)}] fitted {quote_date} ({len(surface.fits)} expirations)")
        except Exception as error:  # report one bad date and continue the animation
            skipped.append((quote_date, str(error)))
            print(f"[{index}/{len(dates)}] skipped {quote_date}: {error}")

    if not fitted:
        raise RuntimeError("No dates could be fitted")

    z_max = max(float(np.nanmax(grid[2])) for _, grid in fitted)
    z_max = max(1.0, np.ceil(z_max / 5.0) * 5.0)
    fig = plt.figure(figsize=(10, 7), dpi=120)

    def draw(frame: int):
        quote_date, (k, dte, iv) = fitted[frame]
        ax = fig.axes[0] if fig.axes else fig.add_subplot(projection="3d")
        ax.clear()
        K, T = np.meshgrid(k, dte)
        ax.plot_surface(K, T, iv, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel("Log-forward moneyness, k")
        ax.set_ylabel("Days to expiration")
        ax.set_zlabel("Implied volatility (%)")
        ax.set_title(f"SVI implied volatility surface — {quote_date}")
        ax.set_zlim(0, z_max)
        return ax,

    animation = FuncAnimation(fig, draw, frames=len(fitted), interval=1000 / args.fps, blit=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(args.output, writer=PillowWriter(fps=args.fps))
    plt.close(fig)

    print(f"Wrote {args.output} with {len(fitted)} frames")
    if skipped:
        print(f"Skipped {len(skipped)} dates")
        for quote_date, reason in skipped:
            print(f"  {quote_date}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
