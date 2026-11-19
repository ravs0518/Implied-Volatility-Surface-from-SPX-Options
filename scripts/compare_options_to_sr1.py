#!/usr/bin/env python3
"""Compare option-regression rates with a bootstrapped SR1 SOFR curve.

The GLBX folder is a Databento ``statistics`` download for SR1.FUT.  The
script keeps stat_type=3 settlement records for outright SR1 contracts,
converts each price to 100 - price, and bootstraps monthly discount factors.

This is an approximation: SR1 settles to the average SOFR rate for a calendar
month, not a point zero-coupon rate.  The output therefore labels the result
as an SR1 spot-rate approximation.

Example:
    python3 scripts/compare_options_to_sr1.py
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path


MONTH_CODES = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
SR1_SYMBOL = re.compile(r"^SR1([FGHJKMNQUVXZ])(\d)$")


def parse_date(value: str) -> date:
    value = value.strip()[:10]
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            if fmt == "%Y-%m-%d":
                return date.fromisoformat(value)
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date: {value}")


def parse_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (ValueError, TypeError):
        return None


def month_end(year: int, month: int) -> date:
    if month == 12:
        return date(year + 1, 1, 1) - timedelta(days=1)
    return date(year, month + 1, 1) - timedelta(days=1)


def contract_month(symbol: str) -> tuple[date, date] | None:
    match = SR1_SYMBOL.match(symbol)
    if not match:
        return None
    month = MONTH_CODES[match.group(1)]
    year = 2020 + int(match.group(2))
    start = date(year, month, 1)
    return start, month_end(year, month)


def iter_statistics_files(glbx_dir: Path):
    zstd = shutil.which("zstd")
    if not zstd:
        raise RuntimeError("zstd is required. Install it with: brew install zstd")
    for path in sorted(glbx_dir.glob("*.statistics.csv.zst")):
        process = subprocess.Popen(
            [zstd, "-dc", str(path)],
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        yield path, process.stdout
        process.stdout.close()
        if process.wait() != 0:
            raise RuntimeError(f"Could not decompress {path}")


def load_sr1_settlements(glbx_dir: Path):
    """Return settlements[trade_date][symbol] = (rate, month_start, month_end)."""
    settlements = defaultdict(dict)
    for path, stream in iter_statistics_files(glbx_dir):
        file_date = parse_date(path.name.split("-")[2].split(".")[0])
        for row in csv.DictReader(stream):
            if row.get("stat_type") != "3":
                continue
            symbol = row.get("symbol", "")
            contract = contract_month(symbol)
            if contract is None:
                continue
            price = parse_float(row.get("price"))
            if price is None or not 0 < price < 100:
                continue
            trade_text = row.get("ts_ref") or ""
            trade_date = parse_date(trade_text) if trade_text else file_date
            month_start, month_end = contract
            # Keep the last settlement update published for each symbol/date.
            settlements[trade_date][symbol] = (
                (100.0 - price) / 100.0,
                month_start,
                month_end,
            )
    return settlements


def sr1_rate_to_target(curve, trade_date: date, target: date):
    """Bootstrap a continuous zero rate to an option expiry from SR1 months."""
    if target <= trade_date:
        return None, "unavailable"
    discount_factor = 1.0
    current = trade_date
    used = []
    for rate, start, end in sorted(curve, key=lambda item: item[1]):
        if end <= current:
            continue
        period_start = max(current, start)
        period_end = min(target, end)
        if period_end <= period_start:
            continue
        days = (period_end - period_start).days
        discount_factor /= 1.0 + rate * days / 360.0
        used.append(start.strftime("%Y-%m"))
        current = period_end
        if target <= end:
            break
    if current < target or not used:
        return None, "unavailable"
    years = (target - trade_date).days / 365.0
    zero_rate = -math.log(discount_factor) / years
    return zero_rate, ",".join(used)


def write_svg_plot(rows, output: Path, min_dte: float, max_dte: float):
    daily = defaultdict(lambda: {"options": [], "sr1": []})
    for row in rows:
        dte = float(row["dte"])
        if not min_dte <= dte <= max_dte:
            continue
        if row["options_rate_pct"] != "" and row["sr1_rate_pct"] != "":
            daily[row["quote_date"]]["options"].append(float(row["options_rate_pct"]))
            daily[row["quote_date"]]["sr1"].append(float(row["sr1_rate_pct"]))
    points = []
    for text, values in sorted(daily.items()):
        points.append((parse_date(text), sum(values["options"]) / len(values["options"]), sum(values["sr1"]) / len(values["sr1"])))
    if not points:
        return
    width, height = 1000, 620
    left, right, top, bottom = 90, 35, 72, 85
    pw, ph = width - left - right, height - top - bottom
    x0, x1 = points[0][0], points[-1][0]
    ys = [value for _, option, sr1 in points for value in (option, sr1)]
    y0, y1 = min(ys), max(ys)
    xpad = max((x1 - x0).days * 0.02, 1)
    ypad = max((y1 - y0) * 0.08, 0.1)
    x_min, x_max = x0 - timedelta(days=xpad), x1 + timedelta(days=xpad)
    y_min, y_max = y0 - ypad, y1 + ypad

    def sx(x): return left + (x - x_min).days / (x_max - x_min).days * pw
    def sy(y): return top + (y_max - y) / (y_max - y_min) * ph

    option_points = " ".join(f"{sx(x):.2f},{sy(option):.2f}" for x, option, _ in points)
    sr1_points = " ".join(f"{sx(x):.2f},{sy(sr1):.2f}" for x, _, sr1 in points)
    x_grid = "".join(
        f'<line x1="{sx(x):.2f}" y1="{top}" x2="{sx(x):.2f}" y2="{top + ph}" stroke="#d9d9d9" stroke-width="1"/>'
        for x in [x_min + (x_max - x_min) * i / 5 for i in range(1, 5)]
    )
    y_grid = "".join(
        f'<line x1="{left}" y1="{sy(y):.2f}" x2="{left + pw}" y2="{sy(y):.2f}" stroke="#d9d9d9" stroke-width="1"/>'
        for y in [y_min + (y_max - y_min) * i / 5 for i in range(1, 5)]
    )
    subtitle = html.escape(f"Daily averages for {min_dte:.0f}-{max_dte:.0f} DTE | {len(points)} shared dates")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2:.0f}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold">Options regression versus SR1 SOFR spot-rate approximation</text>
<text x="{width/2:.0f}" y="53" text-anchor="middle" font-family="Arial" font-size="13">{subtitle}</text>
<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#333"/><line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#333"/>
{x_grid}{y_grid}
<polyline points="{option_points}" fill="none" stroke="#1f77b4" stroke-width="2"/><polyline points="{sr1_points}" fill="none" stroke="#d62728" stroke-width="2"/>
<text x="{left+pw/2:.0f}" y="{height-28}" text-anchor="middle" font-family="Arial" font-size="14">Quote date</text>
<text x="22" y="{top+ph/2:.0f}" text-anchor="middle" transform="rotate(-90 22 {top+ph/2:.0f})" font-family="Arial" font-size="14">Annualized rate (%)</text>
<text x="{left}" y="{top+ph+21}" text-anchor="middle" font-family="Arial" font-size="11">{x0}</text><text x="{left+pw}" y="{top+ph+21}" text-anchor="middle" font-family="Arial" font-size="11">{x1}</text>
<text x="{left-8}" y="{sy(y_max)+4:.0f}" text-anchor="end" font-family="Arial" font-size="11">{y_max:.1f}</text><text x="{left-8}" y="{sy(y_min)+4:.0f}" text-anchor="end" font-family="Arial" font-size="11">{y_min:.1f}</text>
<line x1="{width-235}" y1="{height-43}" x2="{width-220}" y2="{height-43}" stroke="#1f77b4" stroke-width="2"/><text x="{width-213}" y="{height-39}" font-family="Arial" font-size="12">Options regression</text>
<line x1="{width-235}" y1="{height-20}" x2="{width-220}" y2="{height-20}" stroke="#d62728" stroke-width="2"/><text x="{width-213}" y="{height-16}" font-family="Arial" font-size="12">SR1 spot approximation</text>
</svg>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glbx-dir", type=Path, help="GLBX Databento folder; auto-detects GLBX-* if omitted.")
    parser.add_argument("--options", type=Path, default=Path("data/cleaned/spx_implied_rates_2023.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/cleaned/options_vs_sr1_2023.csv"))
    parser.add_argument("--plot-output", type=Path, default=Path("data/cleaned/options_vs_sr1_2023.svg"))
    parser.add_argument("--plot-min-dte", type=float, default=20.0)
    parser.add_argument("--plot-max-dte", type=float, default=60.0)
    args = parser.parse_args()

    if args.glbx_dir is None:
        candidates = sorted(Path(".").glob("GLBX-*"))
        if len(candidates) != 1:
            raise ValueError("Pass --glbx-dir because zero or multiple GLBX-* folders were found.")
        args.glbx_dir = candidates[0]
    settlements = load_sr1_settlements(args.glbx_dir)
    rows = []
    with args.options.open(newline="", encoding="utf-8-sig") as input_file:
        for source in csv.DictReader(input_file):
            if source.get("accepted", "True").lower() != "true":
                continue
            quote_date = parse_date(source["quote_date"])
            expire_date = parse_date(source["expire_date"])
            curve = settlements.get(quote_date, {})
            sr1_curve = list(curve.values())
            sr1_rate, contracts = sr1_rate_to_target(sr1_curve, quote_date, expire_date)
            options_rate = parse_float(source.get("implied_rate_pct"))
            difference = (options_rate - sr1_rate) * 100 if options_rate is not None and sr1_rate is not None else None
            rows.append({
                "quote_date": source["quote_date"], "expire_date": source["expire_date"], "dte": source["dte"],
                "options_rate_pct": options_rate if options_rate is not None else "",
                "options_r_squared": source.get("r_squared", ""), "sr1_rate_pct": 100 * sr1_rate if sr1_rate is not None else "",
                "difference_bps": difference if difference is not None else "", "sr1_contract_months": contracts,
            })
    columns = ["quote_date", "expire_date", "dte", "options_rate_pct", "options_r_squared", "sr1_rate_pct", "difference_bps", "sr1_contract_months"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    write_svg_plot(rows, args.plot_output, args.plot_min_dte, args.plot_max_dte)
    matched = sum(row["sr1_rate_pct"] != "" for row in rows)
    print(f"Wrote {len(rows):,} comparisons to {args.output}; SR1 curve matched {matched:,}")
    if matched:
        print(f"Wrote plot to {args.plot_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
