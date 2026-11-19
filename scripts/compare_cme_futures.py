#!/usr/bin/env python3
"""Compare SPX option-regression forwards/rates with CME ES futures.

The CME input should contain one row per trade date and futures contract. The
script recognizes these common column names automatically:

    trade_date/date
    expire_date/expiration_date
    settlement_price/settle/settlement

Futures provide forward prices. Without a dividend assumption, the script
reports futures-implied carry, r - q, rather than a pure interest rate.

Examples:
    python3 scripts/compare_cme_futures.py \
        --futures data/cme_es_settlements_2023.csv

    python3 scripts/compare_cme_futures.py \
        --futures data/cme_es_settlements_2023.csv \
        --dividend-yield 0.015
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path


def parse_date(value: str) -> date:
    value = value.strip()[:10]
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return date.fromisoformat(value) if fmt == "%Y-%m-%d" else datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value}")


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def find_column(fieldnames: list[str], requested: str | None, choices: tuple[str, ...]) -> str:
    if requested:
        if requested not in fieldnames:
            raise ValueError(f"Column '{requested}' was not found. Available columns: {fieldnames}")
        return requested
    normalized = {
        name.strip().lower().replace(" ", "_"): name
        for name in fieldnames
    }
    for choice in choices:
        if choice in normalized:
            return normalized[choice]
    raise ValueError(f"Could not find one of {choices}. Available columns: {fieldnames}")


def load_futures(path: Path, date_column: str | None, expiry_column: str | None, price_column: str | None):
    curves = defaultdict(dict)
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        fields = reader.fieldnames or []
        trade_col = find_column(fields, date_column, ("trade_date", "date", "trading_date"))
        expiry_col = find_column(fields, expiry_column, ("expire_date", "expiration_date", "expiry_date", "maturity_date"))
        price_col = find_column(fields, price_column, ("settlement_price", "settle", "settlement", "close", "price"))
        for row in reader:
            try:
                trade_date = parse_date(row[trade_col])
                expire_date = parse_date(row[expiry_col])
            except (ValueError, TypeError):
                continue
            price = parse_float(row[price_col])
            if price is None or price <= 0 or expire_date <= trade_date:
                continue
            curves[trade_date][expire_date] = price
    return curves


def interpolate_forward(curve: dict[date, float], target: date):
    """Return (forward, exact/interpolated, lower_date, upper_date)."""
    if target in curve:
        return curve[target], "exact", target, target
    maturities = sorted(expiry for expiry in curve if expiry > target)
    lower = sorted(expiry for expiry in curve if expiry < target)
    if not lower or not maturities:
        return None, "unavailable", None, None
    lower_date = lower[-1]
    upper_date = maturities[0]
    t1 = (lower_date - min(curve)).days
    t2 = (upper_date - min(curve)).days
    target_t = (target - min(curve)).days
    if t2 <= t1:
        return None, "unavailable", None, None
    weight = (target_t - t1) / (t2 - t1)
    log_forward = math.log(curve[lower_date]) + weight * (
        math.log(curve[upper_date]) - math.log(curve[lower_date])
    )
    return math.exp(log_forward), "interpolated", lower_date, upper_date


def write_svg_plot(rows: list[dict], output: Path):
    usable = [
        row for row in rows
        if row.get("futures_carry_pct") not in (None, "")
        and row.get("implied_rate_pct") not in (None, "")
    ]
    if not usable:
        return
    usable.sort(key=lambda row: float(row["dte"]))
    width, height = 960, 620
    left, right, top, bottom = 85, 35, 70, 80
    pw, ph = width - left - right, height - top - bottom
    xs = [float(row["dte"]) for row in usable]
    option_y = [float(row["implied_rate_pct"]) for row in usable]
    futures_y = [float(row["futures_carry_pct"]) for row in usable]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(option_y + futures_y), max(option_y + futures_y)
    xpad = max((x1 - x0) * 0.04, 1)
    ypad = max((y1 - y0) * 0.08, 0.1)
    x0, x1, y0, y1 = x0 - xpad, x1 + xpad, y0 - ypad, y1 + ypad

    def sx(x): return left + (x - x0) / (x1 - x0) * pw
    def sy(y): return top + (y1 - y) / (y1 - y0) * ph

    option_points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, option_y))
    future_points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, futures_y))
    title = html.escape("Option-regression rate versus CME futures carry")
    subtitle = html.escape(f"{usable[0]['quote_date']} | {len(usable)} matched expirations")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2:.0f}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold">{title}</text>
<text x="{width/2:.0f}" y="52" text-anchor="middle" font-family="Arial" font-size="13">{subtitle}</text>
<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#333"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#333"/>
<polyline points="{option_points}" fill="none" stroke="#1f77b4" stroke-width="2"/>
<polyline points="{future_points}" fill="none" stroke="#d62728" stroke-width="2"/>
<text x="{left+pw/2:.0f}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="14">Days to expiration</text>
<text x="22" y="{top+ph/2:.0f}" text-anchor="middle" transform="rotate(-90 22 {top+ph/2:.0f})" font-family="Arial" font-size="14">Annualized rate / carry (%)</text>
<text x="{left}" y="{top+ph+20}" text-anchor="middle" font-family="Arial" font-size="11">{x0+xpad:.0f}</text>
<text x="{left+pw}" y="{top+ph+20}" text-anchor="middle" font-family="Arial" font-size="11">{x1-xpad:.0f}</text>
<text x="{left-8}" y="{sy(y1)+4:.0f}" text-anchor="end" font-family="Arial" font-size="11">{y1:.1f}</text>
<text x="{left-8}" y="{sy(y0)+4:.0f}" text-anchor="end" font-family="Arial" font-size="11">{y0:.1f}</text>
<line x1="{width-190}" y1="{height-43}" x2="{width-175}" y2="{height-43}" stroke="#1f77b4" stroke-width="2"/><text x="{width-168}" y="{height-39}" font-family="Arial" font-size="12">Options regression</text>
<line x1="{width-190}" y1="{height-20}" x2="{width-175}" y2="{height-20}" stroke="#d62728" stroke-width="2"/><text x="{width-168}" y="{height-16}" font-family="Arial" font-size="12">CME carry</text>
</svg>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path, default=Path("data/cleaned/spx_implied_rates_2023.csv"))
    parser.add_argument("--futures", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/cleaned/spx_options_vs_cme_2023.csv"))
    parser.add_argument("--plot-output", type=Path, default=Path("data/cleaned/spx_options_vs_cme_2023.svg"))
    parser.add_argument("--quote-date", help="Compare one date only; omit for all dates.")
    parser.add_argument("--futures-date-column")
    parser.add_argument("--futures-expiry-column")
    parser.add_argument("--futures-price-column")
    parser.add_argument("--dividend-yield", type=float, help="Continuous dividend yield as a decimal, e.g. 0.015.")
    args = parser.parse_args()

    futures = load_futures(args.futures, args.futures_date_column, args.futures_expiry_column, args.futures_price_column)
    output_columns = [
        "quote_date", "expire_date", "dte", "underlying_last", "options_forward",
        "options_implied_rate_pct", "options_r_squared", "options_accepted",
        "cme_forward", "cme_match_method", "cme_lower_expiry", "cme_upper_expiry",
        "forward_difference", "forward_difference_pct", "futures_carry_pct",
        "futures_implied_rate_pct", "rate_difference_bps",
    ]
    rows = []
    with args.options.open(newline="", encoding="utf-8-sig") as input_file:
        for source in csv.DictReader(input_file):
            if args.quote_date and source["quote_date"] != args.quote_date:
                continue
            if source.get("accepted", "True").lower() != "true":
                continue
            quote_date = parse_date(source["quote_date"])
            expire_date = parse_date(source["expire_date"])
            dte = parse_float(source["dte"])
            spot = parse_float(source.get("underlying_last"))
            options_forward = parse_float(source.get("forward"))
            options_rate = parse_float(source.get("implied_rate_pct"))
            r_squared = parse_float(source.get("r_squared"))
            curve = futures.get(quote_date, {})
            cme_forward, method, lower, upper = interpolate_forward(curve, expire_date)
            T = dte / 365 if dte and dte > 0 else None
            carry = (
                math.log(cme_forward / spot) / T * 100
                if cme_forward and spot and T
                else None
            )
            futures_rate = carry + 100 * args.dividend_yield if carry is not None and args.dividend_yield is not None else None
            rows.append({
                "quote_date": source["quote_date"], "expire_date": source["expire_date"], "dte": source["dte"],
                "underlying_last": spot or "", "options_forward": options_forward or "",
                "options_implied_rate_pct": options_rate or "", "options_r_squared": r_squared or "",
                "options_accepted": source.get("accepted", "True"), "cme_forward": cme_forward or "",
                "cme_match_method": method, "cme_lower_expiry": lower.isoformat() if lower else "",
                "cme_upper_expiry": upper.isoformat() if upper else "",
                "forward_difference": cme_forward - options_forward if cme_forward is not None and options_forward is not None else "",
                "forward_difference_pct": (cme_forward / options_forward - 1) * 100 if cme_forward is not None and options_forward else "",
                "futures_carry_pct": carry if carry is not None else "",
                "futures_implied_rate_pct": futures_rate if futures_rate is not None else "",
                "rate_difference_bps": (options_rate - futures_rate) * 100 if options_rate is not None and futures_rate is not None else "",
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(rows)
    write_svg_plot(rows, args.plot_output)
    matched = sum(row["cme_forward"] != "" for row in rows)
    print(f"Wrote {len(rows):,} option-regression rows to {args.output}; CME matched/interpolated: {matched:,}")
    if rows:
        print(f"Wrote comparison plot to {args.plot_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
