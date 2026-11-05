"""
Turn one trading day of SPX quotes into SVI slices, one per expiry.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from black76 import black_vega, implied_volatility
from svi import SliceData


@dataclass(frozen=True)
class SliceCriteria:
    min_dte: float = 7.0
    max_relative_spread: float = 0.5
    # Five parameters need comfortably more than five quotes to be identified.
    min_quotes: int = 8
    parity_strikes: int = 40


DEFAULT_CRITERIA = SliceCriteria()


def load_day(path: str | Path) -> pd.DataFrame:
    day = pd.read_csv(path)
    day["mid"] = (day["bid"] + day["ask"]) / 2
    return day


def estimate_forward(chain: pd.DataFrame, criteria: SliceCriteria = DEFAULT_CRITERIA) -> tuple[float, float]:
    """Regress C - P on K over the strikes nearest parity; return (forward, discount)."""
    quoted = chain[chain["bid"] > 0]
    pairs = quoted.pivot_table(index="strike", columns="option_type", values="mid").dropna()
    parity = (pairs["call"] - pairs["put"]).rename("parity").reset_index()
    near = parity.loc[parity["parity"].abs().nsmallest(criteria.parity_strikes).index]
    slope, intercept = np.polyfit(near["strike"], near["parity"], 1)
    return -intercept / slope, -slope


def prepare_slice(chain: pd.DataFrame, criteria: SliceCriteria = DEFAULT_CRITERIA) -> SliceData | None:
    """Build one expiry's slice from OTM mid quotes; None when too few quotes survive."""
    T = float(chain["dte"].iloc[0]) / 365
    forward, discount = estimate_forward(chain, criteria)
    is_put = chain["option_type"] == "put"
    otm = chain[(is_put & (chain["strike"] < forward)) | (~is_put & (chain["strike"] > forward))]
    otm = otm[(otm["bid"] > 0) & ((otm["ask"] - otm["bid"]) / otm["mid"] <= criteria.max_relative_spread)]

    strike = otm["strike"].to_numpy(float)
    # Put-call parity turns an OTM put into the ITM call at the same strike: C/D = P/D + F - K.
    to_call = np.where(otm["option_type"] == "put", forward - strike, 0.0)

    def implied(prices: pd.Series) -> np.ndarray:
        calls = prices.to_numpy(float) / discount + to_call
        return np.array([implied_volatility(c, forward, K, T) for c, K in zip(calls, strike)])

    iv, iv_bid, iv_ask = implied(otm["mid"]), implied(otm["bid"]), implied(otm["ask"])
    keep = np.isfinite(iv)
    if keep.sum() < criteria.min_quotes:
        return None

    k = np.log(strike[keep] / forward)
    vol = iv[keep]
    vega = np.array([black_vega(forward, K, T, v) for K, v in zip(strike[keep], vol)])
    weight = vega / (2 * vol * T) ** 2
    order = np.argsort(k)
    return SliceData(
        k=k[order],
        w=(vol**2 * T)[order],
        weight=(weight / weight.sum())[order],
        T=T,
        forward=float(forward),
        discount=float(discount),
        iv_bid=iv_bid[keep][order],
        iv_ask=iv_ask[keep][order],
        expiry=str(chain["expire_date"].iloc[0]),
    )


def prepare_slices(day: pd.DataFrame, criteria: SliceCriteria = DEFAULT_CRITERIA) -> list[SliceData]:
    """Slices for every expiry at least criteria.min_dte days out, shortest first."""
    long_enough = day[day["dte"] >= criteria.min_dte]
    slices = (prepare_slice(chain, criteria) for _, chain in long_enough.groupby("expire_date"))
    return sorted((s for s in slices if s is not None), key=lambda s: s.T)
