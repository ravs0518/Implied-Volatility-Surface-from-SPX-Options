"""
Black-76 call pricing and implied volatility on undiscounted prices.

Black-76 is Black-Scholes with spot = forward and zero rate, applied to
price / discount factor. Ported from notebooks/iv_inversion_newton_brent.ipynb.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def black_call(forward: float, strike: float, time_to_expiry: float, volatility: float) -> float:
    """Undiscounted Black-76 call price."""
    volatility_sqrt_time = volatility * np.sqrt(time_to_expiry)
    d1 = np.log(forward / strike) / volatility_sqrt_time + 0.5 * volatility_sqrt_time
    d2 = d1 - volatility_sqrt_time
    return forward * norm.cdf(d1) - strike * norm.cdf(d2)


def black_vega(forward: float, strike: float, time_to_expiry: float, volatility: float) -> float:
    """Undiscounted Black-76 vega, identical for calls and puts."""
    volatility_sqrt_time = volatility * np.sqrt(time_to_expiry)
    d1 = np.log(forward / strike) / volatility_sqrt_time + 0.5 * volatility_sqrt_time
    return forward * norm.pdf(d1) * np.sqrt(time_to_expiry)


def implied_volatility(
    undiscounted_call: float,
    forward: float,
    strike: float,
    time_to_expiry: float,
    lower_volatility: float = 1e-4,
    upper_volatility: float = 5.0,
    tolerance: float = 1e-10,
    max_iterations: int = 50,
    initial_guess: float = 0.2,
) -> float:
    """Newton-Raphson with a Brent fallback; NaN when the price is outside no-arbitrage bounds."""
    intrinsic = max(forward - strike, 0.0)
    if not intrinsic < undiscounted_call < forward:
        return np.nan

    volatility = initial_guess
    for _ in range(max_iterations):
        price_error = black_call(forward, strike, time_to_expiry, volatility) - undiscounted_call
        if abs(price_error) < tolerance * max(1.0, undiscounted_call):
            return volatility
        vega = black_vega(forward, strike, time_to_expiry, volatility)
        # Near-zero vega means the Newton step is dominated by rounding noise.
        if vega < 1e-10:
            break
        next_volatility = volatility - price_error / vega
        if not lower_volatility < next_volatility < upper_volatility:
            break
        volatility = next_volatility

    try:
        return brentq(
            lambda vol: black_call(forward, strike, time_to_expiry, vol) - undiscounted_call,
            lower_volatility,
            upper_volatility,
            xtol=1e-12,
        )
    except ValueError:
        return np.nan
