"""
Raw SVI slices, penalised L-BFGS-B calibration, and the interpolated surface.

Parameter order everywhere: (a, b, rho, m, sigma). k = ln(K/F), w = iv^2 * T.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

WING_EPS = 1e-3
PENALTY_TOL = 1e-12
LAMBDA_START = 1e4
LAMBDA_MAX = 1e10
GRID_ATM_STDEVS = 6.0
FIT_GRID_POINTS = 401
CHECK_GRID_POINTS = 4001
RANDOM_STARTS = 10
# The penalty stops at P <= 1e-12, which still allows g ~ -1e-6 at a node and
# deeper dips between nodes; these margins keep the true conditions >= 0.
G_MARGIN = 1e-4
CALENDAR_MARGIN = 1e-5
# The objective is a weighted mean of squared vol errors (~1e-5), so scipy's
# default ftol/gtol would stop long before the parameters settle.
LBFGS_OPTIONS = {"ftol": 1e-15, "gtol": 1e-12, "maxiter": 3000}


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def to_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "SVIParams":
        return cls(*(float(v) for v in x))


@dataclass(frozen=True, eq=False)
class SliceData:
    """One expiry's OTM quotes, sorted by k, with weights summing to 1."""

    k: np.ndarray
    w: np.ndarray
    weight: np.ndarray
    T: float
    forward: float
    discount: float
    iv_bid: np.ndarray
    iv_ask: np.ndarray
    expiry: str = ""

    @property
    def iv(self) -> np.ndarray:
        return np.sqrt(self.w / self.T)


@dataclass(frozen=True, eq=False)
class SliceFit:
    data: SliceData
    params: SVIParams
    rmse_vol_points: float
    penalty: float
    penalty_weight: float
    status: str
    share_inside_bid_ask: float


def total_variance(p: SVIParams, k: np.ndarray) -> np.ndarray:
    x = k - p.m
    return p.a + p.b * (p.rho * x + np.sqrt(x * x + p.sigma**2))


def total_variance_dk(p: SVIParams, k: np.ndarray) -> np.ndarray:
    x = k - p.m
    return p.b * (p.rho + x / np.sqrt(x * x + p.sigma**2))


def total_variance_dkk(p: SVIParams, k: np.ndarray) -> np.ndarray:
    x = k - p.m
    return p.b * p.sigma**2 / (x * x + p.sigma**2) ** 1.5


def durrleman_g(k: np.ndarray, w: np.ndarray, dw: np.ndarray, dww: np.ndarray) -> np.ndarray:
    """Butterfly-arbitrage function g(k); the implied density is non-negative iff g >= 0."""
    return (1 - k * dw / (2 * w)) ** 2 - dw**2 / 4 * (1 / w + 0.25) + dww / 2


def g_function(p: SVIParams, k: np.ndarray) -> np.ndarray:
    return durrleman_g(k, total_variance(p, k), total_variance_dk(p, k), total_variance_dkk(p, k))


def density(p: SVIParams, k: np.ndarray) -> np.ndarray:
    """Risk-neutral density of ln(S_T / F) implied by the slice."""
    w = total_variance(p, k)
    d2 = -k / np.sqrt(w) - np.sqrt(w) / 2
    return g_function(p, k) / np.sqrt(2 * np.pi * w) * np.exp(-(d2**2) / 2)


def density_moments(p: SVIParams) -> tuple[float, float]:
    """Return (integral of p(k) dk, integral of e^k p(k) dk); both are 1 for an arbitrage-free slice."""

    def forward_weighted(k: float) -> float:
        # e^k n(d2) = n(d1), which avoids overflowing e^k far in the right wing.
        w = total_variance(p, k)
        d1 = -k / np.sqrt(w) + np.sqrt(w) / 2
        return g_function(p, k) / np.sqrt(2 * np.pi * w) * np.exp(-(d1**2) / 2)

    def integrate(f) -> float:
        return quad(f, -np.inf, p.m, limit=500)[0] + quad(f, p.m, np.inf, limit=500)[0]

    return integrate(lambda k: density(p, k)), integrate(forward_weighted)


def _grid_range(data: SliceData) -> tuple[float, float]:
    half_width = GRID_ATM_STDEVS * np.sqrt(np.interp(0.0, data.k, data.w))
    return min(-half_width, float(data.k[0])), max(half_width, float(data.k[-1]))


def slice_grid(*slices: SliceData, n: int = FIT_GRID_POINTS) -> np.ndarray:
    """Grid covering every quote and |k| <= 6 * sqrt(w_atm) of each given slice."""
    ranges = np.array([_grid_range(s) for s in slices])
    return np.linspace(ranges[:, 0].min(), ranges[:, 1].max(), n)


def arbitrage_penalty(p: SVIParams, grid: np.ndarray, w_prev: np.ndarray | None) -> float:
    butterfly = np.sum(np.maximum(0.0, G_MARGIN - g_function(p, grid)) ** 2)
    calendar = (
        0.0
        if w_prev is None
        else np.sum(np.maximum(0.0, w_prev + CALENDAR_MARGIN - total_variance(p, grid)) ** 2)
    )
    wings = max(0.0, p.b * (1 + p.rho) - 2 + WING_EPS) ** 2 + max(0.0, p.b * (1 - p.rho) - 2 + WING_EPS) ** 2
    return float(butterfly + calendar + wings)


def _objective(x: np.ndarray, data: SliceData, grid: np.ndarray, w_prev: np.ndarray | None, lam: float) -> float:
    p = SVIParams.from_array(x)
    fit = np.sum(data.weight * (total_variance(p, data.k) - data.w) ** 2)
    return float(fit + lam * arbitrage_penalty(p, grid, w_prev))


def _bounds(data: SliceData) -> list[tuple[float, float]]:
    # a > 0 rather than a >= 0: at the corner a = b = 0 the slice is w = 0 and g is 0/0.
    return [
        (1e-8, float(data.w.max())),
        (0.0, 2.0),
        (-0.999, 0.999),
        (2 * float(data.k.min()), 2 * float(data.k.max())),
        (0.001, 1.0),
    ]


def fit_slice(data: SliceData, prev: SliceFit | None = None, seed: int = 0) -> SliceFit:
    """Calibrate one slice; prev is the fitted slice of the previous expiry, if any."""
    grid = slice_grid(data) if prev is None else slice_grid(data, prev.data)
    w_prev = None if prev is None else total_variance(prev.params, grid)
    bounds = _bounds(data)
    lower, upper = np.array(bounds).T
    guess = np.array([0.5 * data.w.min(), 0.1, -0.5, data.k[np.argmin(data.w)], 0.1])
    rng = np.random.default_rng(seed)
    neighbours = [guess] if prev is None else [guess, prev.params.to_array()]
    starts = [np.clip(x, lower, upper) for x in neighbours] + [rng.uniform(lower, upper) for _ in range(RANDOM_STARTS)]

    def run(x0: np.ndarray, lam: float):
        return minimize(
            _objective, x0, args=(data, grid, w_prev, lam), method="L-BFGS-B", bounds=bounds, options=LBFGS_OPTIONS
        )

    lam = LAMBDA_START
    best = min((run(x0, lam) for x0 in starts), key=lambda r: r.fun)
    while arbitrage_penalty(SVIParams.from_array(best.x), grid, w_prev) > PENALTY_TOL and lam < LAMBDA_MAX:
        lam *= 10
        best = run(best.x, lam)

    params = SVIParams.from_array(best.x)
    model_iv = np.sqrt(total_variance(params, data.k) / data.T)
    inside = (model_iv >= np.nan_to_num(data.iv_bid, nan=0.0)) & (model_iv <= np.nan_to_num(data.iv_ask, nan=np.inf))
    return SliceFit(
        data=data,
        params=params,
        rmse_vol_points=float(100 * np.sqrt(np.mean((model_iv - data.iv) ** 2))),
        penalty=arbitrage_penalty(params, grid, w_prev),
        penalty_weight=lam,
        status=str(best.message),
        share_inside_bid_ask=float(np.mean(inside)),
    )


@dataclass(frozen=True, eq=False)
class SVISurface:
    """Fitted slices sorted by T; w is linear in T at fixed k, IV flat outside the quoted expiries."""

    spot: float
    fits: tuple[SliceFit, ...]

    @property
    def maturities(self) -> np.ndarray:
        return np.array([f.data.T for f in self.fits])

    def log_forward(self, T: float) -> float:
        times = np.concatenate([[0.0], self.maturities])
        log_forwards = np.log(np.concatenate([[self.spot], [f.data.forward for f in self.fits]]))
        if T <= times[-1]:
            return float(np.interp(T, times, log_forwards))
        slope = (log_forwards[-1] - log_forwards[-2]) / (times[-1] - times[-2])
        return float(log_forwards[-1] + slope * (T - times[-1]))

    def _slice_mix(self, T: float) -> list[tuple[SVIParams, float]]:
        times = self.maturities
        if T <= times[0]:
            return [(self.fits[0].params, T / times[0])]
        if T >= times[-1]:
            return [(self.fits[-1].params, T / times[-1])]
        i = int(np.searchsorted(times, T))
        theta = (T - times[i - 1]) / (times[i] - times[i - 1])
        return [(self.fits[i - 1].params, 1 - theta), (self.fits[i].params, theta)]

    def total_variance(self, k: np.ndarray, T: float) -> np.ndarray:
        return sum(c * total_variance(p, k) for p, c in self._slice_mix(T))

    def g(self, k: np.ndarray, T: float) -> np.ndarray:
        mix = self._slice_mix(T)
        return durrleman_g(
            k,
            sum(c * total_variance(p, k) for p, c in mix),
            sum(c * total_variance_dk(p, k) for p, c in mix),
            sum(c * total_variance_dkk(p, k) for p, c in mix),
        )

    def implied_vol(self, K: np.ndarray, T: float) -> np.ndarray:
        k = np.log(K) - self.log_forward(T)
        return np.sqrt(self.total_variance(k, T) / T)


def fit_surface(slices: list[SliceData], spot: float, seed: int = 0) -> SVISurface:
    """Fit expiries shortest first, each penalised against the previous fitted slice."""
    fits: list[SliceFit] = []
    for data in sorted(slices, key=lambda s: s.T):
        fits.append(fit_slice(data, prev=fits[-1] if fits else None, seed=seed))
    return SVISurface(spot=spot, fits=tuple(fits))


def min_g(fit: SliceFit) -> float:
    return float(g_function(fit.params, slice_grid(fit.data, n=CHECK_GRID_POINTS)).min())


def calendar_gap(earlier: SliceFit, later: SliceFit) -> float:
    """Smallest w_later - w_earlier over a grid spanning both slices; negative means the slices cross."""
    k = slice_grid(earlier.data, later.data, n=CHECK_GRID_POINTS)
    return float(np.min(total_variance(later.params, k) - total_variance(earlier.params, k)))
