"""Stationarity (ADF), Granger causality F-tests and a small VAR, with NumPy least squares.

The functions give the same statistics as the matching statsmodels functions
(``adfuller`` with ``regression="c"``, ``grangercausalitytests`` ``ssr_ftest``). A test with the
optional ``stats`` extra checks this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .event_study import benjamini_hochberg

# MacKinnon (1994, 2010) approximate p-value coefficients for the ADF test with a constant, 1 series.
_TAU_MAX, _TAU_MIN, _TAU_STAR = 2.74, -18.83, -1.61
_TAU_SMALLP = (2.1659, 1.4412, 0.038269)
_TAU_LARGEP = (1.7339, 0.93202, -0.12745, -0.010368)
ADF_CRITICAL = {"1%": -3.43, "5%": -2.86, "10%": -2.57}  # asymptotic, with constant


def _ols(y: np.ndarray, X: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def mackinnon_p(stat: float) -> float:
    if stat > _TAU_MAX:
        return 1.0
    if stat < _TAU_MIN:
        return 0.0
    coef = _TAU_SMALLP if stat <= _TAU_STAR else _TAU_LARGEP
    return float(stats.norm.cdf(np.polyval(coef[::-1], stat)))


@dataclass
class AdfResult:
    stat: float
    p: float
    lags: int
    nobs: int

    @property
    def stationary(self) -> bool:
        return self.p < 0.05


def adf(series, maxlag: int | None = None) -> AdfResult:
    """Augmented Dickey-Fuller test with a constant. The lag count is selected by AIC."""
    x = np.asarray(pd.Series(series).dropna(), dtype=float)
    n = len(x)
    if maxlag is None:
        maxlag = int(np.ceil(12 * (n / 100) ** 0.25))
    maxlag = min(maxlag, n // 2 - 2)
    dx = np.diff(x)

    def design(k, start):
        rows = range(start, len(dx))
        X = [[1.0, x[t]] + [dx[t - i] for i in range(1, k + 1)] for t in rows]
        return np.array(X), dx[start:]

    best_k, best_aic = 0, np.inf
    for k in range(maxlag + 1):
        X, y = design(k, maxlag)  # one common sample for the AIC comparison
        _, r = _ols(y, X)
        nobs = len(y)
        llf = -nobs / 2 * (np.log(2 * np.pi) + np.log((r @ r) / nobs) + 1)
        aic = -2 * llf + 2 * X.shape[1]
        if aic < best_aic:
            best_k, best_aic = k, aic
    X, y = design(best_k, best_k)
    beta, r = _ols(y, X)
    s2 = (r @ r) / (len(y) - X.shape[1])
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    stat = float(beta[1] / se)
    return AdfResult(stat, mackinnon_p(stat), best_k, len(y))


def _lagged(df: pd.DataFrame, cols, lags: int) -> pd.DataFrame:
    parts = {f"{c}_l{i}": df[c].shift(i) for c in cols for i in range(1, lags + 1)}
    return pd.DataFrame(parts, index=df.index)


@dataclass
class GrangerResult:
    cause: str
    effect: str
    lag: int
    f: float
    p: float
    nobs: int


def granger(df: pd.DataFrame, cause: str, effect: str, lag: int) -> GrangerResult:
    """F-test: do ``lag`` lags of ``cause`` add to ``lag`` lags of ``effect``?

    Lags are taken on the trading-day index first. Then each row with a NaN is dropped, so a day
    with no text never gets an invented value.
    """
    lags_y = _lagged(df, [effect], lag)
    lags_x = _lagged(df, [cause], lag)
    data = pd.concat([df[[effect]], lags_y, lags_x], axis=1).dropna()
    n = len(data)
    if n <= 3 * lag + 2:
        raise ValueError(f"too few complete rows ({n}) for lag {lag}")
    y = data[effect].to_numpy()
    ones = np.ones((n, 1))
    Xr = np.hstack([ones, data[lags_y.columns].to_numpy()])
    Xu = np.hstack([Xr, data[lags_x.columns].to_numpy()])
    _, rr = _ols(y, Xr)
    _, ru = _ols(y, Xu)
    df_den = n - Xu.shape[1]
    f = ((rr @ rr - ru @ ru) / lag) / ((ru @ ru) / df_den)
    return GrangerResult(cause, effect, lag, float(f), float(stats.f.sf(f, lag, df_den)), n)


def granger_table(df: pd.DataFrame, causes, effects, lags=(1, 2, 3, 5)) -> pd.DataFrame:
    rows = []
    for c in causes:
        for e in effects:
            for k in lags:
                try:
                    r = granger(df, c, e, k)
                    rows.append(vars(r))
                except ValueError as exc:
                    rows.append({"cause": c, "effect": e, "lag": k, "f": np.nan, "p": np.nan, "nobs": 0,
                                 "note": str(exc)})
    out = pd.DataFrame(rows)
    out["q_bh"] = benjamini_hochberg(out["p"].to_numpy())
    return out


@dataclass
class VarModel:
    columns: list[str]
    lag: int
    coef: np.ndarray  # (1 + k * lag, k)
    aic: float

    def forecast_one(self, history: pd.DataFrame) -> np.ndarray:
        last = history[self.columns].to_numpy()[-self.lag:][::-1]  # lag 1 first
        x = np.concatenate([[1.0], last.ravel()])
        return x @ self.coef


def _var_design(values: np.ndarray, lag: int):
    n, k = values.shape
    X = np.ones((n - lag, 1 + k * lag))
    for i in range(1, lag + 1):
        X[:, 1 + (i - 1) * k:1 + i * k] = values[lag - i:n - i]
    return X, values[lag:]


def fit_var(df: pd.DataFrame, columns, max_lag: int = 5) -> VarModel:
    """VAR by OLS. The lag (1..max_lag) with the lowest AIC on a common sample wins."""
    values = df[list(columns)].dropna().to_numpy()
    n, k = values.shape
    best = None
    for lag in range(1, max_lag + 1):
        X, Y = _var_design(values[max_lag - lag:], lag)
        B, *_ = np.linalg.lstsq(X, Y, rcond=None)
        R = Y - X @ B
        sigma = (R.T @ R) / len(Y)
        aic = np.log(np.linalg.det(sigma)) + 2 * lag * k * k / len(Y)
        if best is None or aic < best[1]:
            best = (lag, aic)
    lag = best[0]
    X, Y = _var_design(values, lag)
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return VarModel(list(columns), lag, B, float(best[1]))
