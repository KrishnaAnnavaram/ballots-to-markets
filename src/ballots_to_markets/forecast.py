"""Walk-forward forecasts of the next-day return, with baselines.

At each refit date the model sees only the rows before that date. Scalers are steps of a
scikit-learn pipeline, so they are fit on the training rows only. Every model is compared with
the zero-return forecast and with the expanding historical mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = ("zero", "hist_mean", "ar", "ridge", "gbr", "lstm")


def design(returns: pd.DataFrame, exog: pd.DataFrame | None, target: str, lags: int = 3):
    """Features known at the close of day t, and the target return of day t+1."""
    parts = {f"{c}_l{i}": returns[c].shift(i) for c in returns for i in range(lags)}
    X = pd.DataFrame(parts, index=returns.index)
    if exog is not None and not exog.empty:
        X = X.join(exog, how="left")
    y = returns[target].shift(-1).rename("y")
    data = X.join(y).dropna()
    return data.drop(columns="y"), data["y"]


class _Zero:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class _HistMean:
    def fit(self, X, y):
        self.mu = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu)


class _OwnLags:
    """AR(p) on the target's own lags only."""

    def __init__(self, cols):
        self.cols = cols
        self.lr = LinearRegression()

    def fit(self, X, y):
        self.lr.fit(X[self.cols], y)
        return self

    def predict(self, X):
        return self.lr.predict(X[self.cols])


def make_model(name: str, X: pd.DataFrame, target: str, seed: int = 42):
    if name == "zero":
        return _Zero()
    if name == "hist_mean":
        return _HistMean()
    if name == "ar":
        return _OwnLags([c for c in X.columns if c.startswith(f"{target}_l")])
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "gbr":
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                             early_stopping=True, validation_fraction=0.2, random_state=seed)
    if name == "lstm":
        from .lstm import LSTMRegressor

        return LSTMRegressor(seed=seed)
    raise ValueError(f"unknown model {name!r}. Known: {MODELS}")


@dataclass
class WalkForward:
    predictions: pd.DataFrame  # index = forecast origin day, columns = y and each model
    refits: int


def walk_forward(X: pd.DataFrame, y: pd.Series, models, target: str, initial: float = 0.6,
                 refit_every: int = 20, seed: int = 42) -> WalkForward:
    n = len(y)
    start = int(n * initial)
    if start < 30 or n - start < 10:
        raise ValueError(f"not enough rows ({n}) for a walk-forward test")
    preds = {m: np.full(n, np.nan) for m in models}
    refits = 0
    for block in range(start, n, refit_every):
        stop = min(n, block + refit_every)
        Xtr, ytr = X.iloc[:block], y.iloc[:block]
        for m in models:
            model = make_model(m, X, target, seed).fit(Xtr, ytr)
            preds[m][block:stop] = model.predict(X.iloc[block:stop])
        refits += 1
    out = pd.DataFrame({"y": y.to_numpy(), **preds}, index=y.index).iloc[start:]
    return WalkForward(out, refits)


def diebold_mariano(e_model: np.ndarray, e_base: np.ndarray) -> tuple[float, float]:
    """DM test for equal squared-error loss (one-step forecasts). Negative stat: the model is better."""
    d = e_model ** 2 - e_base ** 2
    if d.std(ddof=1) == 0:
        return 0.0, 1.0
    stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(stat), float(2 * stats.t.sf(abs(stat), df=len(d) - 1))


def score(wf: WalkForward) -> pd.DataFrame:
    p = wf.predictions
    y = p["y"].to_numpy()
    sse_mean = ((y - p["hist_mean"].to_numpy()) ** 2).sum() if "hist_mean" in p else np.nan
    rows = []
    for m in [c for c in p.columns if c != "y"]:
        f = p[m].to_numpy()
        e = y - f
        nz = f != 0
        dm, dm_p = diebold_mariano(e, y) if m != "zero" else (np.nan, np.nan)
        rows.append({
            "model": m, "n": len(y), "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "r2_oos_vs_mean": float(1 - (e ** 2).sum() / sse_mean) if np.isfinite(sse_mean) else np.nan,
            "direction_acc": float((np.sign(f[nz]) == np.sign(y[nz])).mean()) if nz.any() else np.nan,
            "dm_vs_zero": dm, "dm_p": dm_p,
        })
    return pd.DataFrame(rows).set_index("model")
