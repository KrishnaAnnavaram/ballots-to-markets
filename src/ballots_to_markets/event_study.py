"""Event study: abnormal returns around election and FOMC events, on trading days.

- Event day 0 is the first trading day at or after the event date.
- The estimation window ends ``gap`` trading days before day 0 and has ``est_len`` days.
- Normal return model: the market model (alpha + beta * market return) for each asset, and the
  constant-mean model for the market itself.
- CAR = sum of abnormal returns in the event window. t = CAR / (sigma * sqrt(window length)).
- p-values of all (event, asset) pairs get a Benjamini-Hochberg correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import MARKET


@dataclass(frozen=True)
class EventWindow:
    start: int = -1
    end: int = 1
    est_len: int = 120
    gap: int = 10
    min_est: int = 60


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Adjusted p-values (q-values) of the Benjamini-Hochberg procedure. NaN stays NaN."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    if pv.size == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * pv.size / np.arange(1, pv.size + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty_like(pv)
    res[order] = np.minimum(q, 1.0)
    out[ok] = res
    return out


def abnormal_returns(returns: pd.DataFrame, event_date, asset: str, w: EventWindow = EventWindow(),
                     market: str = MARKET) -> dict | None:
    days = returns.index
    t0 = days.searchsorted(pd.Timestamp(event_date))
    if t0 >= len(days) or t0 + w.end >= len(days) or t0 + w.start < 0:
        return None
    est_end = t0 + w.start - w.gap
    est_start = max(0, est_end - w.est_len)
    if est_end - est_start < w.min_est:
        return None
    est = returns.iloc[est_start:est_end]
    win = returns.iloc[t0 + w.start:t0 + w.end + 1]
    y = est[asset].to_numpy()
    if asset == market:
        alpha, beta, k = y.mean(), 0.0, 1
        resid = y - alpha
        expected = np.full(len(win), alpha)
    else:
        x = est[market].to_numpy()
        beta, alpha = np.polyfit(x, y, 1)
        resid = y - (alpha + beta * x)
        k = 2
        expected = alpha + beta * win[market].to_numpy()
    sigma = np.sqrt((resid ** 2).sum() / (len(y) - k))
    ar = win[asset].to_numpy() - expected
    car = float(ar.sum())
    t = car / (sigma * np.sqrt(len(ar)))
    p = float(2 * stats.t.sf(abs(t), df=len(y) - k))
    return {"asset": asset, "day0": days[t0].date().isoformat(), "car": car, "t": float(t), "p": p,
            "beta": float(beta), "sigma": float(sigma), "n_est": int(len(y)),
            "ar_day0": float(ar[-w.start]) if w.start <= 0 <= w.end else float("nan")}


def run(returns: pd.DataFrame, events: pd.DataFrame, assets=None, w: EventWindow = EventWindow(),
        market: str = MARKET) -> pd.DataFrame:
    rows = []
    for _, ev in events.iterrows():
        for a in assets or list(returns.columns):
            r = abnormal_returns(returns, ev["date"], a, w, market)
            if r:
                rows.append({"event": ev["name"], "kind": ev["kind"], "date": ev["date"].date().isoformat(), **r})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["q_bh"] = benjamini_hochberg(df["p"].to_numpy())
    return df


def by_kind(table: pd.DataFrame) -> pd.DataFrame:
    """Mean CAR (CAAR) of each event kind and asset, with a cross-event t-test."""
    rows = []
    for (kind, asset), g in table.groupby(["kind", "asset"]):
        cars = g["car"].to_numpy()
        if len(cars) >= 2 and cars.std(ddof=1) > 0:
            t = cars.mean() / (cars.std(ddof=1) / np.sqrt(len(cars)))
            p = float(2 * stats.t.sf(abs(t), df=len(cars) - 1))
        else:
            t, p = float("nan"), float("nan")
        rows.append({"kind": kind, "asset": asset, "n_events": len(cars), "caar": float(cars.mean()),
                     "t": float(t), "p": p})
    return pd.DataFrame(rows)
