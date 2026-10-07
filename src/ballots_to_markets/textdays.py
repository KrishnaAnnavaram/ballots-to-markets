"""Map each text to the trading day on which the market can first react, then aggregate.

A text after the market close, or on a weekend or holiday, belongs to the NEXT trading day.
A trading day with no text keeps ``n_texts = 0`` and NaN sentiment. Nothing is interpolated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sentiment import SCORE


def to_trading_day(ts: pd.Series, days: pd.DatetimeIndex, tz: str = "America/New_York",
                   close_hour: int = 16) -> pd.Series:
    """Return the trading day of each timestamp. NaT if it is after the last trading day."""
    t = pd.to_datetime(ts)
    if t.dt.tz is None:
        # A naive time is local time in ``tz``. In the repeated autumn hour it counts as standard time.
        t = t.dt.tz_localize(tz, ambiguous=np.zeros(len(t), dtype=bool), nonexistent="shift_forward")
    else:
        t = t.dt.tz_convert(tz)
    date = t.dt.tz_localize(None).dt.normalize()
    after_close = t.dt.hour >= close_hour
    date = date + pd.to_timedelta(after_close.astype(int), unit="D")
    days = pd.DatetimeIndex(days).normalize()
    pos = days.searchsorted(date.values, side="left")
    out = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
    ok = pos < len(days)
    out[ok] = days[pos[ok]]
    return out


def daily_features(texts: pd.DataFrame, labels: list[str], days: pd.DatetimeIndex, tz: str,
                   close_hour: int) -> pd.DataFrame:
    """One row for each trading day: text count, mean score, class shares, a has-text flag."""
    df = texts.copy()
    df["pred"] = labels
    df["day"] = to_trading_day(df["timestamp"], days, tz, close_hour)
    df = df.dropna(subset=["day"])
    df["score"] = df["pred"].map(SCORE)
    g = df.groupby("day")
    out = pd.DataFrame(index=pd.DatetimeIndex(days, name="date"))
    out["n_texts"] = g.size().reindex(out.index).fillna(0).astype(int)
    out["sentiment_mean"] = g["score"].mean().reindex(out.index)
    for lab in ("negative", "neutral", "positive"):
        out[f"share_{lab}"] = g["pred"].apply(lambda s, lab=lab: (s == lab).mean()).reindex(out.index)
    out["has_text"] = (out["n_texts"] > 0).astype(int)
    return out


def coverage(features: pd.DataFrame) -> float:
    """The share of trading days with at least one text."""
    return float(features["has_text"].mean()) if len(features) else 0.0


def fill_for_models(features: pd.DataFrame) -> pd.DataFrame:
    """For forecasting only: a day with no text gets a neutral value 0, and ``has_text`` tells the model."""
    out = features.copy()
    cols = [c for c in out.columns if c.startswith(("sentiment_", "share_"))]
    out[cols] = out[cols].fillna(0.0)
    return out.replace([np.inf, -np.inf], 0.0)
