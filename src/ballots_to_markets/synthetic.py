"""Synthetic markets, policy rate and texts with KNOWN planted effects.

- Five assets on business days (weekends and a list of holidays are closed), driven by one market
  factor plus noise.
- Planted event effect: abnormal returns on the trading day after election day.
- Planted lead-lag effect: the daily sentiment index of day t moves the S&P 500 on day t+1.
  Gold has no link to sentiment, so it is the null control.
- Texts every calendar day (also weekends), from word lists that match a 3-class label.
The data only tests that the methods find planted effects. It says nothing about real markets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOLIDAYS = ["2023-11-23", "2023-12-25", "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
            "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25", "2025-01-01", "2025-01-09",
            "2025-01-20"]

WORDS = {
    "positive": ["great", "strong", "hope", "excited", "win", "progress", "proud", "confident"],
    "negative": ["disaster", "weak", "fear", "angry", "corrupt", "crisis", "chaos", "terrible"],
    "neutral": ["says", "plans", "schedule", "meeting", "update", "statement", "visit", "agenda"],
}
FILLER = ["the", "campaign", "town", "voters", "policy", "today", "economy", "speech", "debate", "candidate"]
RATE_STEPS = [("2023-11-01", 5.33), ("2024-09-19", 4.83), ("2024-11-08", 4.58), ("2024-12-19", 4.33)]


@dataclass(frozen=True)
class SynthSpec:
    start: str = "2023-11-01"
    end: str = "2025-02-10"
    election_effect: float = 0.025  # abnormal log return of the S&P 500 on the day after election day
    sentiment_beta: float = 0.006  # S&P 500 return per unit of the previous day's sentiment index
    texts_per_day: float = 20.0
    seed: int = 42


def trading_days(start: str, end: str) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, end)
    return days[~days.isin(pd.to_datetime(HOLIDAYS))]


def generate(spec: SynthSpec = SynthSpec()):
    """Return (prices DataFrame, rate Series, texts DataFrame, sentiment index Series)."""
    rng = np.random.default_rng(spec.seed)
    cal = pd.date_range(spec.start, spec.end, freq="D")
    # Daily sentiment index: AR(1) on calendar days, in [-1, 1] after tanh.
    s = np.zeros(len(cal))
    for i in range(1, len(cal)):
        s[i] = 0.8 * s[i - 1] + rng.normal(0, 0.6)
    sent = pd.Series(np.tanh(s), index=cal, name="sentiment_index")

    days = trading_days(spec.start, spec.end)
    n = len(days)
    f = rng.normal(0.0004, 0.008, n)
    # The S&P 500 reacts to the sentiment of the previous trading day.
    sent_prev = sent.reindex(days).shift(1).fillna(0).to_numpy()
    r = {
        "sp500": f + spec.sentiment_beta * sent_prev,
        "dow": 0.95 * f + rng.normal(0, 0.003, n),
        "gold": -0.1 * f + rng.normal(0.0003, 0.009, n),
    }
    oil = rng.normal(0, 0.018, n)
    r["wti"] = 0.3 * f + oil
    r["brent"] = 0.3 * f + 0.95 * oil + rng.normal(0, 0.004, n)
    after_election = days.searchsorted(pd.Timestamp("2024-11-06"))
    if after_election < n:
        r["sp500"][after_election] += spec.election_effect
        r["dow"][after_election] += 1.3 * spec.election_effect
    start_price = {"sp500": 4200.0, "dow": 33000.0, "gold": 1980.0, "wti": 80.0, "brent": 85.0}
    prices = pd.DataFrame({k: start_price[k] * np.exp(np.cumsum(v)) for k, v in r.items()}, index=days)

    rate = pd.Series({pd.Timestamp(d): v for d, v in RATE_STEPS}).reindex(cal).ffill().rename("rate")

    rows = []
    for d, si in sent.items():
        p_pos = 0.25 + 0.35 * max(si, 0)
        p_neg = 0.25 + 0.35 * max(-si, 0)
        probs = np.array([p_neg, 1 - p_pos - p_neg, p_pos])
        for _ in range(rng.poisson(spec.texts_per_day)):
            lab = ("negative", "neutral", "positive")[rng.choice(3, p=probs)]
            words = list(rng.choice(FILLER, 5)) + [str(rng.choice(WORDS[lab]))]
            rng.shuffle(words)
            ts = d + pd.Timedelta(seconds=int(rng.integers(0, 86400)))
            rows.append((ts, " ".join(words), lab, str(rng.choice(["Candidate A", "Candidate B"]))))
    texts = pd.DataFrame(rows, columns=["timestamp", "text", "label", "group"])
    return prices, rate, texts, sent


def write_csv(out_dir, spec: SynthSpec = SynthSpec()):
    """Write the synthetic data in the layout of the real files (see data/README.md)."""
    from pathlib import Path

    out = Path(out_dir)
    (out / "prices").mkdir(parents=True, exist_ok=True)
    prices, rate, texts, _ = generate(spec)
    for col in prices:
        pd.DataFrame({"Date": prices.index.date, "Close": prices[col].round(4)}).to_csv(
            out / "prices" / f"{col}.csv", index=False)
    pd.DataFrame({"date": rate.index.date, "value": rate.values}).to_csv(out / "fed_funds.csv", index=False)
    texts.rename(columns={"text": "tweet_text", "label": "sentiment", "group": "candidate"}).to_csv(
        out / "texts.csv", index=False)
    return out
