"""Returns on a TRADING-DAY index. No weekend or holiday is invented and no price is interpolated.

Each asset return uses two consecutive trading days of that asset. The panel then keeps only the
days on which all assets traded (an inner join). The report counts the dropped days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ReturnPanel:
    returns: pd.DataFrame  # log returns, index = common trading days
    dropped_days: int
    first: pd.Timestamp
    last: pd.Timestamp

    @property
    def trading_days(self) -> pd.DatetimeIndex:
        return self.returns.index


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Each column on its own trading days, then a log difference."""
    out = {}
    for col in prices:
        s = prices[col].dropna()
        out[col] = np.log(s).diff().iloc[1:]
    return pd.DataFrame(out)


def return_panel(prices: pd.DataFrame, start: str | None = None, end: str | None = None) -> ReturnPanel:
    if start or end:
        prices = prices.loc[start:end]
    r = log_returns(prices)
    full = len(r)
    r = r.dropna(how="any")
    if len(r) < 30:
        raise ValueError(f"only {len(r)} common trading days. Check the date range and the files")
    return ReturnPanel(r, full - len(r), r.index[0], r.index[-1])


def rate_changes(rate: pd.Series, days: pd.DatetimeIndex) -> pd.DataFrame:
    """The policy rate on each trading day (last known value) and its change from the day before.

    The rate changes only a few times in a year. The level is almost constant, so the models use the
    change and an indicator of the change days.
    """
    level = rate.sort_index().reindex(rate.index.union(days)).ffill().reindex(days)
    change = level.diff().fillna(0.0)
    return pd.DataFrame({"rate_level": level, "rate_change": change, "rate_move_day": (change != 0).astype(int)})


def informative_columns(df: pd.DataFrame, min_distinct: int = 5) -> list[str]:
    """Columns with at least ``min_distinct`` distinct values. Near-constant columns add no information."""
    return [c for c in df.columns if df[c].nunique(dropna=True) >= min_distinct]
