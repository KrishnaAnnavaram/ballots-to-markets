"""Optional downloads with a local cache: Yahoo Finance prices (extra ``fetch``) and FRED series.

Files go into ``<data_dir>/prices/<asset>.csv`` and ``<data_dir>/fed_funds.csv``. A cached file is
used again unless ``refresh`` is true. The FRED key is sent to FRED only and is never printed.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from . import ASSETS

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def fred_url(series: str, key: str, start: str, end: str) -> str:
    q = {"series_id": series, "api_key": key, "file_type": "json", "observation_start": start, "observation_end": end}
    return f"{FRED_URL}?{urllib.parse.urlencode(q)}"


def parse_fred(payload: dict) -> pd.DataFrame:
    rows = [(o["date"], o["value"]) for o in payload.get("observations", []) if o.get("value") not in (".", None)]
    return pd.DataFrame(rows, columns=["date", "value"])


def fetch_fred(key: str, out: Path, start: str, end: str, series: str = "DFF", refresh: bool = False) -> Path:
    if out.exists() and not refresh:
        return out
    if not key:
        raise RuntimeError("FRED_API_KEY is not set. Get a free key at fred.stlouisfed.org")
    with urllib.request.urlopen(fred_url(series, key, start, end), timeout=30) as resp:  # pragma: no cover
        df = parse_fred(json.load(resp))
    out.parent.mkdir(parents=True, exist_ok=True)  # pragma: no cover
    df.to_csv(out, index=False)  # pragma: no cover
    return out  # pragma: no cover


def fetch_prices(folder: Path, start: str, end: str, refresh: bool = False) -> list[Path]:  # pragma: no cover
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError('yfinance is not installed: pip install -e ".[fetch]"') from exc
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    end_inclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
    for name, ticker in ASSETS.items():
        p = folder / f"{name}.csv"
        if refresh or not p.exists():
            df = yf.download(ticker, start=start, end=end_inclusive, progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.reset_index()[["Date", "Close"]].to_csv(p, index=False)
        paths.append(p)
    return paths
