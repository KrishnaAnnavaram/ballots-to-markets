"""Loaders with schema checks for prices, the policy rate, text and the event calendar."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import pandas as pd

from . import ASSETS


class SchemaError(ValueError):
    pass


def _require(df: pd.DataFrame, cols, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: missing columns {missing}")


def load_prices_csv(path: str | Path) -> pd.Series:
    """A Yahoo-style CSV (``Date``, ``Close``, ...) -> close prices indexed by trading date."""
    df = pd.read_csv(path)
    _require(df, ["Date", "Close"], str(path))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=False)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    if df["Date"].duplicated().any():
        raise SchemaError(f"{path}: duplicate dates")
    if (df["Close"] <= 0).any():
        raise SchemaError(f"{path}: a close price is not positive")
    s = df.set_index("Date")["Close"].sort_index()
    s.index = s.index.tz_localize(None).normalize() if s.index.tz is not None else s.index.normalize()
    return s


def load_prices(folder: str | Path, files: dict[str, str] | None = None) -> pd.DataFrame:
    """Load each asset file. The result keeps NaN on days on which one market was closed."""
    folder = Path(folder)
    files = files or {name: f"{name}.csv" for name in ASSETS}
    series = {}
    for name, fname in files.items():
        p = folder / fname
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. See data/README.md or run 'ballots-to-markets fetch'.")
        series[name] = load_prices_csv(p)
    return pd.DataFrame(series).sort_index()


def load_rate_csv(path: str | Path) -> pd.Series:
    """FRED CSV (``date``, ``value``) or a FRED download (``observation_date``, ``<SERIES>``)."""
    df = pd.read_csv(path)
    if "date" in df.columns and "value" in df.columns:
        date, value = "date", "value"
    elif "observation_date" in df.columns and len(df.columns) == 2:
        date, value = "observation_date", df.columns[1]
    else:
        raise SchemaError(f"{path}: expected date,value columns")
    s = pd.Series(pd.to_numeric(df[value], errors="coerce").values, index=pd.to_datetime(df[date]), name="rate")
    return s.dropna().sort_index()


@dataclass
class TextColumns:
    timestamp: str = "timestamp"
    text: str = "tweet_text"
    label: str | None = "sentiment"
    group: str | None = "candidate"


def load_text_csv(paths, cols: TextColumns = TextColumns()) -> pd.DataFrame:
    """One or more CSV files -> ``timestamp``, ``text``, optional ``label`` and ``group`` columns."""
    frames = []
    for p in [paths] if isinstance(paths, (str, Path)) else paths:
        df = pd.read_csv(p)
        _require(df, [cols.timestamp, cols.text], str(p))
        out = pd.DataFrame({"timestamp": pd.to_datetime(df[cols.timestamp], errors="coerce"),
                            "text": df[cols.text].astype(str)})
        if cols.label and cols.label in df.columns:
            out["label"] = df[cols.label].astype(str).str.lower().str.strip()
        if cols.group and cols.group in df.columns:
            out["group"] = df[cols.group].astype(str)
        frames.append(out)
    df = pd.concat(frames, ignore_index=True).dropna(subset=["timestamp"])
    df = df[df["text"].str.strip() != ""]
    return df.sort_values("timestamp").reset_index(drop=True)


def load_events(path: str | Path | None = None) -> pd.DataFrame:
    """The event calendar: ``date``, ``name``, ``kind`` (``election`` or ``fomc``), ``source``."""
    if path is None:
        with resources.files("ballots_to_markets").joinpath("data/events_2024.csv").open(encoding="utf-8") as fh:
            df = pd.read_csv(fh)
    else:
        df = pd.read_csv(path)
    _require(df, ["date", "name", "kind"], "events")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)
