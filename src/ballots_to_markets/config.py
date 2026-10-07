"""Settings from environment variables. The FRED key is never printed."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace


@dataclass(frozen=True)
class Settings:
    data_dir: str = "data"
    out_dir: str = "out"
    start: str = "2023-11-01"
    end: str = "2025-02-10"
    seed: int = 42
    text_timezone: str = "America/New_York"
    market_close_hour: int = 16
    min_text_coverage: float = 0.6  # share of trading days with text, needed for causality tests
    sentiment_model: str = "lexicon"  # "lexicon" or a Hugging Face model name
    fred_api_key: str = field(default="", repr=False)

    ENV = {
        "data_dir": "B2M_DATA_DIR", "out_dir": "B2M_OUT_DIR",
        "start": "B2M_START", "end": "B2M_END", "seed": "B2M_SEED", "text_timezone": "B2M_TEXT_TIMEZONE",
        "market_close_hour": "B2M_MARKET_CLOSE_HOUR", "min_text_coverage": "B2M_MIN_TEXT_COVERAGE",
        "sentiment_model": "B2M_SENTIMENT_MODEL", "fred_api_key": "FRED_API_KEY",
    }

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        types = {f.name: f.type for f in fields(cls)}
        values = {}
        for name, var in cls.ENV.items():
            raw = env.get(var)
            if raw:
                t = types[name]
                values[name] = int(raw) if t in ("int", int) else float(raw) if t in ("float", float) else raw
        return cls(**values).validate()

    def merge(self, **kw) -> "Settings":
        return replace(self, **{k: v for k, v in kw.items() if v is not None}).validate()

    def validate(self) -> "Settings":
        if self.start >= self.end:
            raise ValueError("start must be before end")
        if not 0 <= self.min_text_coverage <= 1:
            raise ValueError("min_text_coverage must be between 0 and 1")
        return self

    def public(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "fred_api_key"}
        out["fred_api_key"] = "set" if self.fred_api_key else "not set"
        return out
