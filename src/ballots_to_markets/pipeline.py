"""Assemble the study data and run the three analyses: event study, causality, forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import ASSETS
from .causality import adf, granger_table
from .config import Settings
from .data import load_events, load_prices, load_rate_csv, load_text_csv
from .event_study import EventWindow, by_kind
from .event_study import run as run_events
from .forecast import design, score, walk_forward
from .market import ReturnPanel, informative_columns, rate_changes, return_panel
from .sentiment import Validation, build, validate
from .synthetic import SynthSpec, generate
from .textdays import coverage, daily_features, fill_for_models


@dataclass
class Study:
    panel: ReturnPanel
    rate: pd.DataFrame
    text: pd.DataFrame
    events: pd.DataFrame
    synthetic: bool
    sentiment_model: str
    validation: Validation | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return coverage(self.text)


def load_study(s: Settings, synthetic: bool = False, text_paths=None, spec: SynthSpec | None = None,
               events_path=None) -> Study:
    if synthetic:
        prices, rate, texts, _ = generate(spec or SynthSpec(seed=s.seed))
    else:
        base = Path(s.data_dir)
        prices = load_prices(base / "prices")
        rate = load_rate_csv(base / "fed_funds.csv")
        texts = load_text_csv(text_paths or [base / "texts.csv"])
    panel = return_panel(prices, s.start, s.end)
    model = build(s.sentiment_model)
    labels = model.predict(texts["text"].tolist())
    val = validate(model, texts["text"].tolist(), texts["label"].tolist()) if "label" in texts else None
    daily = daily_features(texts, labels, panel.trading_days, s.text_timezone, s.market_close_hour)
    study = Study(panel, rate_changes(rate, panel.trading_days), daily, load_events(events_path), synthetic,
                  model.name, val)
    if study.coverage < s.min_text_coverage:
        study.notes.append(f"text coverage {study.coverage:.0%} is below {s.min_text_coverage:.0%} of trading "
                           "days: the causality tests on text are not reported")
    if "rate_level" not in informative_columns(study.rate[["rate_level"]]):
        study.notes.append("the policy-rate level has fewer than 5 distinct values: models use only its changes")
    return study


def event_study(study: Study, w: EventWindow = EventWindow()):
    table = run_events(study.panel.returns, study.events, w=w)
    return table, by_kind(table) if not table.empty else pd.DataFrame()


def causality(study: Study, s: Settings, lags=(1, 2, 3, 5)) -> dict:
    r = study.panel.returns
    out = {"adf": {c: vars(adf(r[c])) for c in r}}
    if study.coverage < s.min_text_coverage:
        out["granger"] = None
        out["refused"] = f"text coverage {study.coverage:.0%} < {s.min_text_coverage:.0%}"
        return out
    sent = study.text["sentiment_mean"]
    out["adf"]["sentiment_mean"] = vars(adf(sent.dropna()))
    df = r.join(sent)
    out["granger"] = granger_table(df, ["sentiment_mean"], list(r.columns), lags)
    return out


def forecasts(study: Study, target: str, models, lags: int = 3, refit_every: int = 20, seed: int = 42):
    if target not in study.panel.returns:
        raise KeyError(f"unknown asset {target!r}. Known: {list(ASSETS)}")
    exog = fill_for_models(study.text).join(study.rate[["rate_change"]])
    X, y = design(study.panel.returns, exog, target, lags)
    wf = walk_forward(X, y, models, target, refit_every=refit_every, seed=seed)
    return score(wf), wf
