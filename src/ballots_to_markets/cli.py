"""Command line: ``ballots-to-markets synth | fetch | sentiment | events | causality | forecast | report | config``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# joblib asks Windows for the physical core count with a tool that is often absent. A count below the
# logical core count stops that check. Set it before scikit-learn is imported.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))

import pandas as pd

from . import ASSETS
from .config import Settings
from .event_study import EventWindow
from .fetch import fetch_fred, fetch_prices
from .forecast import MODELS
from .pipeline import causality, event_study, forecasts, load_study
from .report import build as build_report
from .report import write as write_report
from .synthetic import SynthSpec, write_csv

DEFAULT_MODELS = "zero,hist_mean,ar,ridge,gbr"


def _settings(args) -> Settings:
    keys = ("data_dir", "out_dir", "start", "end", "seed", "sentiment_model", "min_text_coverage")
    return Settings.from_env().merge(**{k: getattr(args, k, None) for k in keys})


def _study(args, s):
    return load_study(s, synthetic=args.synthetic, text_paths=args.texts or None, events_path=args.events)


def _show(df: pd.DataFrame) -> None:
    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.precision", 4):
        print(df.to_string(index=False) if not df.empty else "(no rows)")


def cmd_synth(args) -> int:
    out = write_csv(args.out, SynthSpec(seed=args.seed))
    print(f"wrote synthetic prices, policy rate and texts to {out}")
    return 0


def cmd_fetch(args) -> int:
    s = _settings(args)
    base = Path(s.data_dir)
    for p in fetch_prices(base / "prices", s.start, s.end, args.refresh):
        print(f"prices: {p}")
    print(f"policy rate: {fetch_fred(s.fred_api_key, base / 'fed_funds.csv', s.start, s.end, refresh=args.refresh)}")
    return 0


def cmd_sentiment(args) -> int:
    s = _settings(args)
    st = _study(args, s)
    if st.validation is None:
        print("the texts have no labels: nothing to validate")
        return 0
    print(json.dumps(st.validation.as_dict(), indent=2))
    print(f"text coverage: {st.coverage:.1%} of {len(st.text)} trading days")
    return 0


def cmd_events(args) -> int:
    s = _settings(args)
    table, kinds = event_study(_study(args, s), EventWindow(args.window_start, args.window_end))
    _show(table[["event", "date", "asset", "day0", "car", "t", "p", "q_bh"]] if not table.empty else table)
    print()
    _show(kinds)
    return 0


def cmd_causality(args) -> int:
    s = _settings(args)
    res = causality(_study(args, s), s)
    _show(pd.DataFrame([{"series": k, **v} for k, v in res["adf"].items()]))
    print()
    if res["granger"] is None:
        print(f"Granger tests not reported: {res['refused']}")
    else:
        _show(res["granger"])
    return 0


def cmd_forecast(args) -> int:
    s = _settings(args)
    st = _study(args, s)
    table, _ = forecasts(st, args.target, args.models.split(","), refit_every=args.refit_every, seed=s.seed)
    _show(table.reset_index())
    return 0


def cmd_report(args) -> int:
    s = _settings(args)
    st = _study(args, s)
    table, kinds = event_study(st)
    causal = causality(st, s)
    fc = {t: forecasts(st, t, args.models.split(","), seed=s.seed)[0] for t in args.targets.split(",")}
    path = write_report(build_report(st, table, kinds, causal, fc), args.out or s.out_dir)
    print(f"wrote {path}")
    return 0


def cmd_config(args) -> int:
    for k, v in _settings(args).public().items():
        print(f"{k:>18} = {v}")
    return 0


def _common(p) -> None:
    p.add_argument("--synthetic", action="store_true", help="use synthetic data with planted effects")
    p.add_argument("--data-dir", help="folder with prices/, fed_funds.csv, texts.csv (B2M_DATA_DIR)")
    p.add_argument("--texts", nargs="*", help="one or more text CSV files")
    p.add_argument("--events", help="event calendar CSV (default: the built-in 2024 calendar)")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--seed", type=int)
    p.add_argument("--sentiment-model", help="'lexicon' or a 3-class Hugging Face model")
    p.add_argument("--min-text-coverage", type=float)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ballots-to-markets", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("synth", help="write synthetic CSV files")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_synth)
    p = sub.add_parser("fetch", help="download prices (yfinance) and the policy rate (FRED) to the cache")
    p.add_argument("--data-dir")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("sentiment", help="validate the sentiment model on labelled texts")
    _common(p)
    p.set_defaults(fn=cmd_sentiment)
    p = sub.add_parser("events", help="event study around election and FOMC dates")
    _common(p)
    p.add_argument("--window-start", type=int, default=-1)
    p.add_argument("--window-end", type=int, default=1)
    p.set_defaults(fn=cmd_events)
    p = sub.add_parser("causality", help="ADF and Granger tests on trading days")
    _common(p)
    p.set_defaults(fn=cmd_causality)
    p = sub.add_parser("forecast", help="walk-forward next-day forecasts with baselines")
    _common(p)
    p.add_argument("--target", default="sp500", choices=list(ASSETS))
    p.add_argument("--models", default=DEFAULT_MODELS, help=f"comma list of {MODELS}")
    p.add_argument("--refit-every", type=int, default=20)
    p.set_defaults(fn=cmd_forecast)
    p = sub.add_parser("report", help="all analyses in one Markdown report")
    _common(p)
    p.add_argument("--targets", default="sp500,gold")
    p.add_argument("--models", default=DEFAULT_MODELS)
    p.add_argument("--out", help="report folder (B2M_OUT_DIR)")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("config", help="show the settings (the FRED key shows only as set or not set)")
    p.set_defaults(fn=cmd_config)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
