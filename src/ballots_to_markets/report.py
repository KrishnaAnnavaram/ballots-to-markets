"""The Markdown report. Every table comes from the computed results without change."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _table(df: pd.DataFrame, cols, fmt: str = "{:.4f}") -> str:
    if df is None or df.empty:
        return "_(no rows)_\n"
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt.format(v) if isinstance(v, float) else str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def build(study, events, kinds, causal, forecasts: dict) -> str:
    p = study.panel
    lines = [
        "# ballots-to-markets report", "",
        f"Data: {'SYNTHETIC (planted effects, not real markets)' if study.synthetic else 'local files'}. "
        f"{len(p.returns)} common trading days, {p.first.date()} to {p.last.date()}, "
        f"{p.dropped_days} days dropped because one market was closed.",
        f"Text coverage: {study.coverage:.1%} of trading days. Sentiment model: `{study.sentiment_model}`.", "",
    ]
    for n in study.notes:
        lines.append(f"- NOTE: {n}")
    if study.validation:
        v = study.validation
        lines += ["", "## Sentiment validation", "",
                  f"{v.n} labelled texts. Accuracy {v.accuracy:.3f}, macro F1 {v.macro_f1:.3f}. Recall: "
                  + ", ".join(f"{k} {x:.3f}" for k, x in v.recall.items()) + "."]
    lines += ["", "## Event study (CAR over days -1..+1)", "",
              _table(events, ["event", "date", "asset", "car", "t", "p", "q_bh"]),
              "## Mean CAR by event kind", "", _table(kinds, ["kind", "asset", "n_events", "caar", "t", "p"]),
              "## Stationarity (ADF with constant)", ""]
    adf_df = pd.DataFrame([{"series": k, **v} for k, v in causal["adf"].items()])
    lines.append(_table(adf_df, ["series", "stat", "p", "lags", "nobs"]))
    lines += ["## Granger causality: sentiment -> returns", ""]
    if causal.get("granger") is None:
        lines.append(f"Not reported: {causal.get('refused')}.\n")
    else:
        lines.append(_table(causal["granger"], ["cause", "effect", "lag", "f", "p", "q_bh", "nobs"]))
    for target, table in forecasts.items():
        lines += [f"## Walk-forward forecasts: {target}", "",
                  _table(table.reset_index(), ["model", "n", "rmse", "mae", "r2_oos_vs_mean", "direction_acc",
                                               "dm_vs_zero", "dm_p"], "{:.5f}")]
    return "\n".join(lines) + "\n"


def write(text: str, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.md"
    path.write_text(text, encoding="utf-8")
    return path
