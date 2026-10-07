import pandas as pd
import pytest

from ballots_to_markets.cli import main
from ballots_to_markets.config import Settings
from ballots_to_markets.data import SchemaError, load_events, load_prices_csv, load_rate_csv, load_text_csv
from ballots_to_markets.fetch import fred_url, parse_fred
from ballots_to_markets.pipeline import causality, load_study
from ballots_to_markets.synthetic import write_csv

KEY = "abcdef" + "0123456789" * 3


def test_price_loader_checks(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("Date,Close\n2024-01-02,10\n2024-01-03,11\n", encoding="utf-8")
    assert load_prices_csv(p).tolist() == [10.0, 11.0]
    p.write_text("Date,Open\n2024-01-02,10\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        load_prices_csv(p)
    p.write_text("Date,Close\n2024-01-02,10\n2024-01-02,11\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        load_prices_csv(p)
    p.write_text("Date,Close\n2024-01-02,-1\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        load_prices_csv(p)


def test_rate_and_text_loaders(tmp_path):
    r = tmp_path / "r.csv"
    r.write_text("date,realtime_start,realtime_end,value\n2024-01-02,x,x,5.33\n2024-01-03,x,x,.\n", encoding="utf-8")
    assert load_rate_csv(r).tolist() == [5.33]
    t = tmp_path / "t.csv"
    t.write_text('tweet_id,user_handle,timestamp,tweet_text,candidate,party,retweets,likes,sentiment\n'
                 '1,@u,2024-11-03 08:45:00,"Great day",A,P,1,2,Positive\n2,@v,bad-date,"x",A,P,1,2,neutral\n',
                 encoding="utf-8")
    df = load_text_csv([t])
    assert len(df) == 1 and df.loc[0, "label"] == "positive" and df.loc[0, "group"] == "A"


def test_builtin_event_calendar():
    ev = load_events()
    assert len(ev) == 16 and set(ev["kind"]) == {"election", "fomc"}
    assert (ev["date"].diff().dropna() >= pd.Timedelta(0)).all()


def test_low_text_coverage_refuses_granger(tmp_path):
    """Problem 1: with sparse text, the causality test on text is refused, not run on filled values."""
    write_csv(tmp_path)
    texts = pd.read_csv(tmp_path / "texts.csv")
    texts["day"] = pd.to_datetime(texts["timestamp"]).dt.date
    keep = sorted(texts["day"].unique())[:120]
    texts[texts["day"].isin(keep)].drop(columns="day").to_csv(tmp_path / "texts.csv", index=False)
    s = Settings(data_dir=str(tmp_path))
    st = load_study(s)
    assert st.coverage < 0.6 and any("coverage" in n for n in st.notes)
    res = causality(st, s)
    assert res["granger"] is None and "coverage" in res["refused"]


def test_fred_helpers_and_key_masking(monkeypatch, capsys):
    url = fred_url("DFF", KEY, "2024-01-01", "2024-02-01")
    assert "series_id=DFF" in url and "file_type=json" in url
    df = parse_fred({"observations": [{"date": "2024-01-02", "value": "5.33"}, {"date": "2024-01-03", "value": "."}]})
    assert df.to_dict("records") == [{"date": "2024-01-02", "value": "5.33"}]
    monkeypatch.setenv("FRED_API_KEY", KEY)
    assert KEY not in repr(Settings.from_env())
    assert main(["config"]) == 0
    out = capsys.readouterr().out
    assert KEY not in out and "fred_api_key = set" in out


def test_cli_end_to_end(tmp_path, capsys):
    data = tmp_path / "d"
    assert main(["synth", "--out", str(data)]) == 0
    for cmd in (["sentiment"], ["events"], ["causality"], ["forecast", "--models", "zero,hist_mean,ar"]):
        assert main(cmd + ["--data-dir", str(data)]) == 0
    assert main(["report", "--data-dir", str(data), "--out", str(tmp_path / "o"), "--models", "zero,hist_mean,ar",
                 "--targets", "sp500"]) == 0
    text = (tmp_path / "o" / "report.md").read_text(encoding="utf-8")
    assert "## Event study" in text and "## Granger causality" in text and "## Walk-forward forecasts: sp500" in text
    capsys.readouterr()
    assert main(["events", "--data-dir", str(tmp_path / "missing")]) == 2
    with pytest.raises(ValueError):
        Settings(start="2025-01-01", end="2024-01-01").validate()
