import numpy as np
import pandas as pd
import pytest

from ballots_to_markets.market import informative_columns, log_returns, rate_changes, return_panel
from ballots_to_markets.sentiment import HFSentiment, LexiconSentiment, validate
from ballots_to_markets.textdays import coverage, daily_features, fill_for_models, to_trading_day

DAYS = pd.DatetimeIndex(["2024-11-01", "2024-11-04", "2024-11-05", "2024-11-06", "2024-11-08"])  # 11-07 closed


def prices():
    idx = pd.bdate_range("2024-01-01", periods=60)
    rng = np.random.default_rng(0)
    a = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60))), index=idx)
    b = a.copy() * 2
    b.iloc[10] = np.nan  # one market closed on one day
    return pd.DataFrame({"a": a, "b": b})


def test_returns_use_trading_days_only():
    """Problem 3: no weekend row and no interpolated price."""
    panel = return_panel(prices())
    assert not (panel.returns.index.dayofweek >= 5).any()
    assert panel.dropped_days == 1
    r = log_returns(prices())
    # Asset b's return after its closed day spans two trading days. It is not split or invented.
    p = prices()["b"].dropna()
    assert r["b"].loc[p.index[10]] == pytest.approx(np.log(p.iloc[10] / p.iloc[9]))


def test_rate_changes_and_flat_level():
    """Problem 7: the almost constant level is flagged. The changes are kept."""
    rate = pd.Series([5.33, 5.33, 4.83], index=pd.to_datetime(["2024-11-01", "2024-11-03", "2024-11-06"]))
    rc = rate_changes(rate, DAYS)
    assert rc["rate_level"].tolist() == [5.33, 5.33, 5.33, 4.83, 4.83]
    assert rc["rate_change"].round(2).tolist() == [0.0, 0.0, 0.0, -0.5, 0.0]
    assert informative_columns(rc[["rate_level"]]) == []


def test_text_goes_to_the_first_tradeable_day():
    ts = pd.Series(pd.to_datetime([
        "2024-11-01 10:00",  # Friday, market open -> same day
        "2024-11-01 17:30",  # Friday after close -> Monday
        "2024-11-02 12:00",  # Saturday -> Monday
        "2024-11-06 16:00",  # Wednesday at close -> next trading day (Friday, Thursday closed)
        "2024-11-09 09:00",  # after the last day -> NaT
    ]))
    out = to_trading_day(ts, DAYS)
    assert [str(d.date()) if pd.notna(d) else None for d in out] == [
        "2024-11-01", "2024-11-04", "2024-11-04", "2024-11-08", None]


def test_ambiguous_dst_hour_is_accepted():
    ts = pd.Series(pd.to_datetime(["2023-11-05 01:30"]))
    out = to_trading_day(ts, pd.DatetimeIndex(["2023-11-06"]))
    assert str(out.iloc[0].date()) == "2023-11-06"


def test_days_without_text_are_not_interpolated():
    """Problem 1: a day with no text has count 0 and NaN sentiment, not an invented value."""
    texts = pd.DataFrame({"timestamp": pd.to_datetime(["2024-11-01 10:00", "2024-11-06 09:00"]),
                          "text": ["great win", "terrible chaos"]})
    f = daily_features(texts, ["positive", "negative"], DAYS, "America/New_York", 16)
    assert f["n_texts"].tolist() == [1, 0, 0, 1, 0]
    assert f["sentiment_mean"].isna().tolist() == [False, True, True, False, True]
    assert coverage(f) == pytest.approx(0.4)
    filled = fill_for_models(f)
    assert filled["sentiment_mean"].tolist() == [1.0, 0.0, 0.0, -1.0, 0.0]
    assert filled["has_text"].tolist() == [1, 0, 0, 1, 0]


def test_lexicon_has_a_neutral_class_and_negation():
    """Problem 2: the model can output neutral."""
    m = LexiconSentiment()
    assert m.predict(["A great win for voters", "The plan is a disaster", "The candidate visits Ohio",
                      "This is not good"]) == ["positive", "negative", "neutral", "negative"]
    v = validate(m, ["great", "bad", "schedule"], ["positive", "negative", "neutral"])
    assert v.recall == {"negative": 1.0, "neutral": 1.0, "positive": 1.0}
    assert len(v.confusion) == 3


def test_binary_hugging_face_model_is_refused(monkeypatch):
    import sys
    import types

    class Pipe:
        model = types.SimpleNamespace(config=types.SimpleNamespace(id2label={0: "NEGATIVE", 1: "POSITIVE"}))

    fake = types.ModuleType("transformers")
    fake.pipeline = lambda *a, **k: Pipe()
    monkeypatch.setitem(sys.modules, "transformers", fake)
    with pytest.raises(ValueError, match="not a 3-class"):
        HFSentiment("distilbert-base-uncased-finetuned-sst-2-english")
