import numpy as np
import pandas as pd
import pytest

from ballots_to_markets.causality import adf, fit_var, granger, granger_table, mackinnon_p
from ballots_to_markets.event_study import EventWindow, abnormal_returns, benjamini_hochberg, by_kind
from ballots_to_markets.pipeline import causality, event_study


def test_benjamini_hochberg_known_values():
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, np.nan, 0.5]))
    assert np.allclose(q[[0, 1, 2, 4]], [0.04, 0.05333333, 0.05333333, 0.5])
    assert np.isnan(q[3])


def test_planted_election_effect_is_found(study):
    table, kinds = event_study(study)
    # The effect is planted on day +1 (the first session after the result). A one-day window finds it.
    one_day = abnormal_returns(study.panel.returns, "2024-11-05", "sp500", EventWindow(start=1, end=1))
    assert one_day["car"] > 0.03 and one_day["p"] < 0.01
    assert table.loc[table["event"] == "Election day", "day0"].iloc[0] == "2024-11-05"
    assert set(kinds["kind"]) == {"election", "fomc"}
    # A Sunday event starts on the Monday.
    assert table.loc[table["event"] == "President Biden ends his campaign", "day0"].iloc[0] == "2024-07-22"


def test_event_needs_an_estimation_window(study):
    r = study.panel.returns
    assert abnormal_returns(r, r.index[20], "sp500", EventWindow()) is None
    assert abnormal_returns(r, r.index[-1], "sp500", EventWindow()) is None


def test_adf_separates_noise_and_random_walk():
    rng = np.random.default_rng(1)
    assert adf(rng.normal(size=300)).stationary
    assert not adf(np.cumsum(rng.normal(size=300))).stationary
    assert mackinnon_p(5.0) == 1.0 and mackinnon_p(-30.0) == 0.0


def test_granger_finds_the_planted_lead_and_not_the_null(study, settings):
    """Problem 6: sentiment (not counts) is tested, with a multiple-testing correction."""
    res = causality(study, settings)
    g = res["granger"]
    sp = g[(g["effect"] == "sp500") & (g["lag"] == 1)].iloc[0]
    gold = g[(g["effect"] == "gold") & (g["lag"] == 1)].iloc[0]
    assert sp["q_bh"] < 0.05 and gold["p"] > 0.05
    assert res["adf"]["sentiment_mean"]["p"] < 0.05


def test_granger_drops_missing_rows_instead_of_filling():
    rng = np.random.default_rng(2)
    x = rng.normal(size=200)
    y = np.r_[0, 0.5 * x[:-1]] + rng.normal(size=200)
    df = pd.DataFrame({"x": x, "y": y})
    df.loc[50:59, "x"] = np.nan
    r = granger(df, "x", "y", 1)
    assert r.nobs == 199 - 10  # the 10 rows whose x lag is missing are dropped, not filled
    assert r.p < 1e-6
    with pytest.raises(ValueError):
        granger(df.iloc[:5], "x", "y", 2)
    t = granger_table(df, ["x"], ["y"], lags=(1, 2))
    assert list(t["lag"]) == [1, 2] and "q_bh" in t


def test_var_recovers_known_coefficients():
    rng = np.random.default_rng(3)
    A = np.array([[0.5, 0.2], [0.0, 0.3]])
    z = np.zeros((2000, 2))
    for t in range(1, 2000):
        z[t] = A @ z[t - 1] + rng.normal(0, 1, 2)
    m = fit_var(pd.DataFrame(z, columns=["a", "b"]), ["a", "b"], max_lag=4)
    assert m.lag == 1
    assert np.allclose(m.coef[1:].T, A, atol=0.06)
    assert m.forecast_one(pd.DataFrame(z, columns=["a", "b"])).shape == (2,)


def test_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.tsa.stattools")
    rng = np.random.default_rng(4)
    x = 0.5 * np.cumsum(rng.normal(size=200)) + rng.normal(size=200)
    ref = sm.adfuller(x, regression="c", autolag="AIC")
    ours = adf(x)
    assert ours.stat == pytest.approx(ref[0]) and ours.p == pytest.approx(ref[1], abs=1e-6)
    a = rng.normal(size=300)
    df = pd.DataFrame({"y": np.r_[0, 0.3 * a[:-1]] + rng.normal(size=300), "x": a})
    f_ref = sm.grangercausalitytests(df[["y", "x"]], [2])[2][0]["ssr_ftest"]
    assert granger(df, "x", "y", 2).f == pytest.approx(f_ref[0])


def test_by_kind_needs_two_events():
    t = pd.DataFrame({"kind": ["a", "a", "b"], "asset": ["x", "x", "x"], "car": [0.01, 0.03, 0.02]})
    k = by_kind(t).set_index("kind")
    assert k.loc["a", "n_events"] == 2 and np.isfinite(k.loc["a", "t"])
    assert np.isnan(k.loc["b", "t"])
