import numpy as np
import pandas as pd
import pytest

from ballots_to_markets import forecast as fc
from ballots_to_markets.pipeline import forecasts


def toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"a": rng.normal(0, 0.01, n), "b": rng.normal(0, 0.01, n)}, index=idx)


def test_target_is_the_next_day_return():
    r = toy()
    X, y = fc.design(r, None, "a", lags=2)
    day = X.index[5]
    nxt = r.index[r.index.get_loc(day) + 1]
    assert y.loc[day] == r.loc[nxt, "a"]
    assert X.loc[day, "a_l0"] == r.loc[day, "a"] and X.loc[day, "a_l1"] == r["a"].shift(1).loc[day]


def test_walk_forward_trains_only_on_the_past(monkeypatch):
    """Problem 4: each fit sees only rows before the forecast block, scaler included."""
    r = toy()
    X, y = fc.design(r, None, "a")
    seen = []

    class Spy:
        def fit(self, Xtr, ytr):
            seen.append(Xtr.index.max())
            return self

        def predict(self, Xte):
            assert Xte.index.min() > seen[-1]
            return np.zeros(len(Xte))

    real = fc.make_model
    monkeypatch.setattr(fc, "make_model", lambda name, *a, **k: Spy() if name == "spy" else real(name, *a, **k))
    wf = fc.walk_forward(X, y, ["spy", "zero"], "a", refit_every=25)
    assert wf.refits == len(seen) and wf.refits >= 3
    with pytest.raises(ValueError):
        fc.walk_forward(X.iloc[:20], y.iloc[:20], ["zero"], "a")


def test_baselines_are_always_in_the_table(study):
    """Problem 5: every model is compared with the zero and historical-mean forecasts."""
    table, wf = forecasts(study, "sp500", ["zero", "hist_mean", "ar", "ridge", "gbr"], seed=7)
    assert set(table.index) == {"zero", "hist_mean", "ar", "ridge", "gbr"}
    assert table.loc["hist_mean", "r2_oos_vs_mean"] == pytest.approx(0.0)
    assert np.isnan(table.loc["zero", "dm_vs_zero"])
    assert wf.predictions.index.is_monotonic_increasing


def test_diebold_mariano_sign():
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    good, bad = y - rng.normal(0, 0.1, 300), y - rng.normal(0, 1.0, 300)
    stat, p = fc.diebold_mariano(good, bad)
    assert stat < 0 and p < 0.01
    assert fc.diebold_mariano(good, good) == (0.0, 1.0)


def test_unknown_model_and_target(study):
    with pytest.raises(ValueError):
        fc.make_model("prophet", None, "a")
    with pytest.raises(KeyError):
        forecasts(study, "bitcoin", ["zero"])


def test_lstm_optional():
    pytest.importorskip("torch")
    from ballots_to_markets.lstm import LSTMRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3)).astype("float32")
    y = 0.5 * X[:, 0] + rng.normal(0, 0.1, 120)
    m = LSTMRegressor(window=5, epochs=30, seed=1).fit(X[:100], y[:100])
    p = m.predict(X[100:])
    assert p.shape == (20,) and np.isfinite(p).all()
