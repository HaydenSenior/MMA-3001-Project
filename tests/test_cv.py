"""Tests for cmapss.cv on synthetic engines with known structure, plus real-data sanity checks."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cmapss.cv import cross_validate, fit_predict_test, last_rows
from cmapss.models import MeanBaseline, RidgeRegressor


def make_train(n_engines=20, seed=0):
    """Engines with random lengths; feature 'rul_feat' equals the true RUL exactly, 'noise' is useless."""
    rng = np.random.default_rng(seed)
    frames = []
    for unit in range(1, n_engines + 1):
        n = int(rng.integers(150, 260))
        cycle = np.arange(1, n + 1)
        frames.append(pd.DataFrame({"unit": unit, "cycle": cycle, "rul_feat": (n - cycle).astype(float),
                                    "noise": rng.normal(size=n)}))
    return pd.concat(frames, ignore_index=True)


# ---------- last_rows ----------

def test_last_rows_picks_final_cycle_per_engine_sorted_by_unit():
    df = pd.DataFrame({"unit": [2, 2, 1, 1, 1], "cycle": [1, 2, 1, 2, 3], "v": [10, 20, 1, 2, 3]})
    r = last_rows(df)
    assert r["unit"].tolist() == [1, 2] and r["v"].tolist() == [3, 20] and r.index.tolist() == [1, 2]


# ---------- cross_validate ----------

def test_perfect_feature_gives_zero_error_when_uncapped():
    folds, _ = cross_validate(make_train(), ["rul_feat"], lambda: RidgeRegressor(0), cap=None, n_splits=5)
    assert (folds["rmse"] < 1e-6).all() and (folds["score"] < 1e-6).all()


def test_informative_feature_beats_mean_predictor_and_noise_does_not():
    df = make_train()
    ridge, _ = cross_validate(df, ["rul_feat"], lambda: RidgeRegressor(1.0), cap=None)
    mean, _ = cross_validate(df, ["rul_feat"], MeanBaseline, cap=None)
    noise, _ = cross_validate(df, ["noise"], lambda: RidgeRegressor(1.0), cap=None)
    assert ridge["rmse"].mean() < 0.01 * mean["rmse"].mean()
    assert noise["rmse"].mean() > 0.8 * mean["rmse"].mean()      # no real improvement over the mean


def test_scored_rows_respect_rul_range_and_min_cycle():
    df = make_train()
    _, preds = cross_validate(df, ["rul_feat"], lambda: RidgeRegressor(0), cap=None,
                              eval_rul_range=(10, 150), eval_min_cycle=31)
    assert preds["y_true"].between(10, 150).all() and (preds["cycle"] >= 31).all()


def test_every_engine_is_validated_in_exactly_one_fold_and_never_trained_on_in_that_fold():
    df = make_train(n_engines=20)
    folds, preds = cross_validate(df, ["rul_feat"], lambda: RidgeRegressor(0), cap=None, n_splits=5)
    per_unit_folds = preds.groupby("unit")["fold"].nunique()
    assert (per_unit_folds == 1).all() and preds["unit"].nunique() == 20
    assert (folds["train_engines"] + folds["val_engines"] == 20).all()


def test_fresh_model_per_fold():
    calls = []
    def factory():
        calls.append(1); return RidgeRegressor(1.0)
    cross_validate(make_train(), ["rul_feat"], factory, cap=None, n_splits=4)
    assert len(calls) == 4


def test_same_seed_reproducible():
    df = make_train()
    a, _ = cross_validate(df, ["rul_feat", "noise"], lambda: RidgeRegressor(1.0), seed=3)
    b, _ = cross_validate(df, ["rul_feat", "noise"], lambda: RidgeRegressor(1.0), seed=3)
    pd.testing.assert_frame_equal(a.drop(columns="fit_seconds"), b.drop(columns="fit_seconds"))


class _Const:
    def __init__(self, v): self.v = v
    def fit(self, X, y): return self
    def predict(self, X): return np.full(len(X), self.v)


def test_predictions_are_clipped_to_zero_and_cap():
    df = make_train()
    _, hi = cross_validate(df, ["rul_feat"], lambda: _Const(1000.0), cap=125)
    _, lo = cross_validate(df, ["rul_feat"], lambda: _Const(-50.0), cap=125)
    assert (hi["y_pred"] == 125).all() and (lo["y_pred"] == 0).all()


def test_errors_are_measured_against_uncapped_truth():
    df = make_train()
    _, preds = cross_validate(df, ["rul_feat"], lambda: _Const(125.0), cap=125)
    assert preds["y_true"].max() > 125          # truth above the cap is kept, not clipped


def test_empty_evaluation_range_raises():
    with pytest.raises(ValueError, match="no validation rows"):
        cross_validate(make_train(), ["rul_feat"], MeanBaseline, eval_rul_range=(5000, 6000))


# ---------- fit_predict_test ----------

def test_test_prediction_uses_only_the_last_cycle_of_each_engine():
    train = make_train()
    test = pd.DataFrame({"unit": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2],
                         "rul_feat": [1e9, 1e9, 40.0, 1e9, 90.0], "noise": 0.0})
    pred = fit_predict_test(train, test, ["rul_feat"], lambda: RidgeRegressor(0), cap=None)
    assert pred.index.tolist() == [1, 2]
    np.testing.assert_allclose(pred.to_numpy(), [40.0, 90.0], atol=1e-6)


# ---------- real data (skipped if not present) ----------

DATA_DIR = Path(os.environ.get("CMAPSS_DATA_DIR", "data"))
HAVE_DATA = all((DATA_DIR / f"{k}_FD001.txt").exists() for k in ("train", "test", "RUL"))


@pytest.mark.skipif(not HAVE_DATA, reason="C-MAPSS files not found in CMAPSS_DATA_DIR")
def test_real_fd001_ridge_beats_mean_predictor_in_every_fold():
    from cmapss.features import select_columns
    from cmapss.loader import SENSOR_COLUMNS, SETTING_COLUMNS, load_raw
    train = load_raw(DATA_DIR / "train_FD001.txt")
    kept, _ = select_columns(train, SETTING_COLUMNS + SENSOR_COLUMNS)
    ridge, _ = cross_validate(train, kept, lambda: RidgeRegressor(1.0))
    mean, _ = cross_validate(train, kept, MeanBaseline)
    assert (ridge["rmse"] < mean["rmse"]).all()


@pytest.mark.skipif(not HAVE_DATA, reason="C-MAPSS files not found in CMAPSS_DATA_DIR")
def test_real_fd001_test_prediction_shape_without_scoring():
    from cmapss.features import select_columns
    from cmapss.loader import SENSOR_COLUMNS, SETTING_COLUMNS, load_subset
    d = load_subset("FD001", DATA_DIR)
    kept, _ = select_columns(d.train, SETTING_COLUMNS + SENSOR_COLUMNS)
    pred = fit_predict_test(d.train, d.test, kept, lambda: RidgeRegressor(1.0))
    assert len(pred) == 100 and pred.index.tolist() == list(range(1, 101))
    assert np.isfinite(pred).all() and pred.between(0, 125).all()
