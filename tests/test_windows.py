"""Tests for cmapss.windows: hand-computed windows, causality, engine isolation, input validation."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cmapss.windows import STATS, feature_names, window_features


def one_engine(values, unit=1, col="y"):
    return pd.DataFrame({"unit": unit, "cycle": np.arange(1, len(values) + 1), col: np.asarray(values, float)})


# ---------- hand-computed ----------

def test_hand_computed_statistics():
    """y = [1, 2, 4, 8], window 3 -> windows [1,2,4] (cycle 3) and [2,4,8] (cycle 4).
    mean 7/3, 14/3 ; slope (y_last - y_first)/2 = 1.5, 3.0 ; population std sqrt(14)/3, sqrt(56)/3 ; last 4, 8."""
    out = window_features(one_engine([1, 2, 4, 8]), ["y"], window=3, stats=STATS)
    assert out["cycle"].tolist() == [3, 4] and out["unit"].tolist() == [1, 1]
    np.testing.assert_allclose(out["y_last"], [4, 8])
    np.testing.assert_allclose(out["y_mean"], [7 / 3, 14 / 3])
    np.testing.assert_allclose(out["y_slope"], [1.5, 3.0])
    np.testing.assert_allclose(out["y_std"], [np.sqrt(14) / 3, np.sqrt(56) / 3])


def test_slope_recovers_exact_linear_trend_and_std_follows_formula():
    cycles = np.arange(1, 101)
    w = 20
    out = window_features(one_engine(5 + 0.3 * cycles), ["y"], window=w, stats=("slope", "std"))
    np.testing.assert_allclose(out["y_slope"], 0.3, atol=1e-12)
    np.testing.assert_allclose(out["y_std"], 0.3 * np.sqrt((w**2 - 1) / 12), atol=1e-12)


def test_constant_signal_has_zero_slope_and_std():
    out = window_features(one_engine([4.0] * 10), ["y"], window=5, stats=("mean", "slope", "std"))
    assert (out["y_mean"] == 4.0).all() and (out["y_slope"].abs() < 1e-15).all() and (out["y_std"] == 0).all()


# ---------- shape, naming, ordering ----------

def test_row_count_first_cycle_and_column_order():
    df = pd.concat([one_engine(np.arange(50), 1, "a").assign(b=1.0), one_engine(np.arange(40), 2, "a").assign(b=2.0)])
    out = window_features(df, ["a", "b"], window=30, stats=("mean", "slope"))
    assert len(out) == (50 - 29) + (40 - 29)
    assert out.groupby("unit")["cycle"].min().tolist() == [30, 30]
    assert list(out.columns) == ["unit", "cycle", "a_mean", "a_slope", "b_mean", "b_slope"]
    assert feature_names(["a", "b"], ("mean", "slope")) == ["a_mean", "a_slope", "b_mean", "b_slope"]


def test_last_statistic_equals_the_raw_value_at_that_cycle():
    rng = np.random.default_rng(0)
    df = one_engine(rng.normal(size=60))
    out = window_features(df, ["y"], window=10, stats=("last",))
    np.testing.assert_allclose(out["y_last"], df["y"].iloc[9:].to_numpy())


def test_rows_given_out_of_order_are_sorted():
    df = one_engine(np.arange(10.0))
    shuffled = df.sample(frac=1, random_state=1)
    pd.testing.assert_frame_equal(window_features(shuffled, ["y"], 4), window_features(df, ["y"], 4))


# ---------- causality and isolation ----------

def test_features_never_use_future_cycles():
    """Truncating an engine after cycle k must leave the features at cycles <= k unchanged."""
    rng = np.random.default_rng(1)
    df = one_engine(rng.normal(size=80))
    full = window_features(df, ["y"], window=15, stats=STATS)
    cut = window_features(df.iloc[:50], ["y"], window=15, stats=STATS)
    pd.testing.assert_frame_equal(full.iloc[: len(cut)].reset_index(drop=True), cut)


def test_changing_a_later_value_does_not_change_earlier_features():
    df = one_engine(np.arange(40.0))
    changed = df.copy(); changed.loc[changed.cycle == 35, "y"] = 1e6
    a, b = window_features(df, ["y"], 10, STATS), window_features(changed, ["y"], 10, STATS)
    early = a["cycle"] < 35
    pd.testing.assert_frame_equal(a[early], b[early])


def test_windows_do_not_cross_engine_boundaries():
    df = pd.concat([one_engine([0.0] * 10, 1), one_engine([100.0] * 10, 2)])
    out = window_features(df, ["y"], window=5, stats=("mean", "std"))
    assert (out.loc[out.unit == 1, "y_mean"] == 0).all() and (out.loc[out.unit == 2, "y_mean"] == 100).all()
    assert (out["y_std"] == 0).all()                     # a window mixing the two engines would have std > 0


# ---------- short engines ----------

def test_short_engine_raises_by_default_and_can_be_dropped_explicitly():
    df = pd.concat([one_engine(np.arange(10.0), 1), one_engine(np.arange(3.0), 2)])
    with pytest.raises(ValueError, match="unit 2 has 3 cycles"):
        window_features(df, ["y"], window=5)
    out = window_features(df, ["y"], window=5, drop_short=True)
    assert out["unit"].unique().tolist() == [1]


def test_window_equal_to_engine_length_gives_one_row():
    assert len(window_features(one_engine(np.arange(7.0)), ["y"], window=7)) == 1


def test_all_engines_too_short_raises_even_when_dropping():
    with pytest.raises(ValueError, match="no engine"):
        window_features(one_engine(np.arange(3.0)), ["y"], window=5, drop_short=True)


# ---------- input validation ----------

@pytest.mark.parametrize("window", [0, -3, 2.5])
def test_bad_window_raises(window):
    with pytest.raises(ValueError, match="window"):
        window_features(one_engine(np.arange(10.0)), ["y"], window=window)


def test_slope_needs_two_points():
    with pytest.raises(ValueError, match="slope"):
        window_features(one_engine(np.arange(10.0)), ["y"], window=1, stats=("slope",))
    assert len(window_features(one_engine(np.arange(10.0)), ["y"], window=1, stats=("mean", "std"))) == 10


@pytest.mark.parametrize("stats", [["median"], [], ["mean", "foo"]])
def test_unknown_or_empty_stats_raise(stats):
    with pytest.raises(ValueError, match="stats"):
        window_features(one_engine(np.arange(10.0)), ["y"], window=3, stats=stats)


def test_missing_column_and_non_finite_values_raise():
    with pytest.raises(ValueError, match="missing columns"):
        window_features(one_engine(np.arange(10.0)), ["nope"], window=3)
    bad = one_engine(np.arange(10.0)); bad.loc[3, "y"] = np.nan
    with pytest.raises(ValueError, match="NaN|infinite"):
        window_features(bad, ["y"], window=3)


# ---------- real data (skipped if not present) ----------

DATA_DIR = Path(os.environ.get("CMAPSS_DATA_DIR", "data"))
HAVE_DATA = all((DATA_DIR / f"{k}_FD001.txt").exists() for k in ("train", "test", "RUL"))


@pytest.mark.skipif(not HAVE_DATA, reason="C-MAPSS files not found in CMAPSS_DATA_DIR")
def test_real_fd001_shapes_and_every_test_engine_is_covered():
    from cmapss.cv import last_rows
    from cmapss.loader import load_subset

    d = load_subset("FD001", DATA_DIR)
    cols = ["s2", "s3", "s4"]
    tr = window_features(d.train, cols, window=30)
    assert len(tr) == len(d.train) - 100 * 29 and tr.shape[1] == 2 + 3 * 3
    te = window_features(d.test, cols, window=30)                 # shortest test engine has 31 cycles
    assert last_rows(te)["unit"].tolist() == list(range(1, 101))
    with pytest.raises(ValueError, match="fewer than window"):
        window_features(d.test, cols, window=32)                  # at least one test engine is shorter
