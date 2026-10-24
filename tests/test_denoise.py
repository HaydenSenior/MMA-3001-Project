"""Tests for cmapss.denoise: hand-computed values, causality, engine isolation, validation."""
import numpy as np
import pandas as pd
import pytest

from cmapss.denoise import baseline_normalise, ewma, rolling_median


def eng(values, unit=1, col="y"):
    return pd.DataFrame({"unit": unit, "cycle": np.arange(1, len(values) + 1), col: np.asarray(values, float)})


ALL = [ewma, rolling_median, baseline_normalise]


# ---------- hand-computed ----------

def test_ewma_hand_computed():
    """span 3 -> alpha 0.5. y = [0, 2, 2]: s = 0, 1, 1.5."""
    np.testing.assert_allclose(ewma(eng([0, 2, 2]), ["y"], span=3)["y"], [0.0, 1.0, 1.5])


def test_ewma_span_one_is_identity_and_constant_stays_constant():
    np.testing.assert_allclose(ewma(eng([3, 1, 4, 1]), ["y"], span=1)["y"], [3, 1, 4, 1])
    np.testing.assert_allclose(ewma(eng([5.0] * 6), ["y"], span=4)["y"], 5.0)


def test_rolling_median_hand_computed_and_removes_a_spike():
    """w=3, y=[1,5,2,8]: medians of [1], [1,5], [1,5,2], [5,2,8] = 1, 3, 2, 5."""
    np.testing.assert_allclose(rolling_median(eng([1, 5, 2, 8]), ["y"], window=3)["y"], [1, 3, 2, 5])
    spiky = rolling_median(eng([1, 1, 100, 1, 1]), ["y"], window=3)["y"]
    assert spiky.iloc[2:].max() == 1.0


def test_baseline_normalise_hand_computed():
    """k=2, y=[2,4,9,9]: baselines 2, 3, 3, 3 -> [0, 1, 6, 6]."""
    np.testing.assert_allclose(baseline_normalise(eng([2, 4, 9, 9]), ["y"], n_baseline=2)["y"], [0, 1, 6, 6])


def test_baseline_normalise_removes_engine_offsets():
    df = pd.concat([eng(np.arange(60.0) * 0.1 + 500, 1), eng(np.arange(60.0) * 0.1 + 900, 2)])
    out = baseline_normalise(df, ["y"], n_baseline=30)
    a, b = out[out.unit == 1]["y"].to_numpy(), out[out.unit == 2]["y"].to_numpy()
    np.testing.assert_allclose(a, b)                  # identical once the offset is removed


def test_baseline_shorter_than_k_uses_what_exists():
    out = baseline_normalise(eng([4.0, 6.0]), ["y"], n_baseline=30)["y"]
    np.testing.assert_allclose(out, [0.0, 1.0])       # baselines 4, 5


# ---------- causality and isolation ----------

@pytest.mark.parametrize("fn", ALL)
def test_changing_a_later_value_does_not_change_earlier_output(fn):
    base = eng(np.random.default_rng(0).normal(size=60))
    changed = base.copy(); changed.loc[changed.cycle == 45, "y"] += 1e6
    a, b = fn(base, ["y"]), fn(changed, ["y"])
    pd.testing.assert_frame_equal(a[a.cycle < 45], b[b.cycle < 45])


@pytest.mark.parametrize("fn", ALL)
def test_truncating_an_engine_leaves_earlier_output_unchanged(fn):
    base = eng(np.random.default_rng(1).normal(size=50))
    full, cut = fn(base, ["y"]), fn(base.iloc[:30], ["y"])
    pd.testing.assert_frame_equal(full.iloc[:30].reset_index(drop=True), cut.reset_index(drop=True))


def test_baseline_is_causal_inside_the_baseline_period_too():
    """Values at cycles before t must not change when a value at cycle t (t <= n_baseline) changes,
    and truncating the engine inside the baseline period must not change the earlier output."""
    base = eng(np.random.default_rng(2).normal(size=60))
    changed = base.copy(); changed.loc[changed.cycle == 20, "y"] += 1e6
    a, b = baseline_normalise(base, ["y"], 30), baseline_normalise(changed, ["y"], 30)
    pd.testing.assert_frame_equal(a[a.cycle < 20], b[b.cycle < 20])
    cut = baseline_normalise(base.iloc[:15], ["y"], 30)
    pd.testing.assert_frame_equal(a.iloc[:15].reset_index(drop=True), cut.reset_index(drop=True))


@pytest.mark.parametrize("fn", ALL)
def test_engines_do_not_influence_each_other(fn):
    one, two = eng(np.arange(20.0), 1), eng(np.arange(20.0) + 1000, 2)
    both = fn(pd.concat([one, two]), ["y"])
    alone = fn(one, ["y"])
    pd.testing.assert_frame_equal(both[both.unit == 1].reset_index(drop=True), alone.reset_index(drop=True))


@pytest.mark.parametrize("fn", ALL)
def test_input_is_not_modified_and_other_columns_untouched(fn):
    df = eng(np.arange(10.0)).assign(other=7.0)
    before = df.copy()
    out = fn(df, ["y"])
    pd.testing.assert_frame_equal(df, before)
    assert (out["other"] == 7.0).all() and out["unit"].tolist() == df["unit"].tolist()


@pytest.mark.parametrize("fn", ALL)
def test_unsorted_input_is_sorted(fn):
    df = eng(np.arange(10.0))
    pd.testing.assert_frame_equal(fn(df.sample(frac=1, random_state=3), ["y"]), fn(df, ["y"]))


# ---------- validation ----------

def test_bad_parameters_raise():
    with pytest.raises(ValueError, match="span"):
        ewma(eng([1, 2, 3]), ["y"], span=0.5)
    with pytest.raises(ValueError, match="window"):
        rolling_median(eng([1, 2, 3]), ["y"], window=0)
    with pytest.raises(ValueError, match="n_baseline"):
        baseline_normalise(eng([1, 2, 3]), ["y"], n_baseline=0)


@pytest.mark.parametrize("fn", ALL)
def test_missing_column_and_non_finite_raise(fn):
    with pytest.raises(ValueError, match="missing columns"):
        fn(eng([1, 2, 3]), ["nope"])
    bad = eng([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="NaN|infinite"):
        fn(bad, ["y"])
