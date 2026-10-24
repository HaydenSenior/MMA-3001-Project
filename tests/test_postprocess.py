"""Tests for cmapss.postprocess.propagate_smooth."""
import numpy as np
import pandas as pd
import pytest

from cmapss.postprocess import propagate_smooth


def preds(values, unit=1, start=1):
    return pd.DataFrame({"unit": unit, "cycle": np.arange(start, start + len(values)), "y_pred": np.asarray(values, float)})


def test_hand_computed_window_two():
    """p = [10, 7, 8]. t=0: 10. t=1: mean(7, 10-1) = 8. t=2: mean(8, 7-1, 10-2) = 22/3."""
    out = propagate_smooth(preds([10, 7, 8]), window=2)
    np.testing.assert_allclose(out, [10.0, 8.0, 22 / 3])


def test_hand_computed_window_one():
    out = propagate_smooth(preds([10, 7, 8]), window=1)
    np.testing.assert_allclose(out, [10.0, 8.0, 7.0])          # t=2: mean(8, 7-1)


def test_window_zero_is_identity():
    np.testing.assert_allclose(propagate_smooth(preds([10, 7, 8]), window=0), [10, 7, 8])


def test_perfect_countdown_is_unchanged():
    p = preds([50, 49, 48, 47, 46])
    np.testing.assert_allclose(propagate_smooth(p, window=4), p["y_pred"])


def test_smoothing_reduces_noise_on_a_noisy_countdown():
    rng = np.random.default_rng(0)
    truth = np.arange(100, 0, -1.0)
    noisy = preds(truth + rng.normal(0, 8, 100))
    err0 = np.sqrt(np.mean((noisy["y_pred"] - truth) ** 2))
    sm = propagate_smooth(noisy, window=15)
    assert np.sqrt(np.mean((sm - truth) ** 2)) < 0.6 * err0


def test_uses_only_earlier_predictions():
    base = preds(np.arange(30, 0, -1.0))
    changed = base.copy(); changed.loc[changed.cycle == 20, "y_pred"] += 500
    a, b = propagate_smooth(base, 5), propagate_smooth(changed, 5)
    assert (a[base.cycle < 20] == b[base.cycle < 20]).all()


def test_engines_are_independent_and_index_is_preserved():
    df = pd.concat([preds([10, 7, 8], 1), preds([100, 100, 100], 2)], ignore_index=True).sample(frac=1, random_state=0)
    out = propagate_smooth(df, window=2)
    assert out.index.equals(df.index)
    u2 = df[df.unit == 2].sort_values("cycle").index
    np.testing.assert_allclose(out.loc[u2], [100.0, 99.5, 99.0])      # 100, mean(100, 99), mean(100, 99, 98)


def test_result_is_clipped_to_zero_and_cap():
    out = propagate_smooth(preds([3, 2, 1]), window=2, cap=125)
    assert (out >= 0).all()
    low = propagate_smooth(preds([1, 1, 1]), window=2)
    assert (low >= 0).all() and low.iloc[2] == pytest.approx(max(0.0, np.mean([1, 0, -1])))
    assert propagate_smooth(preds([500, 500]), window=1, cap=125).max() == 125


def test_errors():
    with pytest.raises(ValueError, match="window"):
        propagate_smooth(preds([1, 2]), window=-1)
    with pytest.raises(ValueError, match="missing"):
        propagate_smooth(pd.DataFrame({"unit": [1]}), window=1)
    gap = preds([5, 4, 3]).iloc[[0, 2]]
    with pytest.raises(ValueError, match="consecutive"):
        propagate_smooth(gap, window=1)
