"""Tests for cmapss.metrics, using hand-computed cases.

Reference case used throughout: true = [10, 20, 30], pred = [12, 18, 33]
so d = pred - true = [+2, -2, +3] (late, early, late).
"""
import math

import numpy as np
import pandas as pd
import pytest

from cmapss.metrics import (
    bias,
    evaluate,
    mae,
    nasa_score,
    nasa_score_terms,
    rmse,
)

TRUE = [10, 20, 30]
PRED = [12, 18, 33]


# ---------- RMSE / MAE / bias (by hand) ----------

def test_rmse_hand_computed():
    # mean(d^2) = (4 + 4 + 9) / 3 = 17/3 ; rmse = sqrt(17/3)
    assert rmse(TRUE, PRED) == pytest.approx(2.3804761428476167)


def test_mae_hand_computed():
    assert mae(TRUE, PRED) == pytest.approx(7 / 3)      # (2 + 2 + 3) / 3


def test_bias_hand_computed_and_sign():
    assert bias(TRUE, PRED) == pytest.approx(1.0)        # (2 - 2 + 3) / 3, positive = late
    assert bias([50], [40]) == pytest.approx(-10.0)      # early prediction is negative


def test_perfect_prediction_gives_zero_everywhere():
    assert rmse(TRUE, TRUE) == 0 and mae(TRUE, TRUE) == 0 and bias(TRUE, TRUE) == 0
    assert nasa_score(TRUE, TRUE) == 0


def test_rmse_is_symmetric_in_error_sign():
    assert rmse([50], [60]) == rmse([50], [40]) == pytest.approx(10.0)


def test_accepts_lists_arrays_and_series():
    t, p = np.array(TRUE), np.array(PRED)
    assert rmse(t, p) == rmse(TRUE, PRED) == rmse(pd.Series(TRUE), pd.Series(PRED))


# ---------- asymmetric score (by hand) ----------

def test_score_hand_computed():
    # d = +2 -> e^(2/10) - 1 ; d = -2 -> e^(2/13) - 1 ; d = +3 -> e^(3/10) - 1
    expected = (math.exp(0.2) - 1) + (math.exp(2 / 13) - 1) + (math.exp(0.3) - 1)
    assert nasa_score(TRUE, PRED) == pytest.approx(expected)
    assert nasa_score(TRUE, PRED) == pytest.approx(0.7375730061821031)   # same number, evaluated separately


def test_score_terms_per_engine():
    terms = nasa_score_terms(TRUE, PRED)
    np.testing.assert_allclose(terms, [0.22140275816016985, 0.16631144044593005, 0.3498588075760032])


def test_score_exact_reference_points():
    """Late error of 10 and early error of 13 both give exactly e - 1."""
    assert nasa_score([100], [110]) == pytest.approx(math.e - 1)
    assert nasa_score([100], [87]) == pytest.approx(math.e - 1)


def test_zero_error_scores_zero_and_is_counted_as_late_branch():
    assert nasa_score([42], [42]) == 0.0


def test_score_is_sum_over_engines():
    a = nasa_score([10, 20], [12, 18])
    b = nasa_score([30], [33])
    assert nasa_score(TRUE, PRED) == pytest.approx(a + b)


# ---------- the a1 / a2 check against the paper ----------

def test_late_errors_are_penalised_more_than_early_errors():
    """Intent stated in Saxena et al. (2008): late predictions penalised more than early.
    Same error size (10 cycles), opposite direction."""
    late = nasa_score([100], [110])      # e^(10/10) - 1 = 1.7183
    early = nasa_score([100], [90])      # e^(10/13) - 1 = 1.1581
    assert late > early
    assert early == pytest.approx(1.1581055339484458)


def test_printed_constants_in_paper_eq11_contradict_its_own_text():
    """Eq. 11 as printed gives a1 = 10 to d < 0 and a2 = 13 to d >= 0. With that assignment an early
    error is penalised MORE than a late one, which contradicts the text. This test documents the
    discrepancy; the module's defaults use the corrected assignment (13 early, 10 late)."""
    printed_early = nasa_score([100], [90], a_early=10, a_late=13)
    printed_late = nasa_score([100], [110], a_early=10, a_late=13)
    assert printed_early > printed_late                       # contradicts "late is penalised more"
    assert nasa_score([100], [110]) > nasa_score([100], [90])  # defaults agree with the text


def test_score_grows_exponentially_with_error():
    s10, s20, s40 = (nasa_score([200], [200 + e]) for e in (10, 20, 40))
    assert s20 > 2 * s10 and s40 > 2 * s20       # faster than linear


# ---------- evaluate ----------

def test_evaluate_collects_everything():
    r = evaluate(TRUE, PRED)
    assert set(r) == {"rmse", "mae", "bias", "score", "n", "n_late", "n_early"}
    assert r["n"] == 3 and r["n_late"] == 2 and r["n_early"] == 1
    assert r["rmse"] == pytest.approx(rmse(TRUE, PRED))
    assert r["score"] == pytest.approx(nasa_score(TRUE, PRED))


def test_evaluate_exact_prediction_counts_neither_late_nor_early():
    r = evaluate([5, 6], [5, 6])
    assert r["n_late"] == 0 and r["n_early"] == 0


# ---------- input validation ----------

@pytest.mark.parametrize("fn", [rmse, mae, bias, nasa_score, evaluate])
def test_length_mismatch_raises(fn):
    with pytest.raises(ValueError, match="length mismatch"):
        fn([1, 2, 3], [1, 2])


@pytest.mark.parametrize("fn", [rmse, mae, bias, nasa_score, evaluate])
def test_empty_raises(fn):
    with pytest.raises(ValueError, match="empty"):
        fn([], [])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_raises(bad):
    with pytest.raises(ValueError, match="NaN|infinite"):
        rmse([1, 2], [1, bad])
    with pytest.raises(ValueError, match="NaN|infinite"):
        nasa_score([1, bad], [1, 2])


def test_non_1d_raises():
    with pytest.raises(ValueError, match="1-D"):
        rmse([[1, 2], [3, 4]], [[1, 2], [3, 4]])


@pytest.mark.parametrize("kw", [{"a_early": 0}, {"a_late": -1}])
def test_nonpositive_constants_raise(kw):
    with pytest.raises(ValueError, match="positive"):
        nasa_score(TRUE, PRED, **kw)
