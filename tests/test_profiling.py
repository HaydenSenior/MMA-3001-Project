"""Tests for cmapss.profiling: hand-computed FLOP counts, scaling, timing and memory behaviour."""
import time

import numpy as np
import pytest

from cmapss.profiling import (
    arithmetic_intensity,
    model_size_bytes,
    peak_memory,
    ridge_fit_flops,
    ridge_predict_bytes,
    ridge_predict_flops,
    time_callable,
)


# ---------- FLOP counts ----------

def test_predict_flops_hand_computed():
    assert ridge_predict_flops(10, 3) == 60.0            # 2 * 10 * 3
    assert ridge_predict_flops(1, 1) == 2.0


def test_predict_flops_scale_linearly_in_both_arguments():
    assert ridge_predict_flops(2000, 42) == 2 * ridge_predict_flops(1000, 42)
    assert ridge_predict_flops(1000, 84) == 2 * ridge_predict_flops(1000, 42)


def test_predict_bytes_hand_computed():
    assert ridge_predict_bytes(10, 3) == 8 * (30 + 3 + 10)


def test_predict_is_memory_bound_at_just_under_a_quarter_flop_per_byte():
    """2np flops over 8(np + p + n) bytes gives p / (4(p + 1)), which approaches 0.25 as the
    number of FEATURES grows (not as the number of rows grows: the output write scales with n)."""
    for p in (14, 42, 1000):
        ai = arithmetic_intensity(ridge_predict_flops(100000, p), ridge_predict_bytes(100000, p))
        assert ai == pytest.approx(p / (4 * (p + 1)), rel=0.01)
        assert 0.2 < ai < 0.25


def test_fit_flops_hand_computed_against_the_documented_formula():
    n, p = 1000, 10
    expected = 6 * n * p + 2 * (n + p) * p**2 - (2 / 3) * p**3 + p**2
    assert ridge_fit_flops(n, p) == pytest.approx(expected)


def test_fit_flops_grow_linearly_in_rows_and_faster_than_linearly_in_features():
    a, b = ridge_fit_flops(10000, 42), ridge_fit_flops(20000, 42)
    assert b / a == pytest.approx(2.0, rel=0.02)          # n >> p, so doubling rows doubles work
    c, d = ridge_fit_flops(100000, 20), ridge_fit_flops(100000, 40)
    assert d / c > 3.0                                    # quadratic in p


def test_fit_costs_more_than_predict_for_the_same_data():
    assert ridge_fit_flops(17731, 42) > ridge_predict_flops(17731, 42)


@pytest.mark.parametrize("fn", [ridge_fit_flops, ridge_predict_flops, ridge_predict_bytes])
def test_flop_functions_reject_bad_sizes(fn):
    with pytest.raises(ValueError):
        fn(0, 5)
    with pytest.raises(ValueError):
        fn(5, 0)


def test_arithmetic_intensity_hand_computed_and_validated():
    assert arithmetic_intensity(100, 50) == 2.0
    with pytest.raises(ValueError, match="bytes_moved"):
        arithmetic_intensity(10, 0)
    with pytest.raises(ValueError, match="flops"):
        arithmetic_intensity(-1, 10)


# ---------- timing ----------

def test_time_callable_structure_and_repeat_count():
    calls = []
    r = time_callable(lambda: calls.append(1), repeats=5, warmup=2)
    assert len(calls) == 7                                 # warmups are run but not timed
    assert r["repeats"] == 5
    assert set(r) == {"min", "median", "mean", "std", "repeats"}
    assert r["min"] <= r["median"] <= max(r["mean"], r["median"])


def test_time_callable_measures_a_slower_function_as_slower():
    fast = time_callable(lambda: None, repeats=3, warmup=0)
    slow = time_callable(lambda: time.sleep(0.02), repeats=3, warmup=0)
    assert slow["min"] > fast["min"]
    assert slow["min"] >= 0.015


def test_time_callable_single_repeat_has_zero_spread():
    assert time_callable(lambda: None, repeats=1, warmup=0)["std"] == 0.0


@pytest.mark.parametrize("kwargs", [{"repeats": 0}, {"warmup": -1}])
def test_time_callable_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        time_callable(lambda: None, **kwargs)


# ---------- memory ----------

def test_peak_memory_sees_a_known_numpy_allocation():
    result, peak = peak_memory(lambda: np.zeros(1_000_000))     # 8 MB of float64
    assert result.shape == (1_000_000,)
    assert peak >= 7_500_000


def test_peak_memory_is_small_for_a_trivial_call_and_nests_safely():
    _, outer = peak_memory(lambda: peak_memory(lambda: np.zeros(10)))
    assert outer < 1_000_000


# ---------- model size ----------

def test_model_size_grows_with_the_model_and_rejects_unpicklable_objects():
    small = model_size_bytes(np.zeros(10))
    large = model_size_bytes(np.zeros(10000))
    assert large > small > 0
    with pytest.raises(ValueError, match="cannot be pickled"):
        model_size_bytes(lambda x: x)
