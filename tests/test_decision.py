"""Tests for cmapss.decision: hand-computed costs and shifts, and the cost/quantile link."""
import numpy as np
import pytest

from cmapss.decision import apply_shift, best_shift, expected_cost, optimal_quantile
from cmapss.metrics import nasa_score, rmse


# ---------- apply_shift ----------

def test_apply_shift_hand_computed_with_clipping():
    np.testing.assert_allclose(apply_shift([100.0, 20.0, 5.0], 10, cap=125), [90.0, 10.0, 0.0])


def test_zero_shift_is_identity_and_cap_is_applied():
    np.testing.assert_allclose(apply_shift([10.0, 130.0], 0), [10.0, 130.0])
    np.testing.assert_allclose(apply_shift([10.0, 130.0], 0, cap=125), [10.0, 125.0])


def test_negative_shift_moves_predictions_later():
    np.testing.assert_allclose(apply_shift([50.0], -5), [55.0])


def test_apply_shift_validates_input():
    with pytest.raises(ValueError, match="1-D"):
        apply_shift([[1.0, 2.0]], 1)
    with pytest.raises(ValueError, match="NaN|infinite"):
        apply_shift([1.0, np.nan], 1)


# ---------- expected_cost ----------

def test_expected_cost_hand_computed():
    """true [100, 100], pred [110, 90] -> d = +10 (late), -10 (early).
    c_early 1, c_late 10 -> (10*10 + 1*10) / 2 = 55."""
    assert expected_cost([100, 100], [110, 90], c_early=1, c_late=10) == pytest.approx(55.0)


def test_symmetric_costs_give_mean_absolute_error():
    t, p = [10.0, 20.0, 30.0], [12.0, 18.0, 33.0]
    assert expected_cost(t, p, 1, 1) == pytest.approx(7 / 3)


def test_perfect_prediction_costs_nothing_and_late_costs_more():
    assert expected_cost([50.0], [50.0], 1, 10) == 0.0
    assert expected_cost([50.0], [60.0], 1, 10) > expected_cost([50.0], [40.0], 1, 10)


def test_expected_cost_validates_input():
    with pytest.raises(ValueError, match="length mismatch"):
        expected_cost([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="empty"):
        expected_cost([], [])
    with pytest.raises(ValueError, match="non-negative"):
        expected_cost([1.0], [1.0], c_early=-1)


# ---------- optimal_quantile ----------

def test_optimal_quantile_hand_computed():
    assert optimal_quantile(1, 10) == pytest.approx(1 / 11)
    assert optimal_quantile(1, 1) == pytest.approx(0.5)      # symmetric cost -> median
    assert optimal_quantile(0, 5) == 0.0


def test_optimal_quantile_matches_the_empirical_minimiser():
    """For a constant prediction against a known distribution, the cost-minimising value is that quantile."""
    rng = np.random.default_rng(0)
    truth = rng.normal(100, 20, 20000)
    for c_late in (1.0, 4.0, 10.0):
        alpha = optimal_quantile(1.0, c_late)
        grid = np.arange(40, 160, 0.5)
        costs = [expected_cost(truth, np.full_like(truth, v), 1.0, c_late) for v in grid]
        assert grid[int(np.argmin(costs))] == pytest.approx(np.quantile(truth, alpha), abs=1.5)


def test_optimal_quantile_errors():
    with pytest.raises(ValueError, match="positive"):
        optimal_quantile(0, 0)


# ---------- best_shift ----------

def test_best_shift_finds_a_known_constant_bias():
    rng = np.random.default_rng(1)
    truth = rng.uniform(20, 120, 4000)
    pred = truth + 15 + rng.normal(0, 3, 4000)          # systematically 15 cycles late
    assert best_shift(truth, pred, lambda t, p: rmse(t, p), np.arange(0, 31, 1)) == pytest.approx(15, abs=1)


def test_asymmetric_objective_chooses_a_larger_shift_than_rmse():
    rng = np.random.default_rng(2)
    truth = rng.uniform(20, 120, 3000)
    pred = truth + rng.normal(0, 12, 3000)              # unbiased but noisy
    grid = np.arange(0, 41, 1)
    d_rmse = best_shift(truth, pred, lambda t, p: rmse(t, p), grid)
    d_score = best_shift(truth, pred, nasa_score, grid)
    d_cost = best_shift(truth, pred, lambda t, p: expected_cost(t, p, 1, 10), grid)
    assert d_rmse < 3
    assert d_score > d_rmse and d_cost > d_rmse


def test_best_shift_respects_the_grid_and_cap():
    truth = np.array([100.0] * 10)
    pred = truth + 50
    assert best_shift(truth, pred, lambda t, p: rmse(t, p), [0, 5, 10]) == 10      # best available, not 50
    assert best_shift(truth, pred, lambda t, p: rmse(t, p), [0], cap=125) == 0


def test_best_shift_empty_grid_raises():
    with pytest.raises(ValueError, match="grid is empty"):
        best_shift([1.0], [1.0], lambda t, p: rmse(t, p), [])
