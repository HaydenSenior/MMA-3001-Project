"""Turning a RUL prediction into a maintenance decision under asymmetric costs.

A late prediction (the model says more life remains than there is) risks an in-service
failure; an early one wastes remaining life. The two are not equally bad, so the
prediction a maintenance planner should act on is not the one that minimises RMSE.

Sign convention, as in :mod:`cmapss.metrics`: ``d = predicted - true``, so ``d > 0``
is late and ``d < 0`` is early.

This module provides the simplest way to act on that asymmetry: shift every prediction
down by a constant number of cycles, and choose the shift from an explicit cost model.

Linear cost model
    ``cost = c_early * max(0, -d) + c_late * max(0, d)`` per prediction.

This is the pinball (quantile) loss up to a scale factor, so the cost-minimising
prediction is the quantile ``alpha = c_early / (c_early + c_late)`` of the predictive
distribution. A model trained with a quantile loss at that ``alpha`` targets the same
decision directly; :func:`optimal_quantile` returns it.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def apply_shift(y_pred: ArrayLike, delta: float, cap: Optional[float] = None) -> np.ndarray:
    """Subtract ``delta`` cycles from every prediction and clip to ``[0, cap]``.

    Args:
        y_pred: Predicted RUL in cycles.
        delta: Cycles to subtract. Positive moves predictions earlier (safer).
        cap: Upper clip, or None for no upper clip. The lower clip is always 0,
            because RUL cannot be negative.

    Returns:
        1-D float array of shifted predictions.

    Raises:
        ValueError: If ``y_pred`` is not 1-D or contains NaN or infinite values.
    """
    p = np.asarray(y_pred, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"y_pred must be 1-D, got shape {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("y_pred contains NaN or infinite values")
    return np.clip(p - float(delta), 0.0, cap)


def expected_cost(y_true: ArrayLike, y_pred: ArrayLike, c_early: float = 1.0, c_late: float = 10.0) -> float:
    """Mean maintenance cost per prediction under the linear cost model.

    Args:
        y_true: True RUL in cycles.
        y_pred: Predicted RUL in cycles, same length.
        c_early: Cost per cycle of predicting too early (wasted remaining life).
        c_late: Cost per cycle of predicting too late (risk of in-service failure).

    Returns:
        ``mean(c_early * max(0, -d) + c_late * max(0, d))`` with ``d = pred - true``.

    Raises:
        ValueError: On length mismatch, empty input, non-finite values, or a negative cost.
    """
    t, p = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: {t.shape} vs {p.shape}")
    if t.size == 0:
        raise ValueError("inputs are empty")
    if not (np.isfinite(t).all() and np.isfinite(p).all()):
        raise ValueError("inputs contain NaN or infinite values")
    if c_early < 0 or c_late < 0:
        raise ValueError("costs must be non-negative")
    d = p - t
    return float(np.mean(c_early * np.maximum(0.0, -d) + c_late * np.maximum(0.0, d)))


def optimal_quantile(c_early: float = 1.0, c_late: float = 10.0) -> float:
    """Quantile that minimises the linear cost model: ``c_early / (c_early + c_late)``.

    With late errors ten times as costly as early ones this is about 0.09, i.e. a model
    should predict close to the 9th percentile of its predictive distribution.

    Raises:
        ValueError: If both costs are zero or either is negative.
    """
    if c_early < 0 or c_late < 0:
        raise ValueError("costs must be non-negative")
    total = c_early + c_late
    if total == 0:
        raise ValueError("at least one cost must be positive")
    return float(c_early / total)


def best_shift(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    objective: Callable[[np.ndarray, np.ndarray], float],
    grid: Sequence[float],
    cap: Optional[float] = None,
) -> float:
    """Shift from ``grid`` that minimises ``objective(y_true, shifted_predictions)``.

    Intended to be called on *training* data only (for example inner-fold predictions),
    so the chosen shift can then be applied to unseen engines without tuning on them.

    Args:
        y_true: True RUL in cycles.
        y_pred: Predicted RUL in cycles.
        objective: Callable taking ``(y_true, y_pred)`` and returning a value to minimise,
            such as ``cmapss.metrics.nasa_score`` or :func:`expected_cost`.
        grid: Candidate shifts in cycles.
        cap: Upper clip passed to :func:`apply_shift`.

    Returns:
        The best shift found. Ties go to the smallest shift in ``grid`` order.

    Raises:
        ValueError: If ``grid`` is empty, or on invalid inputs (see :func:`apply_shift`).
    """
    grid = list(grid)
    if not grid:
        raise ValueError("grid is empty")
    t = np.asarray(y_true, dtype=np.float64)
    values = [objective(t, apply_shift(y_pred, d, cap)) for d in grid]
    return float(grid[int(np.argmin(values))])
