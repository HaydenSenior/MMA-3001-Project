"""Error metrics for remaining-useful-life (RUL) prediction.

Sign convention used everywhere: ``d = predicted - true``.
``d > 0`` is a **late** prediction (the model says the engine has more life
left than it does; dangerous). ``d < 0`` is an **early** prediction (the model
is conservative; wasteful but safe).

The asymmetric score is the one defined for the PHM'08 challenge by Saxena et
al. (2008)::

    s = sum_i ( exp(-d_i / 13) - 1   if d_i < 0     (early)
                exp( d_i / 10) - 1   if d_i >= 0    (late) )

Note on the constants: eq. (11) of the paper, *as printed in the copy used for
this project*, labels the constants ``a1 = 10`` for ``d < 0`` and ``a2 = 13``
for ``d >= 0``. That assignment would penalise early errors *more* than late
ones, which contradicts the paper's own text ("late predictions were more
heavily penalized than early predictions"). The widely used form, which agrees
with the text, divides by 13 for early errors and 10 for late errors, and is
used here. ``nasa_score`` exposes both constants so the printed assignment can
be reproduced for comparison (see ``tests/test_metrics.py``).

Example:
    >>> evaluate([10, 20, 30], [12, 18, 33])["bias"]
    1.0
"""

from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]

A_EARLY = 13.0  # divisor for early predictions (d < 0)
A_LATE = 10.0   # divisor for late predictions (d >= 0)


def _errors(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """Return ``d = y_pred - y_true`` as a validated 1-D float array.

    Raises:
        ValueError: If the inputs are not 1-D, differ in length, are empty, or
            contain NaN or infinite values.
    """
    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if t.ndim != 1 or p.ndim != 1:
        raise ValueError(f"inputs must be 1-D, got shapes {t.shape} and {p.shape}")
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: {t.shape[0]} true values vs {p.shape[0]} predictions")
    if t.size == 0:
        raise ValueError("inputs are empty")
    if not (np.isfinite(t).all() and np.isfinite(p).all()):
        raise ValueError("inputs contain NaN or infinite values")
    return p - t


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean square error, in cycles. Treats early and late errors equally.

    Args:
        y_true: True RUL per engine (cycles).
        y_pred: Predicted RUL per engine (cycles), same length.

    Returns:
        ``sqrt(mean(d**2))``.

    Raises:
        ValueError: On invalid input (see ``_errors``).
    """
    d = _errors(y_true, y_pred)
    return float(np.sqrt(np.mean(d**2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error, in cycles. Less dominated by outliers than RMSE."""
    return float(np.mean(np.abs(_errors(y_true, y_pred))))


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean signed error ``mean(pred - true)`` in cycles.

    Positive means the model is late on average, negative means early.
    """
    return float(np.mean(_errors(y_true, y_pred)))


def nasa_score_terms(
    y_true: ArrayLike, y_pred: ArrayLike, a_early: float = A_EARLY, a_late: float = A_LATE
) -> np.ndarray:
    """Per-engine terms of the asymmetric PHM'08 score.

    Args:
        y_true: True RUL per engine (cycles).
        y_pred: Predicted RUL per engine (cycles).
        a_early: Divisor for early predictions (``d < 0``). Default 13.
        a_late: Divisor for late predictions (``d >= 0``). Default 10.

    Returns:
        Array of ``exp(-d/a_early) - 1`` where ``d < 0`` and
        ``exp(d/a_late) - 1`` where ``d >= 0``. All terms are >= 0.

    Raises:
        ValueError: On invalid input, or if ``a_early`` or ``a_late`` is not positive.
    """
    if a_early <= 0 or a_late <= 0:
        raise ValueError("a_early and a_late must be positive")
    d = _errors(y_true, y_pred)
    return np.where(d < 0, np.exp(-d / a_early) - 1.0, np.exp(d / a_late) - 1.0)


def nasa_score(
    y_true: ArrayLike, y_pred: ArrayLike, a_early: float = A_EARLY, a_late: float = A_LATE
) -> float:
    """Asymmetric PHM'08 score: the sum of :func:`nasa_score_terms`. Lower is better.

    The score grows exponentially with the error, so a single bad late
    prediction can dominate it, and it grows with the number of engines
    (it is a sum, not a mean), so scores are only comparable on the same test
    set. Always report it together with RMSE. A very large error can overflow
    to ``inf``.
    """
    return float(np.sum(nasa_score_terms(y_true, y_pred, a_early, a_late)))


def evaluate(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """All metrics at once.

    Returns:
        Dict with ``rmse``, ``mae``, ``bias``, ``score`` (asymmetric), ``n``,
        ``n_late`` (``d > 0``) and ``n_early`` (``d < 0``).
    """
    d = _errors(y_true, y_pred)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "score": nasa_score(y_true, y_pred),
        "n": int(d.size),
        "n_late": int((d > 0).sum()),
        "n_early": int((d < 0).sum()),
    }
