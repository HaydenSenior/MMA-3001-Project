"""Baseline regression models written in NumPy.

Both models follow the same minimal interface: ``fit(X, y)`` returns the model
and ``predict(X)`` returns a 1-D array.

``RidgeRegressor`` is implemented by hand (rather than imported) so that its
behaviour can be verified against hand-solved cases and against scikit-learn
(``tests/test_models.py``), and so that its cost can be analysed later.
"""

from __future__ import annotations

import numpy as np


def _check_xy(X: np.ndarray, y: np.ndarray) -> None:
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError(f"X must be 2-D and y 1-D, got {X.shape} and {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
    if X.shape[0] < 2:
        raise ValueError("need at least 2 samples")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("X and y must not contain NaN or infinite values")


class MeanBaseline:
    """Predicts the mean of the training target for every input (the weakest sensible floor)."""

    def fit(self, X, y) -> "MeanBaseline":
        """Store the mean of ``y``. ``X`` is validated but otherwise ignored."""
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        _check_xy(X, y)
        self.mean_ = float(y.mean())
        return self

    def predict(self, X) -> np.ndarray:
        """Return the stored mean once per row of ``X``."""
        if not hasattr(self, "mean_"):
            raise RuntimeError("model is not fitted")
        return np.full(np.asarray(X).shape[0], self.mean_)


class RidgeRegressor:
    """Ridge regression (L2-penalised least squares) with standardised inputs.

    Minimises ``||y - b - Z w||^2 + alpha * ||w||^2`` where ``Z`` is ``X`` with
    each column centred and scaled to unit variance using the training data.
    The intercept ``b`` is not penalised. Standardising first means ``alpha``
    treats all features on the same scale regardless of their units (the
    sensors range from 0.03 to 9000).

    The problem is solved as a least-squares system on ``[Z; sqrt(alpha) I]``
    (not by forming ``Z^T Z``), which stays accurate when features are highly
    correlated, and works for ``alpha = 0`` (ordinary least squares).

    Args:
        alpha: Penalty strength, ``>= 0``.

    Attributes:
        coef_: Coefficients in the original units of ``X``, shape ``(p,)``.
        intercept_: Intercept for the original units of ``X``.
    """

    def __init__(self, alpha: float = 1.0):
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.alpha = float(alpha)

    def fit(self, X, y) -> "RidgeRegressor":
        """Fit on ``X`` of shape ``(n, p)`` and ``y`` of shape ``(n,)``.

        A column with zero variance in the training data gets coefficient 0.

        Raises:
            ValueError: On bad shapes, fewer than 2 samples, or NaN/inf values.
        """
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        _check_xy(X, y)
        n, p = X.shape
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)          # constant column: centred to 0, coefficient 0
        Z = (X - mu) / sd
        y_bar = y.mean()
        A = np.vstack([Z, np.sqrt(self.alpha) * np.eye(p)])
        b = np.concatenate([y - y_bar, np.zeros(p)])
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
        self.coef_ = w / sd
        self.intercept_ = float(y_bar - mu @ self.coef_)
        return self

    def predict(self, X) -> np.ndarray:
        """Return ``X @ coef_ + intercept_``.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``X`` has the wrong number of columns or non-finite values.
        """
        if not hasattr(self, "coef_"):
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.coef_.shape[0]:
            raise ValueError(f"expected X with {self.coef_.shape[0]} columns, got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X must not contain NaN or infinite values")
        return X @ self.coef_ + self.intercept_
