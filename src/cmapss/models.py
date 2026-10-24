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


def boosting_factory(iters: int = 300, leaves: int = 8, min_leaf: int = 200, lr: float = 0.05,
                     loss: str = "squared_error", quantile: float | None = None):
    """Return a zero-argument factory for scikit-learn's histogram gradient boosting.

    Early stopping is switched OFF on purpose. scikit-learn's default ('auto')
    holds out a random 10% of *rows* to decide when to stop, and rows of one
    engine are near-duplicates, so that check would leak across the split.

    Args:
        iters: Number of boosting iterations (trees).
        leaves: Maximum leaves per tree.
        min_leaf: Minimum samples per leaf.
        lr: Learning rate.
        loss: scikit-learn loss name. ``"quantile"`` (with ``quantile``) trains the model to
            predict a quantile rather than the mean, which is how an asymmetric maintenance
            cost is targeted directly (see :mod:`cmapss.decision`).
        quantile: Quantile in (0, 1); required when ``loss="quantile"``, ignored otherwise.
            Values below 0.5 bias predictions early (safer).

    Returns:
        Callable returning a fresh, unfitted ``HistGradientBoostingRegressor``.

    Raises:
        ImportError: When called without scikit-learn installed.
        ValueError: If ``loss="quantile"`` and ``quantile`` is not in (0, 1).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    if loss == "quantile" and not (quantile is not None and 0.0 < quantile < 1.0):
        raise ValueError(f"loss='quantile' needs quantile in (0, 1), got {quantile!r}")
    extra = {"quantile": quantile} if loss == "quantile" else {}
    return lambda: HistGradientBoostingRegressor(
        learning_rate=lr, max_iter=iters, max_leaf_nodes=leaves, min_samples_leaf=min_leaf,
        l2_regularization=1.0, early_stopping=False, random_state=0, loss=loss, **extra)


class PCAPipeline:
    """Standardise features, project onto the first ``n_components`` principal components, then fit a model.

    Correlated sensors repeat the same noisy signal; keeping only the strongest shared directions
    discards some of that noise. The standardisation and projection are learned from the training
    rows passed to ``fit`` only, so no validation information leaks in when used inside cross-validation.

    Args:
        base_factory: Zero-argument callable returning a fresh model with ``fit`` and ``predict``.
        n_components: Number of principal components kept, at least 1.
    """

    def __init__(self, base_factory, n_components: int):
        if int(n_components) != n_components or n_components < 1:
            raise ValueError("n_components must be a positive integer")
        self.base_factory, self.n_components = base_factory, int(n_components)

    def fit(self, X, y) -> "PCAPipeline":
        """Learn the projection from ``X`` and fit the base model on the projected data.

        Raises:
            ValueError: On bad inputs (see the other models) or ``n_components`` larger than the number of columns.
        """
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        _check_xy(X, y)
        if self.n_components > X.shape[1]:
            raise ValueError(f"n_components={self.n_components} exceeds the {X.shape[1]} columns")
        self.mu_, sd = X.mean(axis=0), X.std(axis=0)
        self.sd_ = np.where(sd == 0, 1.0, sd)
        _, s, vt = np.linalg.svd((X - self.mu_) / self.sd_, full_matrices=False)
        self.components_ = vt[: self.n_components]
        self.explained_variance_ratio_ = (s**2 / np.sum(s**2))[: self.n_components]
        self.model_ = self.base_factory().fit(self._project(X), y)
        return self

    def _project(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu_) / self.sd_) @ self.components_.T

    def predict(self, X) -> np.ndarray:
        """Project ``X`` with the learned components and predict with the base model.

        Raises:
            RuntimeError: If not fitted.
            ValueError: On the wrong number of columns or non-finite values.
        """
        if not hasattr(self, "model_"):
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.mu_.shape[0]:
            raise ValueError(f"expected X with {self.mu_.shape[0]} columns, got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X must not contain NaN or infinite values")
        return self.model_.predict(self._project(X))
