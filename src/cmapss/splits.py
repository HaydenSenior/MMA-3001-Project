"""Validation splits that keep every engine entirely on one side of the split.

Rows from one engine are near-duplicates of their neighbours (cycle t and t+1
differ by little more than noise), so a row-level split puts almost the same
data in both train and validation and gives optimistic scores. All splitting in
this project is therefore done **by unit (engine)**.

Only the *training* file is split here. The official test file and RUL vector
are held out and used once, at the end.

Example:
    >>> folds = unit_kfold_splits(train_df, n_splits=5, seed=0)   # doctest: +SKIP
    >>> for train_idx, val_idx in folds:                          # doctest: +SKIP
    ...     X_tr, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

Fold = Tuple[np.ndarray, np.ndarray]


class LeakageError(ValueError):
    """Raised when an engine appears in both the train and validation rows."""


def assert_no_unit_overlap(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    """Check that no engine has rows in both ``train_idx`` and ``val_idx``.

    Args:
        df: Frame with a ``unit`` column.
        train_idx: Positional row indices (as used with ``df.iloc``) for training.
        val_idx: Positional row indices for validation.

    Raises:
        LeakageError: If any unit appears in both index sets.
        ValueError: If ``df`` has no ``unit`` column.
    """
    if "unit" not in df.columns:
        raise ValueError("df must have a 'unit' column")
    units = df["unit"].to_numpy()
    shared = np.intersect1d(units[np.asarray(train_idx)], units[np.asarray(val_idx)])
    if shared.size:
        raise LeakageError(
            f"{shared.size} engine(s) appear in both train and validation, e.g. unit {int(shared[0])}"
        )


def unit_kfold_splits(
    df: pd.DataFrame, n_splits: int = 5, shuffle: bool = True, seed: int = 0
) -> List[Fold]:
    """K-fold cross-validation splits grouped by engine.

    Engines (not rows) are divided into ``n_splits`` folds whose engine counts
    differ by at most one. Every engine is in exactly one validation fold. Each
    returned pair is checked with :func:`assert_no_unit_overlap` before it is
    returned.

    Args:
        df: Training frame with a ``unit`` column (e.g. from ``loader.load_raw``).
        n_splits: Number of folds, at least 2 and at most the number of engines.
        shuffle: If True, assign engines to folds at random (reproducible via
            ``seed``); if False, fold k holds consecutive unit numbers.
        seed: Seed for the shuffle. Ignored when ``shuffle`` is False.

    Returns:
        List of ``(train_idx, val_idx)`` pairs of sorted positional row indices
        for use with ``df.iloc``.

    Raises:
        ValueError: If ``df`` has no ``unit`` column or ``n_splits`` is outside
            ``[2, n_engines]``.
        LeakageError: If a split were to share an engine (internal safety check).
    """
    if "unit" not in df.columns:
        raise ValueError("df must have a 'unit' column")
    units = np.unique(df["unit"].to_numpy())
    if not (2 <= n_splits <= len(units)):
        raise ValueError(f"n_splits must be between 2 and {len(units)} engines, got {n_splits}")

    if shuffle:
        units = np.random.default_rng(seed).permutation(units)
    unit_folds = np.array_split(units, n_splits)

    row_units = df["unit"].to_numpy()
    folds: List[Fold] = []
    for held_out in unit_folds:
        in_val = np.isin(row_units, held_out)
        train_idx, val_idx = np.flatnonzero(~in_val), np.flatnonzero(in_val)
        assert_no_unit_overlap(df, train_idx, val_idx)
        folds.append((train_idx, val_idx))
    return folds
