"""Causal per-engine noise and offset reduction applied to sensor columns before feature extraction.

"Causal" means the value at cycle ``t`` uses only cycles ``<= t`` of the *same engine*,
so the transforms can be applied to a test engine's observed history exactly as to a
training engine, and never leak future information or mix engines.

* :func:`ewma`: exponentially weighted moving average (smooths measurement noise).
* :func:`rolling_median`: trailing median (suppresses spikes and quantisation jumps).
* :func:`baseline_normalise`: subtract each engine's own early-life mean (removes the
  engine-to-engine offset caused by initial wear and manufacturing variation).

Each function returns a copy of the frame; only the listed columns change.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _check(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in ["unit", "cycle", *columns] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if not np.isfinite(df[list(columns)].to_numpy()).all():
        raise ValueError("columns contain NaN or infinite values")


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def ewma(df: pd.DataFrame, columns: Sequence[str], span: float = 10) -> pd.DataFrame:
    """Exponentially weighted moving average per engine.

    ``s_t = alpha * x_t + (1 - alpha) * s_{t-1}`` with ``alpha = 2 / (span + 1)`` and
    ``s_1 = x_1``. A larger ``span`` smooths more but lags the true signal more.

    Args:
        df: Frame with ``unit``, ``cycle`` and ``columns``.
        columns: Columns to smooth.
        span: Smoothing span in cycles, ``>= 1``.

    Returns:
        Copy sorted by unit and cycle with ``columns`` replaced by their EWMA.

    Raises:
        ValueError: On ``span < 1``, missing columns, or non-finite values.
    """
    if span < 1:
        raise ValueError("span must be >= 1")
    columns = list(columns); _check(df, columns)
    out = _sorted(df)
    out[columns] = out.groupby("unit")[columns].transform(lambda s: s.ewm(span=span, adjust=False).mean())
    return out


def rolling_median(df: pd.DataFrame, columns: Sequence[str], window: int = 5) -> pd.DataFrame:
    """Trailing median over the last ``window`` cycles per engine (fewer at the start of an engine).

    Args:
        df: Frame with ``unit``, ``cycle`` and ``columns``.
        columns: Columns to filter.
        window: Window length in cycles, a positive integer.

    Returns:
        Copy sorted by unit and cycle with ``columns`` replaced by their trailing median.

    Raises:
        ValueError: On a bad ``window``, missing columns, or non-finite values.
    """
    if int(window) != window or window < 1:
        raise ValueError("window must be a positive integer")
    columns = list(columns); _check(df, columns)
    out = _sorted(df)
    out[columns] = out.groupby("unit")[columns].transform(lambda s: s.rolling(int(window), min_periods=1).median())
    return out


def baseline_normalise(df: pd.DataFrame, columns: Sequence[str], n_baseline: int = 30) -> pd.DataFrame:
    """Subtract each engine's own early-life mean from every column.

    At cycle ``t`` the baseline is the mean of that engine's first ``min(t, n_baseline)``
    cycles, so it uses only cycles ``<= t`` (an expanding mean until ``n_baseline`` cycles have
    been seen, then fixed). From cycle ``n_baseline`` on, a column's value is its change
    relative to the engine's own start. This removes the offset between engines caused by
    initial wear, which the EDA showed is large compared with the degradation trend.

    Args:
        df: Frame with ``unit``, ``cycle`` and ``columns``.
        columns: Columns to normalise.
        n_baseline: Number of initial cycles that define the baseline, a positive integer.

    Returns:
        Copy sorted by unit and cycle with ``columns`` replaced by ``x_t - baseline_t``.

    Raises:
        ValueError: On a bad ``n_baseline``, missing columns, or non-finite values.
    """
    if int(n_baseline) != n_baseline or n_baseline < 1:
        raise ValueError("n_baseline must be a positive integer")
    columns = list(columns); _check(df, columns)
    out = _sorted(df)
    k = int(n_baseline)

    def one(s: pd.Series) -> pd.Series:
        base = s.expanding().mean().to_numpy().copy()
        base[k:] = base[k - 1] if len(s) >= k else base[-1]       # frozen once k cycles are seen
        return s - base

    out[columns] = out.groupby("unit")[columns].transform(one)
    return out
