"""Column screening and degradation-onset analysis for training engines.

These helpers support the exploratory analysis (Phase 1). They work on
*training* data only, because both rely on seeing each engine run to failure.

Scope: the trend-based screening assumes a single operating condition
(FD001, FD003). With six operating conditions (FD002, FD004) a raw sensor
swings with the flight condition rather than with wear, so its per-engine
trend over cycles is not a meaningful measure of degradation there.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def constant_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    """Columns that take a single value across the whole frame (zero variance).

    Args:
        df: Frame containing ``columns``.
        columns: Names of the columns to test.

    Returns:
        Names of the columns with at most one distinct value, in input order.
    """
    return [c for c in columns if df[c].nunique() <= 1]


def trend_strength(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """How consistently each column trends with cycle, within engines.

    For each engine the Spearman rank correlation between the column and
    ``cycle`` is computed. The result is the median over engines of its absolute
    value: 1 means a perfectly monotonic trend in every engine, values near 0
    mean no consistent trend (noise). A constant column scores 0.

    Spearman is used because it picks up any monotonic trend (not only linear)
    and is insensitive to the unit and scale of the sensor. Ties are given average ranks.

    The whole frame is ranked and correlated per engine in a vectorised pass; profiling showed
    the previous column-by-column pandas version dominated the end-to-end pipeline.

    Args:
        df: Training frame with ``unit``, ``cycle`` and ``columns``.
        columns: Names of the columns to score.

    Returns:
        Series indexed by column name, values in [0, 1].
    """
    columns = list(columns)
    per_engine = []
    for _, g in df.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        # Spearman is Pearson on ranks. Cycles are consecutive within an engine, so their
        # ranks are simply 1..n and only the columns need ranking. All columns are ranked and
        # correlated in one vectorised pass rather than one pandas call per column per engine.
        ranks = rankdata(g[columns].to_numpy(dtype=np.float64), axis=0)
        centred = ranks - ranks.mean(axis=0)
        t = np.arange(1, len(g) + 1, dtype=np.float64)
        t -= t.mean()
        denom = np.sqrt((centred**2).sum(axis=0) * (t**2).sum())
        # A constant column has zero rank variance, so its denominator is 0: score it 0, not NaN.
        rho = np.divide(centred.T @ t, denom, out=np.zeros(len(columns)), where=denom > 0)
        per_engine.append(np.abs(rho))
    return pd.Series(np.median(np.vstack(per_engine), axis=0), index=columns, name="trend_strength")


def select_columns(
    df: pd.DataFrame, columns: Sequence[str], min_trend: float = 0.3
) -> Tuple[List[str], pd.DataFrame]:
    """Drop constant columns, then columns without a consistent per-engine trend.

    Args:
        df: Training frame with ``unit``, ``cycle`` and ``columns``.
        columns: Candidate columns (operational settings and sensors).
        min_trend: Minimum :func:`trend_strength` to keep a column.

    Returns:
        ``(kept, report)``. ``kept`` lists the retained columns in input order.
        ``report`` has one row per candidate with ``n_unique``, ``std``,
        ``trend_strength``, ``kept`` and ``reason``.
    """
    const = set(constant_columns(df, columns))
    trend = trend_strength(df, columns)
    rows = []
    for c in columns:
        if c in const:
            keep, reason = False, "constant (zero variance)"
        elif trend[c] < min_trend:
            keep, reason = False, f"no consistent trend (< {min_trend})"
        else:
            keep, reason = True, "kept"
        rows.append((c, int(df[c].nunique()), float(df[c].std()), float(trend[c]), keep, reason))
    report = pd.DataFrame(
        rows, columns=["column", "n_unique", "std", "trend_strength", "kept", "reason"]
    ).set_index("column")
    return [c for c in columns if report.loc[c, "kept"]], report


def degradation_onset(
    df: pd.DataFrame,
    columns: Sequence[str],
    baseline: int = 30,
    k: float = 3.0,
    window: int = 10,
    persist: int = 10,
) -> pd.DataFrame:
    """Estimate, per engine, the cycle at which degradation becomes visible.

    Method:
      1. Each column is oriented so degradation is an increase (the sign is the
         median over engines of mean(last ``baseline`` cycles) - mean(first
         ``baseline`` cycles)).
      2. Per engine, each column is centred on that engine's own baseline mean
         (removes initial-wear offsets) and divided by the column's typical
         baseline noise (median over engines of the baseline std, which is
         robust to quantised sensors that are flat in one engine's baseline).
      3. The z-scores are averaged over columns into a health indicator and
         smoothed with a trailing-centred rolling mean of ``window`` cycles.
      4. Onset is the first cycle after the baseline from which the indicator
         stays above ``mean + k * std`` of its own baseline for ``persist``
         consecutive cycles.

    This is an *estimate* that depends on the settings, intended to show how
    much of an engine's life shows no sensor change, not an exact ground truth.

    Args:
        df: Training frame with ``unit``, ``cycle`` and ``columns``.
        columns: Informative columns (see :func:`select_columns`).
        baseline: Number of initial cycles treated as healthy.
        k: Detection threshold in baseline standard deviations.
        window: Rolling-mean smoothing length in cycles.
        persist: Consecutive cycles that must exceed the threshold.

    Returns:
        DataFrame indexed by ``unit`` with ``life`` (cycles to failure),
        ``onset_cycle`` (NaN if none detected) and ``rul_at_onset``
        (``life - onset_cycle``).

    Raises:
        ValueError: If ``baseline``, ``window`` or ``persist`` is not positive.
    """
    if min(baseline, window, persist) < 1:
        raise ValueError("baseline, window and persist must be positive")
    groups = {u: g.sort_values("cycle") for u, g in df.groupby("unit")}

    sign, noise = {}, {}
    for c in columns:
        shifts = [g[c].iloc[-baseline:].mean() - g[c].iloc[:baseline].mean() for g in groups.values()]
        sign[c] = 1.0 if np.median(shifts) >= 0 else -1.0
        stds = [g[c].iloc[:baseline].std() for g in groups.values()]
        noise[c] = float(np.nanmedian(stds)) or 1.0  # guard against an all-zero baseline

    rows = []
    for unit, g in groups.items():
        life = int(g["cycle"].max())
        onset = np.nan
        if len(g) >= baseline + persist:
            z = pd.concat(
                {c: sign[c] * (g[c] - g[c].iloc[:baseline].mean()) / noise[c] for c in columns},
                axis=1,
            )
            hi = z.mean(axis=1).rolling(window, min_periods=1).mean()
            base = hi.iloc[:baseline]
            above = (hi > base.mean() + k * base.std()).to_numpy()
            for t in range(baseline, len(g) - persist + 1):
                if above[t : t + persist].all():
                    onset = float(g["cycle"].iloc[t])
                    break
        rows.append((unit, life, onset, life - onset))
    return pd.DataFrame(rows, columns=["unit", "life", "onset_cycle", "rul_at_onset"]).set_index("unit")
