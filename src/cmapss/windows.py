"""Trailing-window features: summarise each engine's recent history at every cycle.

For an engine observed up to cycle ``t``, the features at ``t`` describe the
last ``window`` cycles ``t-window+1 .. t`` of each selected column:

* ``last``:  the most recent value (the single-snapshot baseline feature),
* ``mean``:  the average level (averages out sensor noise),
* ``slope``: least-squares slope against cycle, in sensor units per cycle (the trend),
* ``std``:   population standard deviation (the spread; mostly noise level here).

Windows are **causal**: features at cycle ``t`` never use cycles after ``t``,
and never cross from one engine into another.

Two ways of handling the start of an engine (``pad``):

* ``pad="none"`` (default): only cycles with a full window (``cycle >= window``)
  produce a row, so the first ``window - 1`` cycles of every engine have no
  features. An engine shorter than the window raises an error by default, so
  engines are never dropped silently (the shortest C-MAPSS FD001 test engine
  has 31 cycles, so ``window <= 31`` is safe on every test engine).
* ``pad="edge"``: the engine's first row is repeated ``window - 1`` times in
  front of it, so every cycle gets a full window and engines of any length work.
  Rows for ``cycle >= window`` are identical to the unpadded ones. Early rows
  describe a history that is partly invented (a flat start), so they understate
  trends; this is the documented rule for windows longer than the shortest test engine.

Example:
    >>> feats = window_features(train, ["s2", "s3"], window=30)   # doctest: +SKIP
    >>> names = feature_names(["s2", "s3"], ("mean", "slope", "std"))  # doctest: +SKIP
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

STATS = ("last", "mean", "slope", "std")


def feature_names(columns: Sequence[str], stats: Sequence[str]) -> List[str]:
    """Names of the feature columns, ordered by input column then statistic (``s2_mean``, ``s2_slope``, ...)."""
    return [f"{c}_{s}" for c in columns for s in stats]


def window_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    window: int = 30,
    stats: Sequence[str] = ("mean", "slope", "std"),
    drop_short: bool = False,
    pad: str = "none",
) -> pd.DataFrame:
    """Trailing-window statistics of ``columns`` for every engine and cycle.

    Args:
        df: Frame with ``unit``, ``cycle`` and ``columns``; rows of each engine in cycle order.
        columns: Sensor columns to summarise.
        window: Window length in cycles (``>= 1``; ``>= 2`` if ``"slope"`` is requested).
        stats: Subset of ``("last", "mean", "slope", "std")``.
        drop_short: If True, engines with fewer than ``window`` cycles are omitted
            instead of raising an error (only used when ``pad="none"``).
        pad: ``"none"`` or ``"edge"`` (see module docstring).

    Returns:
        Frame with ``unit``, ``cycle`` and one column per feature (see
        :func:`feature_names`), ordered by unit then cycle. With ``pad="none"``
        one row per engine cycle ``>= window``; with ``pad="edge"`` one row per
        engine cycle.

    Raises:
        ValueError: On an unknown statistic or ``pad`` value, a missing column,
            a bad ``window``, non-finite values, or (with ``pad="none"``) an
            engine shorter than ``window`` (unless ``drop_short``).
    """
    columns, stats = list(columns), list(stats)
    unknown = [s for s in stats if s not in STATS]
    if unknown or not stats:
        raise ValueError(f"stats must be a non-empty subset of {STATS}, got {stats}")
    missing = [c for c in ["unit", "cycle", *columns] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if pad not in ("none", "edge"):
        raise ValueError(f"pad must be 'none' or 'edge', got {pad!r}")
    if int(window) != window or window < 1:
        raise ValueError(f"window must be a positive integer, got {window!r}")
    if "slope" in stats and window < 2:
        raise ValueError("a slope needs window >= 2")
    if not np.isfinite(df[columns].to_numpy()).all():
        raise ValueError("columns contain NaN or infinite values")

    window = int(window)
    xc = np.arange(window, dtype=np.float64) - (window - 1) / 2.0     # centred time index
    sxx = float(xc @ xc) if window > 1 else 1.0

    frames = []
    for unit, g in df.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        n = len(g)
        values = g[columns].to_numpy(dtype=np.float64)
        if pad == "edge":
            values = np.vstack([np.repeat(values[:1], window - 1, axis=0), values])   # n + window - 1 rows
        elif n < window:
            if drop_short:
                continue
            raise ValueError(f"unit {unit} has {n} cycles, fewer than window={window}")
        # (rows, p, window): one trailing window per output row
        win = sliding_window_view(values, window, axis=0)
        per_stat = {
            "last": win[..., -1],
            "mean": win.mean(axis=2),
            "slope": (win * xc).sum(axis=2) / sxx,
            "std": win.std(axis=2),
        }
        out = np.stack([per_stat[s] for s in stats], axis=2).reshape(win.shape[0], -1)   # column-major by (col, stat)
        frame = pd.DataFrame(out, columns=feature_names(columns, stats))
        cycles = g["cycle"].to_numpy() if pad == "edge" else g["cycle"].to_numpy()[window - 1:]
        frame.insert(0, "cycle", cycles)
        frame.insert(0, "unit", unit)
        frames.append(frame)

    if not frames:
        raise ValueError("no engine has enough cycles for this window")
    result = pd.concat(frames, ignore_index=True)
    result["unit"] = result["unit"].astype(np.int64)
    result["cycle"] = result["cycle"].astype(np.int64)
    return result
