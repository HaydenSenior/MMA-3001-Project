"""Post-processing of RUL predictions along an engine's own history.

An engine's true RUL falls by exactly one per cycle. A prediction made ``j`` cycles ago,
``p_{t-j}``, therefore implies ``p_{t-j} - j`` for now. Averaging these implied values over
the last ``window`` cycles reduces the noise of any single prediction. It uses only predictions
at cycles ``<= t``, so it is causal.

Limitation: predictions clipped at the cap are flat, so propagating them forward can pull the
estimate too low while the true RUL is still above the cap. The result is re-clipped to ``[0, cap]``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def propagate_smooth(preds: pd.DataFrame, window: int, cap: Optional[float] = None) -> pd.Series:
    """Smooth predictions by averaging ``p_{t-j} - j`` for ``j = 0..window`` within each engine.

    Args:
        preds: Frame with ``unit``, ``cycle`` and ``y_pred``. For each engine the cycles must be
            consecutive (step 1); rows may be in any order.
        window: Number of earlier cycles to use, ``>= 0``. 0 returns the predictions unchanged.
            Near the first row of an engine fewer earlier predictions exist and fewer are used.
        cap: Upper clip for the result (lower clip is 0). ``None`` means no upper clip.

    Returns:
        Series aligned with ``preds.index`` holding the smoothed predictions.

    Raises:
        ValueError: On a negative or non-integer ``window``, missing columns, or non-consecutive cycles.
    """
    if int(window) != window or window < 0:
        raise ValueError("window must be a non-negative integer")
    missing = [c for c in ("unit", "cycle", "y_pred") if c not in preds.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    window = int(window)
    out = pd.Series(np.nan, index=preds.index, name="y_smooth")
    for unit, g in preds.groupby("unit"):
        g = g.sort_values("cycle")
        if len(g) > 1 and not (np.diff(g["cycle"].to_numpy()) == 1).all():
            raise ValueError(f"unit {unit}: cycles are not consecutive")
        p = g["y_pred"].to_numpy(dtype=np.float64)
        sm = np.empty_like(p)
        for t in range(len(p)):
            j = np.arange(0, min(window, t) + 1)
            sm[t] = np.mean(p[t - j] - j)
        out.loc[g.index] = sm
    return out.clip(lower=0.0, upper=cap)
