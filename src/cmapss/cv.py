"""Engine-grouped cross-validation and test-set prediction for RUL models.

Evaluation protocol (identical for every model so comparisons are fair)
----------------------------------------------------------------------
* Folds split by *engine* (``splits.unit_kfold_splits``), so no engine is in
  both train and validation.
* Models train on the capped target ``min(RUL, cap)`` (or ``RUL`` if
  ``cap`` is None), using every cycle of the training engines.
* Validation rows are those that resemble a test cut-off: ``cycle >=
  eval_min_cycle`` (the shortest test engine has 31 observed cycles) and true
  RUL within ``eval_rul_range`` (10 to 150 cycles, the range of the PHM'08 test
  set described by Saxena et al. 2008).
* Errors are measured against the **uncapped** true RUL, as in the official
  test file. Predictions are clipped to ``[0, cap]`` (RUL cannot be negative,
  and a model trained on a capped target cannot be asked for more than the cap).
* The asymmetric score is a sum over engines, so ``score_mean`` (the mean
  per-row term) is reported; ``100 * score_mean`` is the expected score on a
  100-engine test set under this evaluation.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .loader import add_rul
from .metrics import evaluate
from .splits import unit_kfold_splits


def last_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The final observed cycle of every engine, sorted by unit.

    For a test frame this is the single row a model must predict from.
    """
    return df.sort_values(["unit", "cycle"]).groupby("unit", as_index=False).tail(1).set_index("unit", drop=False)


def _target_name(cap: Optional[int]) -> str:
    return "RUL_capped" if cap else "RUL"


def cross_validate(
    df: pd.DataFrame,
    columns: Sequence[str],
    model_factory: Callable[[], object],
    cap: Optional[int] = 125,
    n_splits: int = 5,
    seed: int = 0,
    eval_rul_range: Tuple[int, int] = (10, 150),
    eval_min_cycle: int = 31,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Engine-grouped K-fold cross-validation of a RUL model on training data.

    Args:
        df: Raw training frame (``loader.load_raw``), without RUL columns.
        columns: Feature columns, e.g. the kept sensors.
        model_factory: Zero-argument callable returning a *fresh* model with
            ``fit(X, y)`` and ``predict(X)``.
        cap: RUL cap for the training target, or None for no cap.
        n_splits: Number of folds.
        seed: Seed for assigning engines to folds.
        eval_rul_range: Inclusive ``(low, high)`` true-RUL range of validation rows used for scoring.
        eval_min_cycle: Smallest cycle number of a scored validation row.

    Returns:
        ``(folds, predictions)``. ``folds`` has one row per fold with ``rmse``,
        ``mae``, ``bias``, ``score``, ``score_mean``, ``n`` (scored rows),
        ``n_late``, ``n_early``, ``fit_seconds`` and engine counts.
        ``predictions`` has one row per scored validation row: ``fold``,
        ``unit``, ``cycle``, ``y_true``, ``y_pred``.

    Raises:
        ValueError: If a fold has no rows to score.
    """
    columns = list(columns)
    lo, hi = eval_rul_range
    labelled = add_rul(df, cap=cap) if cap else add_rul(df)
    target = _target_name(cap)
    upper = cap if cap else np.inf

    fold_rows, pred_frames = [], []
    for k, (tr_idx, va_idx) in enumerate(unit_kfold_splits(df, n_splits=n_splits, seed=seed), start=1):
        train, val = labelled.iloc[tr_idx], labelled.iloc[va_idx]
        scored = val[(val["cycle"] >= eval_min_cycle) & (val["RUL"] >= lo) & (val["RUL"] <= hi)]
        if scored.empty:
            raise ValueError(f"fold {k}: no validation rows within the evaluation range")

        model = model_factory()
        t0 = time.perf_counter()
        model.fit(train[columns].to_numpy(), train[target].to_numpy())
        fit_seconds = time.perf_counter() - t0

        y_pred = np.clip(model.predict(scored[columns].to_numpy()), 0.0, upper)
        y_true = scored["RUL"].to_numpy()
        m = evaluate(y_true, y_pred)
        fold_rows.append(
            {"fold": k, "rmse": m["rmse"], "mae": m["mae"], "bias": m["bias"], "score": m["score"],
             "score_mean": m["score"] / m["n"], "n": m["n"], "n_late": m["n_late"], "n_early": m["n_early"],
             "fit_seconds": fit_seconds, "train_engines": int(train["unit"].nunique()),
             "val_engines": int(val["unit"].nunique())}
        )
        pred_frames.append(
            pd.DataFrame({"fold": k, "unit": scored["unit"].to_numpy(), "cycle": scored["cycle"].to_numpy(),
                          "y_true": y_true, "y_pred": y_pred})
        )
    return pd.DataFrame(fold_rows).set_index("fold"), pd.concat(pred_frames, ignore_index=True)


def fit_predict_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: Sequence[str],
    model_factory: Callable[[], object],
    cap: Optional[int] = 125,
) -> pd.Series:
    """Fit on all training engines, then predict RUL from each test engine's last cycle.

    This function never sees the true test RUL; scoring is done by the caller.
    Predictions are clipped to ``[0, cap]`` as in :func:`cross_validate`.

    Args:
        train_df: Raw training frame.
        test_df: Raw test frame.
        columns: Feature columns.
        model_factory: Zero-argument callable returning a fresh model.
        cap: RUL cap for the training target, or None.

    Returns:
        Series of predicted RUL indexed by test ``unit`` (ascending), named ``y_pred``.
    """
    columns = list(columns)
    labelled = add_rul(train_df, cap=cap) if cap else add_rul(train_df)
    model = model_factory().fit(labelled[columns].to_numpy(), labelled[_target_name(cap)].to_numpy())
    rows = last_rows(test_df)
    pred = np.clip(model.predict(rows[columns].to_numpy()), 0.0, cap if cap else np.inf)
    return pd.Series(pred, index=rows["unit"].to_numpy(), name="y_pred")
