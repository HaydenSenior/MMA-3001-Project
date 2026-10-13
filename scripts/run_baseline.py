"""Baseline for RUL prediction on C-MAPSS FD001: ridge regression on single-cycle sensor values.

Prints and saves cross-validation results. The official test set is NOT touched
unless ``--final-test`` is given; run that once, after all model choices are frozen.

    python scripts/run_baseline.py                      # cross-validation only
    python scripts/run_baseline.py --final-test         # also score the test set (once!)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss.cv import cross_validate, fit_predict_test  # noqa: E402
from cmapss.features import select_columns  # noqa: E402
from cmapss.loader import SENSOR_COLUMNS, SETTING_COLUMNS, load_subset  # noqa: E402
from cmapss.metrics import evaluate  # noqa: E402
from cmapss.models import MeanBaseline, RidgeRegressor  # noqa: E402
from cmapss.splits import unit_kfold_splits  # noqa: E402

ALL_COLS = SETTING_COLUMNS + SENSOR_COLUMNS


def summarise(folds: pd.DataFrame) -> pd.Series:
    """Mean and std of each metric across folds."""
    out = {}
    for m in ("rmse", "mae", "bias", "score_mean", "fit_seconds"):
        out[f"{m}_mean"], out[f"{m}_std"] = folds[m].mean(), folds[m].std(ddof=1)
    return pd.Series(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--cap", type=int, default=125)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0, 0.01, 0.1, 1, 10, 100, 1000])
    ap.add_argument("--final-test", action="store_true", help="score the official test set (use once)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    data = load_subset("FD001", a.data_dir)
    train = data.train
    kept, _ = select_columns(train, ALL_COLS, min_trend=0.3)
    print(f"Features ({len(kept)}): {kept}")

    # Screening used only sensor-vs-cycle trends (no RUL labels, no test data), but check it is stable
    # when recomputed on each fold's training engines only.
    stable = all(
        select_columns(train.iloc[tr], ALL_COLS, 0.3)[0] == kept
        for tr, _ in unit_kfold_splits(train, a.splits, seed=a.seed)
    )
    print(f"Column selection identical when recomputed inside every training fold: {stable}\n")

    # 1. Floor: predict the mean training target.
    f_mean, _ = cross_validate(train, kept, MeanBaseline, a.cap, a.splits, a.seed)
    # 2. Ridge over a grid of penalties.
    rows, results = [], {}
    for alpha in a.alphas:
        folds, preds = cross_validate(train, kept, lambda al=alpha: RidgeRegressor(al), a.cap, a.splits, a.seed)
        results[alpha] = (folds, preds)
        rows.append({"model": f"ridge alpha={alpha:g}", **summarise(folds)})
    table = pd.DataFrame([{"model": "mean predictor", **summarise(f_mean)}] + rows).set_index("model")
    print(f"Cross-validation ({a.splits} folds of engines, cap={a.cap}), mean over folds:")
    print(table[["rmse_mean", "rmse_std", "mae_mean", "bias_mean", "score_mean_mean", "fit_seconds_mean"]].round(3).to_string())
    table.to_csv(out / "baseline_cv_summary.csv")

    best = min(results, key=lambda al: results[al][0]["rmse"].mean())
    folds, preds = results[best]
    print(f"\nBest alpha by mean CV RMSE: {best:g}  (differences between alphas are small; see std across folds)")
    print("\nPer-fold results for the chosen baseline:")
    print(folds[["rmse", "mae", "bias", "score_mean", "n", "n_late", "n_early", "fit_seconds"]].round(3).to_string())
    print(f"\nExpected asymmetric score on a 100-engine test set under this evaluation: ~{100 * folds['score_mean'].mean():.0f}")
    folds.to_csv(out / "baseline_cv_folds.csv"); preds.to_csv(out / "baseline_cv_predictions.csv", index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.arange(10, 151, 10)
    preds["bin"] = pd.cut(preds.y_true, bins, include_lowest=True)
    g = preds.groupby("bin", observed=True).agg(true=("y_true", "mean"), pred=("y_pred", "mean"), sd=("y_pred", "std"))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.errorbar(g.true, g.pred, yerr=g.sd, fmt="o-", capsize=3, label="ridge: mean ± sd of prediction")
    ax.plot([0, 150], [0, 150], "k--", label="perfect"); ax.axhline(a.cap, color="C3", ls=":", label=f"cap = {a.cap}")
    ax.set(xlabel="true RUL (cycles)", ylabel="predicted RUL (cycles)", title="Baseline, cross-validation"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "baseline_cv_pred_vs_true.png", dpi=100)

    if a.final_test:
        pred = fit_predict_test(train, data.test, kept, lambda: RidgeRegressor(best), a.cap)
        res = evaluate(data.test_rul.loc[pred.index].to_numpy(), pred.to_numpy())
        print("\nTEST SET (scored once):", {k: round(v, 3) for k, v in res.items()})
        (out / "baseline_test.json").write_text(json.dumps({"alpha": best, "cap": a.cap, **res}, indent=1))
    else:
        print("\nTest set not scored (pass --final-test once the model is frozen).")


if __name__ == "__main__":
    main()
