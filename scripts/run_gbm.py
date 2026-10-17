"""Gradient boosting vs ridge on C-MAPSS FD001, with and without windowed history (a 2x2 comparison).

Question: is the error left after windowing caused by the model being linear, or by the
information in the sensors? Same engine-grouped folds and same scored rows as the other scripts.
The official test set is NOT touched unless ``--final-test`` is given (use once, when frozen).

    python scripts/run_gbm.py
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from cmapss.cv import cross_validate, fit_predict_test  # noqa: E402
from cmapss.features import select_columns  # noqa: E402
from cmapss.loader import SENSOR_COLUMNS, SETTING_COLUMNS, load_subset  # noqa: E402
from cmapss.metrics import evaluate, rmse  # noqa: E402
from cmapss.models import RidgeRegressor  # noqa: E402
from cmapss.windows import feature_names, window_features  # noqa: E402

ALL_COLS = SETTING_COLUMNS + SENSOR_COLUMNS
STATS = ("mean", "slope", "std")


def gbm_factory(iters, leaves, min_leaf, lr=0.05):
    """Boosted trees. Early stopping is OFF on purpose: scikit-learn's default ('auto') holds out a random
    10% of *rows* for stopping, and rows of the same engine are near-duplicates, so that check would leak."""
    return lambda: HistGradientBoostingRegressor(
        learning_rate=lr, max_iter=iters, max_leaf_nodes=leaves, min_samples_leaf=min_leaf,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--cap", type=int, default=125)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--final-test", action="store_true", help="score the official test set (use once)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    data = load_subset("FD001", a.data_dir)
    train = data.train
    kept, _ = select_columns(train, ALL_COLS, min_trend=0.3)
    feats = window_features(train, kept, a.window, STATS)
    wnames = feature_names(kept, STATS)
    key = ["fold", "unit", "cycle"]
    print(f"{len(kept)} sensors; windowed features: {len(wnames)} (N={a.window}); cap={a.cap}; {a.splits} folds\n")

    sets = {"snapshot": (train, kept), "windowed": (feats, wnames)}
    ridge_alpha = {"snapshot": 1.0, "windowed": 100.0}      # from run_baseline.py / run_windowed.py
    results = {}                                              # (model, featureset) -> (folds, preds)
    for fs, (frame, cols) in sets.items():
        results[("ridge", fs)] = cross_validate(frame, cols, lambda al=ridge_alpha[fs]: RidgeRegressor(al), a.cap, a.splits, a.seed)

    # Boosting: small grid, selected by mean CV RMSE separately for each feature set.
    grid_rows, best_cfg = [], {}
    for fs, (frame, cols) in sets.items():
        best = None
        for iters, leaves, min_leaf in itertools.product((100, 300), (8, 31), (20, 200)):
            folds, preds = cross_validate(frame, cols, gbm_factory(iters, leaves, min_leaf), a.cap, a.splits, a.seed)
            grid_rows.append({"features": fs, "iters": iters, "leaves": leaves, "min_leaf": min_leaf, "rmse": folds.rmse.mean(),
                              "rmse_sd": folds.rmse.std(ddof=1), "fit_s": folds.fit_seconds.mean()})
            if best is None or folds.rmse.mean() < best[0]:
                best = (folds.rmse.mean(), (iters, leaves, min_leaf), folds, preds)
        best_cfg[fs] = best[1]; results[("boosting", fs)] = (best[2], best[3])
    grid = pd.DataFrame(grid_rows); grid.to_csv(out / "gbm_grid.csv", index=False)
    print("Boosting grid (lr=0.05, l2=1), mean CV RMSE per configuration:")
    print(grid.round(3).to_string(index=False)); print()
    for fs in sets:
        print(f"chosen for {fs}: iters={best_cfg[fs][0]}, leaves={best_cfg[fs][1]}, min_samples_leaf={best_cfg[fs][2]}")

    # Identical folds and scored rows everywhere (so differences come from model/features only).
    ref = results[("ridge", "snapshot")][1][key].sort_values(key).reset_index(drop=True)
    for k, (_, p) in results.items():
        assert p[key].sort_values(key).reset_index(drop=True).equals(ref), f"scored rows differ for {k}"

    base = results[("ridge", "windowed")][0]            # reference for paired comparison: windowed ridge
    rows = []
    for (model, fs), (f, p) in results.items():
        d = f.rmse - base.rmse
        rows.append({"model": model, "features": fs, "rmse": f.rmse.mean(), "rmse_sd": f.rmse.std(ddof=1), "mae": f.mae.mean(),
                     "bias": f.bias.mean(), "score_mean": f.score_mean.mean(), "fit_s": f.fit_seconds.mean(),
                     "predict_s": f.predict_seconds.mean(), "vs_windowed_ridge": d.mean(), "folds_better": int((d < 0).sum())})
    comp = pd.DataFrame(rows).sort_values(["features", "model"], ascending=[False, True]).set_index(["features", "model"])
    print("\n2x2 COMPARISON (rmse etc. in cycles; fit_s / predict_s = seconds per fold; vs_windowed_ridge = paired RMSE difference):")
    print(comp.round(3).to_string()); comp.to_csv(out / "gbm_comparison.csv")

    f, p = results[("boosting", "windowed")]
    rf, rp = results[("ridge", "windowed")]
    print("\nPer-fold RMSE, ridge vs boosting (windowed):", rf.rmse.round(1).tolist(), "vs", f.rmse.round(1).tolist())

    def region(q, lo, hi):
        s = q[(q.y_true >= lo) & (q.y_true <= hi)]
        return (s.y_pred - s.y_true).mean(), (s.y_pred > s.y_true).mean()
    print("Bias and share of late predictions by true RUL (windowed features):")
    for name, q in (("ridge", rp), ("boosting", p)):
        b1, l1 = region(q, 0, 30); b2, l2 = region(q, 110, 1000)
        print(f"  {name:9s} true RUL <= 30: mean error {b1:+6.1f}, late {l1:4.0%}  |  true RUL >= 110: mean error {b2:+6.1f}, late {l2:4.0%}")

    floor = rmse(p.y_true, np.minimum(p.y_true, a.cap))
    print(f"\nFloor from the cap: an oracle that knows min(true RUL, {a.cap}) still has RMSE {floor:.1f} on these rows "
          f"(truth above the cap cannot be predicted by a model trained on a capped target).")
    print(f"Expected asymmetric score on a 100-engine test set under this evaluation: ridge ~{100 * rf.score_mean.mean():.0f}, boosting ~{100 * f.score_mean.mean():.0f}")
    f.to_csv(out / "gbm_cv_folds.csv"); p.to_csv(out / "gbm_cv_predictions.csv", index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.arange(10, 151, 10)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    for name, q, c in (("ridge (windowed)", rp, "C1"), ("boosting (windowed)", p, "C0")):
        g = q.groupby(pd.cut(q.y_true, bins, include_lowest=True), observed=True).agg(t=("y_true", "mean"), m=("y_pred", "mean"), s=("y_pred", "std"))
        ax.errorbar(g.t, g.m, yerr=g.s, fmt="o-", capsize=3, color=c, label=name, alpha=.9)
    ax.plot([0, 150], [0, 150], "k--", label="perfect"); ax.axhline(a.cap, color="C3", ls=":", label=f"cap = {a.cap}")
    ax.set(xlabel="true RUL (cycles)", ylabel="predicted RUL (cycles)", title="Windowed features: mean ± sd of prediction (CV)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "gbm_pred_vs_true.png", dpi=100)

    if a.final_test:
        it, lv, ml = best_cfg["windowed"]
        fte = window_features(data.test, kept, a.window, STATS)
        pred = fit_predict_test(feats, fte, wnames, gbm_factory(it, lv, ml), a.cap)
        res = evaluate(data.test_rul.loc[pred.index].to_numpy(), pred.to_numpy())
        print("\nTEST SET (scored once):", {k: round(v, 3) for k, v in res.items()})
        (out / "gbm_test.json").write_text(json.dumps({"config": best_cfg["windowed"], "window": a.window, "cap": a.cap, **res}, indent=1))
    else:
        print("\nTest set not scored (pass --final-test once the model is frozen).")


if __name__ == "__main__":
    main()
