"""Windowed-feature ridge regression on C-MAPSS FD001, compared with the single-snapshot baseline.

Question: does giving the same ridge model each engine's recent history (mean, slope,
spread of every kept sensor over the last N cycles) reduce the baseline's error?

Everything uses the same engine-grouped folds and the same scored validation rows as
``run_baseline.py``, so differences come from the features alone. The official test set
is NOT touched unless ``--final-test`` is given; run that once, after all choices are frozen.

    python scripts/run_windowed.py
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
from cmapss.models import RidgeRegressor  # noqa: E402
from cmapss.windows import feature_names, window_features  # noqa: E402

ALL_COLS = SETTING_COLUMNS + SENSOR_COLUMNS
HEADLINE_STATS = ("mean", "slope", "std")      # fixed in advance, not chosen from the tables below


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--cap", type=int, default=125)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=30, help="window for the headline model (<= 31)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0, 0.1, 1, 10, 100, 1000])
    ap.add_argument("--final-test", action="store_true", help="score the official test set (use once)")
    a = ap.parse_args()
    if a.window > 31:
        sys.exit("window must be <= 31 so every scored validation row (cycle >= 31) and every test engine has a full window")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    data = load_subset("FD001", a.data_dir)
    train = data.train
    kept, _ = select_columns(train, ALL_COLS, min_trend=0.3)
    print(f"{len(kept)} sensors kept; window N={a.window}; cap={a.cap}; {a.splits} engine-grouped folds\n")

    # Reference: single-snapshot ridge (alpha=1; alpha is irrelevant for it, see baseline results).
    base_folds, base_preds = cross_validate(train, kept, lambda: RidgeRegressor(1.0), a.cap, a.splits, a.seed)
    key = ["fold", "unit", "cycle"]
    base_keys = base_preds[key].sort_values(key).reset_index(drop=True)

    def run(stats, window, alpha=1.0):
        feats = window_features(train, kept, window, stats)
        names = feature_names(kept, stats)
        folds, preds = cross_validate(feats, names, lambda: RidgeRegressor(alpha), a.cap, a.splits, a.seed)
        # Fair comparison: identical folds and identical scored rows as the baseline.
        assert preds[key].sort_values(key).reset_index(drop=True).equals(base_keys), "scored rows differ from baseline"
        return folds, preds, len(names)

    def row(label, folds, n_feat):
        diff = folds["rmse"] - base_folds["rmse"]
        return {"config": label, "features": n_feat, "rmse": folds.rmse.mean(), "rmse_sd": folds.rmse.std(ddof=1),
                "mae": folds.mae.mean(), "bias": folds.bias.mean(), "score_mean": folds.score_mean.mean(),
                "rmse_vs_baseline": diff.mean(), "folds_better": int((diff < 0).sum())}

    base_row = {"config": "baseline: single snapshot", "features": len(kept), "rmse": base_folds.rmse.mean(),
                "rmse_sd": base_folds.rmse.std(ddof=1), "mae": base_folds.mae.mean(), "bias": base_folds.bias.mean(),
                "score_mean": base_folds.score_mean.mean(), "rmse_vs_baseline": 0.0, "folds_better": 0}
    show = ["features", "rmse", "rmse_sd", "mae", "bias", "score_mean", "rmse_vs_baseline", "folds_better"]

    # 1. Headline model: pre-specified features, alpha chosen by CV.
    alpha_rows, runs = [], {}
    for alpha in a.alphas:
        folds, preds, nf = run(HEADLINE_STATS, a.window, alpha)
        runs[alpha] = (folds, preds, nf); alpha_rows.append(row(f"alpha={alpha:g}", folds, nf))
    alpha_tbl = pd.DataFrame(alpha_rows).set_index("config")
    best = min(runs, key=lambda al: runs[al][0]["rmse"].mean())
    folds, preds, nf = runs[best]
    print(f"HEADLINE: ridge on {'/'.join(HEADLINE_STATS)} of {len(kept)} sensors over the last {a.window} cycles")
    print("Penalty strength (alpha) vs CV error:")
    print(alpha_tbl[show].round(3).to_string())
    print(f"-> alpha = {best:g}\n")
    alpha_tbl.to_csv(out / "windowed_alpha.csv")

    # 2. Ablation: which kind of history helps? (exploratory, same window; not used to pick the headline)
    configs = [("last only (control: same rows, no history)", ("last",)), ("mean only", ("mean",)),
               ("mean + slope", ("mean", "slope")), ("mean + slope + std (headline)", HEADLINE_STATS),
               ("last + mean + slope + std", ("last", "mean", "slope", "std"))]
    abl = [base_row]
    for label, stats in configs:
        f, _, n = run(stats, a.window, best); abl.append(row(f"{label}", f, n))
    abl_tbl = pd.DataFrame(abl).set_index("config")
    print(f"ABLATION (N={a.window}, alpha={best:g}): which features carry the improvement")
    print(abl_tbl[show].round(3).to_string()); abl_tbl.to_csv(out / "windowed_ablation.csv")

    # 3. Window length (exploratory; the proper sweep with tuning is a later step).
    sweep = [base_row]
    for n_win in (5, 10, 20, 30):
        f, _, n = run(HEADLINE_STATS, n_win, best); sweep.append(row(f"N={n_win}", f, n))
    sweep_tbl = pd.DataFrame(sweep).set_index("config")
    print(f"\nWINDOW LENGTH (mean + slope + std, alpha={best:g})")
    print(sweep_tbl[show].round(3).to_string()); sweep_tbl.to_csv(out / "windowed_window_sweep.csv")

    # 4. Where did the baseline fail? Compare errors at low and high true RUL.
    def region(p, lo, hi):
        s = p[(p.y_true >= lo) & (p.y_true <= hi)]
        return (s.y_pred - s.y_true).mean(), (s.y_pred > s.y_true).mean()
    print("\nBias and share of late predictions by true RUL:")
    for name, p in (("baseline", base_preds), ("windowed", preds)):
        b30, l30 = region(p, 0, 30); b110, l110 = region(p, 110, 1000)
        print(f"  {name:9s} true RUL <= 30: mean error {b30:+6.1f}, late {l30:4.0%}  |  true RUL >= 110: mean error {b110:+6.1f}, late {l110:4.0%}")

    print(f"\nPer-fold RMSE, baseline vs windowed: {base_folds.rmse.round(1).tolist()} vs {folds.rmse.round(1).tolist()}")
    print(f"Expected asymmetric score on a 100-engine test set under this evaluation: baseline ~{100 * base_folds.score_mean.mean():.0f}, windowed ~{100 * folds.score_mean.mean():.0f}")
    folds.to_csv(out / "windowed_cv_folds.csv"); preds.to_csv(out / "windowed_cv_predictions.csv", index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.arange(10, 151, 10)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    for name, p, c in (("baseline (single snapshot)", base_preds, "C1"), (f"windowed (N={a.window})", preds, "C0")):
        g = p.groupby(pd.cut(p.y_true, bins, include_lowest=True), observed=True).agg(t=("y_true", "mean"), m=("y_pred", "mean"), s=("y_pred", "std"))
        ax.errorbar(g.t, g.m, yerr=g.s, fmt="o-", capsize=3, color=c, label=name, alpha=.9)
    ax.plot([0, 150], [0, 150], "k--", label="perfect"); ax.axhline(a.cap, color="C3", ls=":", label=f"cap = {a.cap}")
    ax.set(xlabel="true RUL (cycles)", ylabel="predicted RUL (cycles)", title="Ridge: mean ± sd of prediction (cross-validation)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "windowed_pred_vs_true.png", dpi=100)

    if a.final_test:
        names = feature_names(kept, HEADLINE_STATS)
        ftr = window_features(train, kept, a.window, HEADLINE_STATS)
        fte = window_features(data.test, kept, a.window, HEADLINE_STATS)
        pred = fit_predict_test(ftr, fte, names, lambda: RidgeRegressor(best), a.cap)
        res = evaluate(data.test_rul.loc[pred.index].to_numpy(), pred.to_numpy())
        print("\nTEST SET (scored once):", {k: round(v, 3) for k, v in res.items()})
        (out / "windowed_test.json").write_text(json.dumps({"alpha": best, "window": a.window, "cap": a.cap, **res}, indent=1))
    else:
        print("\nTest set not scored (pass --final-test once the model is frozen).")


if __name__ == "__main__":
    main()
