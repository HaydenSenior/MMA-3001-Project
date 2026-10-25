"""How well does the model generalise? Training error vs validation error, and a learning curve.

Two questions the rest of the project does not answer:

  A  Overfitting. Every result elsewhere is a validation score. Comparing it with the error on the
     engines the model was FITTED on shows how much the model memorises rather than learns. The
     gap is the diagnostic; a large gap with good validation error is tolerable, a large gap with
     poor validation error is not.

  B  Would more engines help? Re-running the same cross-validation while training on a growing
     number of engines shows whether validation error has flattened. If it has, collecting more
     engines of the same kind would not help and the limit lies elsewhere (sensor information,
     the RUL cap). If it is still falling, data is the binding constraint.

Both parts use TRAINING data only; the held-out test set is never loaded. Training and validation
rows are filtered identically (the same cycle and true-RUL range used throughout the project), so
the two errors are measured on comparable rows and the gap is not an artefact of row selection.

    python scripts/run_generalisation.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss.features import select_columns  # noqa: E402
from cmapss.loader import SENSOR_COLUMNS, SETTING_COLUMNS, add_rul, load_subset  # noqa: E402
from cmapss.metrics import evaluate  # noqa: E402
from cmapss.models import MeanBaseline, RidgeRegressor, boosting_factory  # noqa: E402
from cmapss.splits import unit_kfold_splits  # noqa: E402
from cmapss.windows import feature_names, window_features  # noqa: E402

ALL_COLS = SETTING_COLUMNS + SENSOR_COLUMNS
STATS = ("mean", "slope", "std")
CFG = dict(iters=200, leaves=16, min_leaf=100)


def scored(frame: pd.DataFrame, lo: int, hi: int, min_cycle: int) -> pd.DataFrame:
    """Rows that resemble a test cut-off, the same filter used everywhere else in the project."""
    return frame[(frame.cycle >= min_cycle) & frame.RUL.between(lo, hi)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--cap", type=int, default=125)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rul-range", type=int, nargs=2, default=[10, 150])
    ap.add_argument("--min-cycle", type=int, default=31)
    ap.add_argument("--engine-counts", type=int, nargs="+", default=[10, 20, 40, 60, 80])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    lo, hi = a.rul_range
    t0 = time.perf_counter()

    train = load_subset("FD001", a.data_dir).train
    kept, _ = select_columns(train, ALL_COLS, min_trend=0.3)
    wnames = feature_names(kept, STATS)
    wfeat = add_rul(window_features(train, kept, a.window, STATS, pad="edge"), cap=a.cap)
    snap = add_rul(train, cap=a.cap)

    models = {
        "mean predictor": (snap, kept, MeanBaseline),
        "baseline: ridge, single cycle": (snap, kept, lambda: RidgeRegressor(1.0)),
        "ridge, windowed": (wfeat, wnames, lambda: RidgeRegressor(100.0)),
        "final: boosting, windowed": (wfeat, wnames, boosting_factory(**CFG)),
    }

    # ---------------- A. training error vs validation error ----------------
    rows = []
    for label, (frame, cols, factory) in models.items():
        for seed in a.seeds:
            for k, (tr_idx, va_idx) in enumerate(unit_kfold_splits(frame, a.splits, seed=seed), start=1):
                tr, va = frame.iloc[tr_idx], frame.iloc[va_idx]
                model = factory().fit(tr[cols].to_numpy(), tr["RUL_capped"].to_numpy())
                for split, part in (("train", scored(tr, lo, hi, a.min_cycle)), ("validation", scored(va, lo, hi, a.min_cycle))):
                    pred = np.clip(model.predict(part[cols].to_numpy()), 0.0, a.cap)
                    m = evaluate(part["RUL"].to_numpy(), pred)
                    rows.append({"model": label, "split": split, "seed": seed, "fold": k,
                                 "rmse": m["rmse"], "mae": m["mae"], "bias": m["bias"],
                                 "score_mean": m["score"] / m["n"], "rows": m["n"]})
        print(f"  A: {label} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "generalisation_raw.csv", index=False)

    piv = raw.groupby(["model", "split"]).rmse.mean().unstack()
    piv["gap"] = piv["validation"] - piv["train"]
    piv["gap_%_of_train"] = 100 * piv["gap"] / piv["train"]
    piv["val_sd_across_folds"] = raw[raw.split == "validation"].groupby("model").rmse.std(ddof=1)
    piv = piv.reindex(list(models))
    piv.to_csv(out / "generalisation_gap.csv")
    print("\nA. TRAINING vs VALIDATION RMSE (cycles; same row filter on both sides)")
    print(piv.round(2).to_string())
    print("\n   'gap' is how much worse the model does on engines it has never seen.")
    print("   A model with no capacity to memorise (the mean predictor) should show almost none.")

    # ---------------- B. learning curve ----------------
    frame, cols, factory = models["final: boosting, windowed"]
    curve = []
    for n_engines in [*a.engine_counts, None]:
        for seed in a.seeds:
            rng = np.random.default_rng(seed)
            for k, (tr_idx, va_idx) in enumerate(unit_kfold_splits(frame, a.splits, seed=seed), start=1):
                tr, va = frame.iloc[tr_idx], frame.iloc[va_idx]
                units = np.unique(tr.unit)
                if n_engines is not None and n_engines < len(units):
                    tr = tr[tr.unit.isin(rng.choice(units, n_engines, replace=False))]
                model = factory().fit(tr[cols].to_numpy(), tr["RUL_capped"].to_numpy())
                part = scored(va, lo, hi, a.min_cycle)
                pred = np.clip(model.predict(part[cols].to_numpy()), 0.0, a.cap)
                curve.append({"train_engines": n_engines or int(tr.unit.nunique()), "seed": seed, "fold": k,
                              "rmse": evaluate(part["RUL"].to_numpy(), pred)["rmse"]})
        print(f"  B: {n_engines or 'all'} engines done ({time.perf_counter() - t0:.0f}s)", flush=True)
    cv = pd.DataFrame(curve)
    lc = cv.groupby("train_engines").rmse.agg(["mean", "std"]).rename(columns={"mean": "val_rmse", "std": "sd_across_folds"})
    lc["improvement_vs_previous"] = -lc.val_rmse.diff()
    lc.to_csv(out / "generalisation_learning_curve.csv")
    print("\nB. LEARNING CURVE for the final model (validation RMSE vs number of training engines)")
    print(lc.round(2).to_string())

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(piv))
    ax[0].bar(x - 0.2, piv["train"], 0.4, label="training engines", color="C0")
    ax[0].bar(x + 0.2, piv["validation"], 0.4, label="unseen engines", color="C1")
    ax[0].set_xticks(x); ax[0].set_xticklabels([m.split(":")[-1].strip() for m in piv.index], rotation=20, ha="right", fontsize=8)
    ax[0].set(ylabel="RMSE (cycles)", title="Training vs validation error")
    ax[1].errorbar(lc.index, lc.val_rmse, yerr=lc.sd_across_folds, fmt="o-", capsize=3)
    ax[1].axhline(lc.val_rmse.iloc[-1], color="grey", ls=":", lw=1, label="all 80 training engines")
    ax[1].set(xlabel="training engines used", ylabel="validation RMSE (cycles)", title="Learning curve (final model)")
    for axis in ax:
        axis.grid(alpha=.3); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "generalisation.png", dpi=100)

    (out / "generalisation_summary.json").write_text(json.dumps({
        "window": a.window, "cap": a.cap, "seeds": a.seeds, "splits": a.splits,
        "gap": {m: {"train_rmse": round(float(r["train"]), 2), "validation_rmse": round(float(r["validation"]), 2),
                    "gap": round(float(r["gap"]), 2)} for m, r in piv.iterrows()},
        "learning_curve": {int(i): round(float(v), 2) for i, v in lc.val_rmse.items()},
        "note": "training data only; test set not loaded"}, indent=1))
    print(f"\nTotal time {time.perf_counter() - t0:.0f}s. Test set not used.")


if __name__ == "__main__":
    main()
