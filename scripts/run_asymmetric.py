"""Targeting the asymmetric maintenance objective instead of RMSE (C-MAPSS FD001).

A late RUL prediction risks an in-service failure; an early one wastes remaining life.
RMSE treats them alike, so the prediction that minimises RMSE is not the one a maintenance
planner should act on. This script measures what it costs, in accuracy, to predict earlier
on purpose, using two ways of doing it:

  A  constant shift   subtract delta cycles from every prediction (no retraining)
  B  quantile loss    train the model to predict a low quantile instead of the mean

Both are evaluated on the same engine-grouped folds and scored rows as every other script.

  C  nested cross-validation: delta and the quantile are chosen on INNER folds of the
     training engines only, then applied to the held-out engines, so the reported numbers
     are not tuned on the rows they are measured on.
  D  cost model: cost = c_early * max(0, -d) + c_late * max(0, d) per prediction, and the
     shift that minimises it for several cost ratios. Theory says the cost-minimising
     prediction is the quantile c_early / (c_early + c_late); Part D checks that empirically.

The official test set is never used.

    python scripts/run_asymmetric.py
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

from cmapss.cv import cross_validate  # noqa: E402
from cmapss.decision import apply_shift, best_shift, expected_cost, optimal_quantile  # noqa: E402
from cmapss.features import select_columns  # noqa: E402
from cmapss.loader import SENSOR_COLUMNS, SETTING_COLUMNS, load_subset  # noqa: E402
from cmapss.metrics import evaluate, nasa_score, rmse  # noqa: E402
from cmapss.models import boosting_factory  # noqa: E402
from cmapss.splits import unit_kfold_splits  # noqa: E402
from cmapss.windows import feature_names, window_features  # noqa: E402

ALL_COLS = SETTING_COLUMNS + SENSOR_COLUMNS
STATS = ("mean", "slope", "std")
CFG = dict(iters=200, leaves=16, min_leaf=100)


def make_plot(tblA: pd.DataFrame, tblB: pd.DataFrame, out: Path) -> None:
    """Three panels: RMSE vs the asymmetric score, RMSE vs maintenance cost, and how often the model is late."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for axis, col, title in ((ax[0], "score_mean", "RMSE vs the PHM'08 score"),
                             (ax[1], "cost", "RMSE vs maintenance cost")):
        axis.plot(tblA.rmse, tblA[col], "o-", color="C1", label="constant shift")
        for d in tblA.index:
            if d in (0, 5, 10, 20, 30):
                axis.annotate(f"{int(d)}", (tblA.rmse[d], tblA[col][d]), textcoords="offset points", xytext=(5, 4), fontsize=8, color="C1")
        axis.plot(tblB.rmse, tblB[col], "s-", color="C0", label="quantile loss")
        for q in tblB.index:
            axis.annotate(f"{q:g}", (tblB.rmse[q], tblB[col][q]), textcoords="offset points", xytext=(5, -11), fontsize=8, color="C0")
        axis.set(xlabel="CV RMSE (cycles) - lower is better", title=title)
    ax[0].set_ylabel("asymmetric score per row - lower is better")
    ax[1].set_ylabel("cost per prediction - lower is better")
    ax[2].plot(tblA.index, tblA.pct_late, "o-", color="C1", label="% late")
    ax[2].plot(tblA.index, tblA.pct_late_gt10, "o--", color="C3", label="% late by more than 10 cycles")
    ax[2].set(xlabel="shift applied (cycles)", ylabel="% of predictions", title="How often the model is late")
    for x in ax:
        x.grid(alpha=.3); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "asym_tradeoff.png", dpi=100)


def row_metrics(y_true, y_pred, c_early, c_late):
    """Metrics for one set of predictions, including the operational ones."""
    m = evaluate(y_true, y_pred)
    d = np.asarray(y_pred) - np.asarray(y_true)
    return {"rmse": m["rmse"], "mae": m["mae"], "bias": m["bias"], "score_mean": m["score"] / m["n"],
            "pct_late": 100 * np.mean(d > 0), "pct_late_gt10": 100 * np.mean(d > 10),
            "cost": expected_cost(y_true, y_pred, c_early, c_late), "n": m["n"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--cap", type=int, default=125)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--deltas", type=float, nargs="+", default=[0, 2, 5, 8, 10, 12, 15, 20, 25, 30])
    ap.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--c-early", type=float, default=1.0)
    ap.add_argument("--c-late", type=float, default=10.0)
    ap.add_argument("--plot-only", action="store_true", help="rebuild the figure from saved CSVs without re-running")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    t0 = time.perf_counter()

    if a.plot_only:
        make_plot(pd.read_csv(out / "asym_shift.csv", index_col="delta"),
                  pd.read_csv(out / "asym_quantile.csv", index_col="quantile"), out)
        print(f"figure rebuilt from {out}/asym_shift.csv and asym_quantile.csv")
        return

    train = load_subset("FD001", a.data_dir).train
    kept, _ = select_columns(train, ALL_COLS, min_trend=0.3)
    names = feature_names(kept, STATS)
    feats = window_features(train, kept, a.window, STATS, pad="edge")
    n_folds = len(a.seeds) * a.splits
    print(f"window {a.window}, cap {a.cap}, boosting {CFG}; {len(a.seeds)} seeds x {a.splits} folds = {n_folds} folds")
    print(f"cost model: early {a.c_early} per cycle, late {a.c_late} per cycle "
          f"-> theory says predict the {optimal_quantile(a.c_early, a.c_late):.3f} quantile\n")

    # ---------------- A: constant shift ----------------
    rowsA, base_preds = [], []
    for seed in a.seeds:
        _, preds = cross_validate(feats, names, boosting_factory(**CFG), a.cap, a.splits, seed)
        preds["seed"] = seed
        base_preds.append(preds)
        for fold, g in preds.groupby("fold"):
            for delta in a.deltas:
                shifted = apply_shift(g.y_pred.to_numpy(), delta, a.cap)
                rowsA.append({"delta": delta, "seed": seed, "fold": fold,
                              **row_metrics(g.y_true.to_numpy(), shifted, a.c_early, a.c_late)})
        print(f"  A: seed {seed} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    base_preds = pd.concat(base_preds, ignore_index=True)
    dA = pd.DataFrame(rowsA)
    tblA = dA.groupby("delta").mean(numeric_only=True).drop(columns=["seed", "fold"])
    tblA.to_csv(out / "asym_shift.csv")
    print("\nPART A: constant shift applied to the RMSE-trained model")
    print(tblA[["rmse", "bias", "score_mean", "pct_late", "pct_late_gt10", "cost"]].round(2).to_string())

    # ---------------- B: quantile loss ----------------
    rowsB = []
    for q in a.quantiles:
        for seed in a.seeds:
            _, preds = cross_validate(feats, names, boosting_factory(**CFG, loss="quantile", quantile=q), a.cap, a.splits, seed)
            for fold, g in preds.groupby("fold"):
                rowsB.append({"quantile": q, "seed": seed, "fold": fold,
                              **row_metrics(g.y_true.to_numpy(), g.y_pred.to_numpy(), a.c_early, a.c_late)})
        print(f"  B: quantile {q} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    dB = pd.DataFrame(rowsB)
    tblB = dB.groupby("quantile").mean(numeric_only=True).drop(columns=["seed", "fold"])
    tblB.to_csv(out / "asym_quantile.csv")
    print("\nPART B: model trained with a quantile loss (no shift)")
    print(tblB[["rmse", "bias", "score_mean", "pct_late", "pct_late_gt10", "cost"]].round(2).to_string())

    # ---------------- C: nested cross-validation ----------------
    # Strategies that need a choice (delta, or the quantile) make it on inner folds of the
    # training engines only, so the outer fold is untouched by tuning.
    rowsC = []
    for seed in a.seeds:
        for k, (tr_idx, va_idx) in enumerate(unit_kfold_splits(feats, a.splits, seed=seed), start=1):
            tr, va = feats.iloc[tr_idx], feats.iloc[va_idx]

            inner = []
            for i_tr, i_va in unit_kfold_splits(tr, a.inner_splits, seed=seed):
                _, p = cross_validate(tr.iloc[np.union1d(i_tr, i_va)], names, boosting_factory(**CFG), a.cap, 2, seed)
                inner.append(p)
            inner = pd.concat(inner, ignore_index=True)
            d_score = best_shift(inner.y_true, inner.y_pred, nasa_score, a.deltas, a.cap)
            d_cost = best_shift(inner.y_true, inner.y_pred, lambda t, p: expected_cost(t, p, a.c_early, a.c_late), a.deltas, a.cap)

            q_rows = []
            for q in a.quantiles:
                _, p = cross_validate(tr, names, boosting_factory(**CFG, loss="quantile", quantile=q), a.cap, a.inner_splits, seed)
                q_rows.append({"q": q, "score": p.pipe(lambda x: nasa_score(x.y_true, x.y_pred) / len(x)),
                               "cost": expected_cost(p.y_true, p.y_pred, a.c_early, a.c_late)})
            q_rows = pd.DataFrame(q_rows)
            q_score = float(q_rows.loc[q_rows.score.idxmin(), "q"])
            q_cost = float(q_rows.loc[q_rows.cost.idxmin(), "q"])

            # Fit on the whole outer training set, evaluate on the held-out engines.
            outer = {}
            _, p_plain = cross_validate(feats, names, boosting_factory(**CFG), a.cap, a.splits, seed)
            g = p_plain[p_plain.fold == k]
            outer["plain (RMSE-trained)"] = g.y_pred.to_numpy()
            outer[f"shift, delta chosen for score"] = apply_shift(g.y_pred.to_numpy(), d_score, a.cap)
            outer[f"shift, delta chosen for cost"] = apply_shift(g.y_pred.to_numpy(), d_cost, a.cap)
            for label, q in (("quantile chosen for score", q_score), ("quantile chosen for cost", q_cost)):
                _, p_q = cross_validate(feats, names, boosting_factory(**CFG, loss="quantile", quantile=q), a.cap, a.splits, seed)
                outer[label] = p_q[p_q.fold == k].y_pred.to_numpy()

            y_true = g.y_true.to_numpy()
            for label, yp in outer.items():
                rowsC.append({"strategy": label, "seed": seed, "fold": k, "chosen": {
                    "plain (RMSE-trained)": 0.0, "shift, delta chosen for score": d_score,
                    "shift, delta chosen for cost": d_cost, "quantile chosen for score": q_score,
                    "quantile chosen for cost": q_cost}[label], **row_metrics(y_true, yp, a.c_early, a.c_late)})
            print(f"  C: seed {seed} fold {k} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    dC = pd.DataFrame(rowsC)
    tblC = dC.groupby("strategy").agg(chosen_mean=("chosen", "mean"), rmse=("rmse", "mean"), bias=("bias", "mean"),
                                      score_mean=("score_mean", "mean"), pct_late=("pct_late", "mean"),
                                      pct_late_gt10=("pct_late_gt10", "mean"), cost=("cost", "mean"))
    tblC.to_csv(out / "asym_nested.csv")
    print("\nPART C: nested cross-validation (delta and quantile chosen on inner folds only)")
    print(tblC.round(2).to_string())

    # ---------------- D: cost ratios ----------------
    rowsD = []
    for ratio in (1, 3, 10, 30, 100):
        obj = lambda t, p, r=ratio: expected_cost(t, p, 1.0, r)
        d = best_shift(base_preds.y_true, base_preds.y_pred, obj, np.arange(0, 61, 1), a.cap)
        shifted = apply_shift(base_preds.y_pred.to_numpy(), d, a.cap)
        rowsD.append({"c_late / c_early": ratio, "theory quantile": optimal_quantile(1.0, ratio), "best shift (cycles)": d,
                      "rmse at that shift": rmse(base_preds.y_true, shifted),
                      "cost at shift 0": expected_cost(base_preds.y_true, base_preds.y_pred, 1.0, ratio),
                      "cost at best shift": expected_cost(base_preds.y_true, shifted, 1.0, ratio),
                      "pct late": 100 * np.mean(shifted - base_preds.y_true.to_numpy() > 0)})
    tblD = pd.DataFrame(rowsD).set_index("c_late / c_early")
    tblD["cost saved %"] = 100 * (1 - tblD["cost at best shift"] / tblD["cost at shift 0"])
    tblD.to_csv(out / "asym_cost_ratios.csv")
    print("\nPART D: how far to shift for a given cost ratio (shift fitted on all CV predictions, so indicative only)")
    print(tblD.round(2).to_string())

    make_plot(tblA, tblB, out)

    (out / "asym_summary.json").write_text(json.dumps({
        "window": a.window, "cap": a.cap, "boosting": CFG, "cost_model": {"c_early": a.c_early, "c_late": a.c_late},
        "nested": {k: {m: round(float(v), 3) for m, v in r.items()} for k, r in tblC.iterrows()},
        "status": "cross-validation only; test set not scored"}, indent=1))
    print(f"\nTotal time {time.perf_counter() - t0:.0f}s. Test set not used.")


if __name__ == "__main__":
    main()
