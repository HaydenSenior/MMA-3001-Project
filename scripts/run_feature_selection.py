"""Noise-reduction variants, prediction smoothing and boosting re-tuning, at window 50.

Part A  Which input treatment helps? Raw features vs per-engine baseline removal, EWMA, rolling median,
        combinations, and PCA of the window features. All causal and all per-engine.
Part B  Does smoothing the PREDICTIONS along an engine's history (averaging p_{t-j} - j) help?
Part C  Re-tune boosting on the chosen features (3 fold seeds).

Adoption rule (fixed in advance, to avoid chasing noise): a variant replaces the reference only if it has a lower
mean CV RMSE AND is better in at least ``--min-wins`` of the paired folds (default 12 of 15). Otherwise the
reference is kept. Same folds, same scored rows everywhere (asserted). Test set not used.

    python scripts/run_feature_selection.py
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss.cv import cross_validate  # noqa: E402
from cmapss.denoise import baseline_normalise, ewma, rolling_median  # noqa: E402
from cmapss.features import select_columns  # noqa: E402
from cmapss.loader import SENSOR_COLUMNS, SETTING_COLUMNS, load_subset  # noqa: E402
from cmapss.metrics import evaluate  # noqa: E402
from cmapss.models import PCAPipeline, RidgeRegressor, boosting_factory  # noqa: E402
from cmapss.postprocess import propagate_smooth  # noqa: E402
from cmapss.windows import feature_names, window_features  # noqa: E402

ALL_COLS = SETTING_COLUMNS + SENSOR_COLUMNS
STATS = ("mean", "slope", "std")
KEY = ["fold", "unit", "cycle"]
RIDGE = lambda: RidgeRegressor(100.0)  # noqa: E731
BOOST = boosting_factory(300, 8, 200)


def paired(df, value, ref):
    """Paired RMSE difference of ``value`` vs ``ref`` over identical (seed, fold) pairs."""
    a = df[df.value == value].set_index(["seed", "fold"]).rmse
    b = df[df.value == ref].set_index(["seed", "fold"]).rmse
    d = a - b
    return d.mean(), int((d < 0).sum()), len(d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--cap", type=int, default=125)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--min-wins", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    t0 = time.perf_counter()

    train = load_subset("FD001", a.data_dir).train
    kept, _ = select_columns(train, ALL_COLS, min_trend=0.3)
    names = feature_names(kept, STATS)
    n_runs = len(a.seeds) * a.splits
    print(f"window {a.window} (edge-padded), cap {a.cap}, {len(a.seeds)} seeds x {a.splits} folds = {n_runs} paired folds per row\n")

    ref_keys, raw_rows = {}, []

    def run(label, frame, cols, model_name, factory):
        for seed in a.seeds:
            folds, preds = cross_validate(frame, cols, factory, a.cap, a.splits, seed)
            keys = preds[KEY].sort_values(KEY).reset_index(drop=True)
            ref_keys.setdefault(seed, keys)
            assert keys.equals(ref_keys[seed]), f"scored rows differ: {label}, seed {seed}"
            for fold, r in folds.iterrows():
                raw_rows.append({"part": "A", "value": label, "model": model_name, "seed": seed, "fold": fold, "rmse": r.rmse,
                                 "mae": r.mae, "bias": r.bias, "score_mean": r.score_mean, "fit_seconds": r.fit_seconds})

    # ---------------- Part A: input treatments ----------------
    treatments = {
        "raw (reference)": lambda d: d,
        "baseline-normalised": lambda d: baseline_normalise(d, kept, 30),
        "EWMA span 10": lambda d: ewma(d, kept, 10),
        "rolling median 5": lambda d: rolling_median(d, kept, 5),
        "normalised + EWMA 10": lambda d: ewma(baseline_normalise(d, kept, 30), kept, 10),
    }
    frames = {k: window_features(f(train), kept, a.window, STATS, pad="edge") for k, f in treatments.items()}
    for label, frame in frames.items():
        for mname, fac in (("ridge", RIDGE), ("boosting", BOOST)):
            run(label, frame, names, mname, fac)
        print(f"  A: {label} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    for k in (5, 10):
        label = f"raw + PCA({k})"
        for mname, fac in (("ridge", RIDGE), ("boosting", BOOST)):
            run(label, frames["raw (reference)"], names, mname, lambda f=fac, k=k: PCAPipeline(f, k))
        print(f"  A: {label} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    rawdf = pd.DataFrame(raw_rows)
    rows = []
    for mname in ("ridge", "boosting"):
        d = rawdf[(rawdf.part == "A") & (rawdf.model == mname)]
        for label in dict.fromkeys(d.value):
            v = d[d.value == label]
            diff, wins, n = paired(d, label, "raw (reference)")
            rows.append({"model": mname, "variant": label, "rmse": v.rmse.mean(), "rmse_sd_folds": v.rmse.std(ddof=1), "bias": v.bias.mean(),
                         "score_mean": v.score_mean.mean(), "paired_vs_raw": diff, "folds_better": f"{wins}/{n}"})
    tblA = pd.DataFrame(rows).set_index(["model", "variant"]); tblA.to_csv(out / "features_variants.csv")
    print("\nPART A: input treatments (RMSE in cycles; paired_vs_raw < 0 means better than raw)")
    print(tblA.round(2).to_string())

    bd = rawdf[(rawdf.part == "A") & (rawdf.model == "boosting")]
    best, best_rmse = "raw (reference)", bd[bd.value == "raw (reference)"].rmse.mean()
    for label in dict.fromkeys(bd.value):
        diff, wins, _ = paired(bd, label, "raw (reference)")
        if label != "raw (reference)" and bd[bd.value == label].rmse.mean() < best_rmse and wins >= a.min_wins:
            best, best_rmse = label, bd[bd.value == label].rmse.mean()
    print(f"\nAdoption rule (boosting, >= {a.min_wins}/{n_runs} paired folds better): chosen input = '{best}'")

    # ---------------- Part B: smoothing the predictions ----------------
    is_pca = best.startswith("raw + PCA")
    frame_best = frames["raw (reference)"] if is_pca else frames[best]
    fac_best = (lambda: PCAPipeline(BOOST, int(best.split("(")[1].rstrip(")")))) if is_pca else BOOST
    js = (0, 5, 10, 20, 30)
    rowsB = []
    for seed in a.seeds:
        folds, preds = cross_validate(frame_best, names, fac_best, a.cap, a.splits, seed)
        for j in js:
            preds["s"] = propagate_smooth(preds.rename(columns={"y_pred": "y_pred"}), j, a.cap)
            for fold, g in preds.groupby("fold"):
                m = evaluate(g.y_true.to_numpy(), g.s.to_numpy())
                rowsB.append({"part": "B", "value": f"J={j}", "model": "boosting", "seed": seed, "fold": fold, "rmse": m["rmse"], "mae": m["mae"],
                              "bias": m["bias"], "score_mean": m["score"] / m["n"], "fit_seconds": np.nan})
    dB = pd.DataFrame(rowsB); raw_rows.extend(rowsB)
    tblB = []
    for j in js:
        v = dB[dB.value == f"J={j}"]; diff, wins, n = paired(dB, f"J={j}", "J=0")
        tblB.append({"smoothing J": j, "rmse": v.rmse.mean(), "bias": v.bias.mean(), "score_mean": v.score_mean.mean(), "paired_vs_J0": diff, "folds_better": f"{wins}/{n}"})
    tblB = pd.DataFrame(tblB).set_index("smoothing J"); tblB.to_csv(out / "features_smoothing.csv")
    print(f"\nPART B: smoothing predictions along each engine's history (input = '{best}', boosting)")
    print(tblB.round(2).to_string())
    bestJ = 0
    for j in js[1:]:
        diff, wins, _ = paired(dB, f"J={j}", "J=0")
        if dB[dB.value == f"J={j}"].rmse.mean() < dB[dB.value == f"J={bestJ}"].rmse.mean() and wins >= a.min_wins:
            bestJ = j
    print(f"Adoption rule: chosen smoothing J = {bestJ}")
    print("(Note: in cross-validation the first scored rows of an engine have fewer earlier predictions to average, so the benefit is slightly understated.)")

    # ---------------- Part C: re-tune boosting ----------------
    rowsC = []
    for it, leaves, ml in itertools.product((200, 400, 600), (8, 16), (100, 200)):
        for seed in a.seeds:
            folds, preds = cross_validate(frame_best, names, (lambda: PCAPipeline(boosting_factory(it, leaves, ml), int(best.split("(")[1].rstrip(")")))) if is_pca else boosting_factory(it, leaves, ml), a.cap, a.splits, seed)
            assert preds[KEY].sort_values(KEY).reset_index(drop=True).equals(ref_keys[seed])
            for fold, r in folds.iterrows():
                rowsC.append({"iters": it, "leaves": leaves, "min_leaf": ml, "seed": seed, "fold": fold, "rmse": r.rmse, "bias": r.bias,
                              "score_mean": r.score_mean, "fit_s": r.fit_seconds})
        print(f"  C: iters={it} leaves={leaves} min_leaf={ml} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    dC = pd.DataFrame(rowsC)
    tblC = dC.groupby(["iters", "leaves", "min_leaf"]).agg(rmse=("rmse", "mean"), rmse_sd_folds=("rmse", "std"), bias=("bias", "mean"),
                                                         score_mean=("score_mean", "mean"), fit_s=("fit_s", "mean")).round(3)
    tblC.to_csv(out / "features_boosting_grid.csv")
    ref_cfg = (300, 8, 200)
    best_cfg = tblC.rmse.idxmin()
    print("\nPART C: boosting grid at the chosen input (mean over all folds and seeds)")
    print(tblC.to_string())
    print(f"\nlowest RMSE: {best_cfg} = {tblC.rmse.min():.3f}; previous setting {ref_cfg} is not in this grid (300 iterations).")
    print(f"spread across the whole grid: {tblC.rmse.max() - tblC.rmse.min():.2f} cycles (fold-to-fold sd is about {tblC.rmse_sd_folds.mean():.2f})")

    pd.DataFrame(raw_rows).to_csv(out / "features_raw.csv", index=False)
    cfg = {"window": a.window, "cap": a.cap, "input": best, "prediction_smoothing_J": bestJ,
           "boosting": {"iters": int(best_cfg[0]), "leaves": int(best_cfg[1]), "min_leaf": int(best_cfg[2])}, "status": "PROPOSED, not frozen; test set not scored"}
    (out / "proposed_config.json").write_text(json.dumps(cfg, indent=1))
    print("\nProposed configuration:", json.dumps(cfg))
    print(f"Total time {time.perf_counter() - t0:.0f}s. Test set not used.")


if __name__ == "__main__":
    main()
