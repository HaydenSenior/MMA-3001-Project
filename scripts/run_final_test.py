"""FINAL EVALUATION on the held-out C-MAPSS FD001 test set. Run once.

The configuration below was frozen before this script was first run; nothing here is tuned.
Every choice came from cross-validation on the training engines only:

    features   14 sensors kept by trend screening (features.select_columns)
    window     50 cycles, edge-padded          (sweeps: error still falling at 30, flat 50-60)
    cap        125 cycles                      (sweeps: best of 75-150 and no cap, for both models)
    baseline   ridge alpha=1 on single-cycle features        (the brief's comparison point)
    final      boosting, 200 iters, 16 leaves, min_samples_leaf=100   (best CV RMSE)
    operational same model trained at quantile 0.1           (maintenance recommendation)

Each model trains on all 100 training engines and predicts from the LAST observed cycle of
each of the 100 test engines, which is the only prediction the official RUL file scores.

The asymmetric score is reported as a plain SUM over the 100 engines, which is how published
FD001 results quote it. Cross-validation numbers elsewhere in this project use a per-row mean
and are NOT comparable to it.

    python scripts/run_final_test.py
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss.cv import fit_predict_test  # noqa: E402
from cmapss.decision import expected_cost  # noqa: E402
from cmapss.features import select_columns  # noqa: E402
from cmapss.loader import SENSOR_COLUMNS, SETTING_COLUMNS, load_subset  # noqa: E402
from cmapss.metrics import evaluate, nasa_score  # noqa: E402
from cmapss.models import RidgeRegressor, boosting_factory  # noqa: E402
from cmapss.windows import feature_names, window_features  # noqa: E402

ALL_COLS = SETTING_COLUMNS + SENSOR_COLUMNS
STATS = ("mean", "slope", "std")
FROZEN = {"window": 50, "cap": 125, "boosting": dict(iters=200, leaves=16, min_leaf=100), "quantile": 0.1}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--c-early", type=float, default=1.0)
    ap.add_argument("--c-late", type=float, default=10.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cap, win = FROZEN["cap"], FROZEN["window"]

    data = load_subset("FD001", a.data_dir)
    kept, _ = select_columns(data.train, ALL_COLS, min_trend=0.3)
    wtrain = window_features(data.train, kept, win, STATS, pad="edge")
    wtest = window_features(data.test, kept, win, STATS, pad="edge")
    wnames = feature_names(kept, STATS)
    truth = data.test_rul.sort_index()
    observed = data.test.groupby("unit").cycle.max().sort_index()
    print(f"Frozen configuration: {json.dumps(FROZEN)}")
    print(f"{len(kept)} sensors; {len(data.train.unit.unique())} training engines; {len(truth)} test engines")
    print(f"Test engines with fewer observed cycles than the window ({win}): "
          f"{int((observed < win).sum())} -> their features rely on edge padding\n")

    models = {
        "baseline (ridge, single cycle)": lambda: fit_predict_test(data.train, data.test, kept, lambda: RidgeRegressor(1.0), cap),
        "final (boosting, windowed)": lambda: fit_predict_test(wtrain, wtest, wnames, boosting_factory(**FROZEN["boosting"]), cap),
        "operational (boosting q0.1)": lambda: fit_predict_test(
            wtrain, wtest, wnames, boosting_factory(**FROZEN["boosting"], loss="quantile", quantile=FROZEN["quantile"]), cap),
    }

    preds, rows = {}, []
    for label, fn in models.items():
        p = fn().sort_index()
        assert p.index.equals(truth.index), f"{label}: prediction index does not match the RUL file"
        preds[label] = p
        m = evaluate(truth.to_numpy(), p.to_numpy())
        d = p.to_numpy() - truth.to_numpy()
        rows.append({"model": label, "rmse": m["rmse"], "mae": m["mae"], "bias": m["bias"],
                     "score_sum": nasa_score(truth.to_numpy(), p.to_numpy()),
                     "score_per_engine": nasa_score(truth.to_numpy(), p.to_numpy()) / len(truth),
                     "cost": expected_cost(truth.to_numpy(), p.to_numpy(), a.c_early, a.c_late),
                     "pct_late": 100 * np.mean(d > 0), "pct_late_gt10": 100 * np.mean(d > 10),
                     "worst_late": d.max(), "worst_early": d.min()})
    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(out / "final_test_summary.csv")
    print("=" * 110)
    print("TEST-SET RESULTS (100 engines, scored once)")
    print("=" * 110)
    print(table.round(2).to_string())
    print("\nscore_sum is the quantity published FD001 papers report; score_per_engine is it divided by 100.")

    per_engine = pd.DataFrame({"true_rul": truth, "observed_cycles": observed})
    for label, p in preds.items():
        per_engine[label] = p
        per_engine[f"err: {label}"] = p - truth
    per_engine.to_csv(out / "final_test_predictions.csv")

    # ---- error by true-RUL band ----
    bands = pd.cut(truth, [0, 25, 50, 75, 100, 150], include_lowest=True)
    band_rows = []
    for label, p in preds.items():
        for band, idx in truth.groupby(bands, observed=True).groups.items():
            d = (p.loc[idx] - truth.loc[idx]).to_numpy()
            band_rows.append({"model": label, "true RUL band": str(band), "engines": len(idx),
                              "rmse": float(np.sqrt(np.mean(d**2))), "bias": float(d.mean()),
                              "pct_late": 100 * float(np.mean(d > 0))})
    bands_tbl = pd.DataFrame(band_rows).set_index(["model", "true RUL band"])
    bands_tbl.to_csv(out / "final_test_by_band.csv")
    print("\nERROR BY TRUE-RUL BAND")
    print(bands_tbl.round(2).to_string())

    # ---- engines whose window relied on padding ----
    short = observed < win
    pad_rows = []
    for label, p in preds.items():
        for name, mask in (("observed < window (padded)", short), ("observed >= window", ~short)):
            d = (p[mask.values] - truth[mask.values]).to_numpy()
            if len(d):
                pad_rows.append({"model": label, "group": name, "engines": int(mask.sum()),
                                 "rmse": float(np.sqrt(np.mean(d**2))), "bias": float(d.mean())})
    pad_tbl = pd.DataFrame(pad_rows).set_index(["model", "group"])
    pad_tbl.to_csv(out / "final_test_padding.csv")
    print("\nENGINES WHOSE FEATURES RELIED ON EDGE PADDING")
    print(pad_tbl.round(2).to_string())

    # ---- worst engines for the final model ----
    key = "final (boosting, windowed)"
    worst = per_engine.assign(abs_err=lambda f: f[f"err: {key}"].abs()).nlargest(10, "abs_err")
    worst_cols = ["true_rul", "observed_cycles", key, f"err: {key}"]
    worst[worst_cols].to_csv(out / "final_test_worst_engines.csv")
    print(f"\nTEN WORST ENGINES for the final model")
    print(worst[worst_cols].round(1).to_string())

    # ---- figure ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharex=True, sharey=True)
    for axis, (label, p) in zip(ax, preds.items()):
        late = p.to_numpy() > truth.to_numpy()
        axis.scatter(truth[~late], p[~late], s=22, color="C0", label="early (safe)")
        axis.scatter(truth[late], p[late], s=22, color="C3", label="late (risky)")
        axis.plot([0, 150], [0, 150], "k--", lw=1, label="perfect")
        axis.axhline(cap, color="grey", ls=":", lw=1)
        r = table.loc[label]
        axis.set(xlabel="true RUL (cycles)", title=f"{label}\nRMSE {r.rmse:.1f}, score {r.score_sum:.0f}, late {r.pct_late:.0f}%")
        axis.grid(alpha=.3); axis.legend(fontsize=8, loc="upper left")
    ax[0].set_ylabel("predicted RUL (cycles)")
    fig.tight_layout(); fig.savefig(out / "final_test.png", dpi=100)

    (out / "final_test_summary.json").write_text(json.dumps({
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "frozen_config": FROZEN,
        "sensors": kept, "n_test_engines": int(len(truth)),
        "padded_engines": int(short.sum()),
        "results": {k: {m: round(float(v), 3) for m, v in r.items()} for k, r in table.iterrows()},
        "note": "single evaluation on the held-out test set; configuration frozen beforehand"}, indent=1))
    print("\nWritten to results/final_test_*.csv/.png/.json")


if __name__ == "__main__":
    main()
