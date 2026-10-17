"""Sensitivity sweeps for the provisional choices: RUL cap and window length.

Questions
  1. How does the RUL cap used for the training target change cross-validated error?
     (Errors are always measured against the UNCAPPED true RUL.)
  2. How does the window length change error, including windows longer than the shortest
     test engine (31 cycles), which need the edge-padding rule in ``windows.py``?
  3. Are the differences bigger than the noise from how engines happen to be assigned to folds?
     Everything is repeated with several fold seeds; a "paired" difference compares two settings
     on identical folds.

Design notes
  * Models: ridge (alpha 100) and gradient boosting (300 iterations, 8 leaves, min leaf 200),
    the settings chosen earlier. They are NOT re-tuned per sweep point, so each sweep isolates one factor.
  * Every window setting uses edge padding so that all settings train on every cycle and are scored on the
    same rows. A control row (window 30, no padding) shows what padding itself changes.
  * Scored rows are always cycle >= 31 with true RUL 10 to 150, the same as every earlier script.
    The script asserts that scored rows are identical across all settings within a seed.
  * The official test set is never loaded for scoring here.

    python scripts/run_sweeps.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss.cv import cross_validate  # noqa: E402
from cmapss.features import select_columns  # noqa: E402
from cmapss.loader import SENSOR_COLUMNS, SETTING_COLUMNS, add_rul, load_subset  # noqa: E402
from cmapss.models import RidgeRegressor, boosting_factory  # noqa: E402
from cmapss.windows import feature_names, window_features  # noqa: E402

ALL_COLS = SETTING_COLUMNS + SENSOR_COLUMNS
STATS = ("mean", "slope", "std")
KEY = ["fold", "unit", "cycle"]
MODELS = {"ridge": lambda: RidgeRegressor(100.0), "boosting": boosting_factory(300, 8, 200)}


def run_point(sweep, value, frame, cols, cap, seeds, splits, ref_keys, raw):
    """Cross-validate both models for one sweep value over all seeds; append per-fold rows to ``raw``."""
    for model, factory in MODELS.items():
        for seed in seeds:
            folds, preds = cross_validate(frame, cols, factory, cap, splits, seed)
            keys = preds[KEY].sort_values(KEY).reset_index(drop=True)
            if seed not in ref_keys:
                ref_keys[seed] = keys
            assert keys.equals(ref_keys[seed]), f"scored rows differ ({sweep}={value}, seed={seed})"
            for fold, r in folds.iterrows():
                raw.append({"sweep": sweep, "value": str(value), "model": model, "seed": seed, "fold": fold,
                            "rmse": r.rmse, "mae": r.mae, "bias": r.bias, "score_mean": r.score_mean,
                            "fit_seconds": r.fit_seconds, "predict_seconds": r.predict_seconds})


def summarise(raw, sweep, ref_value):
    """Per (value, model): mean over all folds and seeds; spread across seeds; paired difference vs the reference value."""
    d = raw[raw.sweep == sweep]
    rows = []
    for model in MODELS:
        m = d[d.model == model]
        ref = m[m.value == ref_value].set_index(["seed", "fold"])
        for value in dict.fromkeys(m.value):
            v = m[m.value == value].set_index(["seed", "fold"])
            diff = (v.rmse - ref.rmse)
            per_seed = v.groupby("seed").rmse.mean()
            rows.append({"model": model, "value": value, "rmse": v.rmse.mean(),
                         "rmse_sd_across_seeds": per_seed.std(ddof=1), "rmse_sd_across_folds": v.rmse.std(ddof=1),
                         "mae": v.mae.mean(), "bias": v.bias.mean(), "score_mean": v.score_mean.mean(),
                         "fit_s": v.fit_seconds.mean(), "predict_s": v.predict_seconds.mean(),
                         "paired_rmse_vs_ref": diff.mean(), "folds_better_than_ref": int((diff < 0).sum()), "n_folds": len(diff)})
    return pd.DataFrame(rows).set_index(["model", "value"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--caps", type=int, nargs="+", default=[75, 100, 110, 125, 140, 150])
    ap.add_argument("--windows", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50, 60, 80])
    ap.add_argument("--ref-cap", type=int, default=125)
    ap.add_argument("--ref-window", type=int, default=30)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    t_start = time.perf_counter()

    data = load_subset("FD001", a.data_dir)
    train = data.train
    kept, _ = select_columns(train, ALL_COLS, min_trend=0.3)
    names = feature_names(kept, STATS)
    print(f"{len(kept)} sensors, {len(names)} windowed features; seeds {a.seeds}; {a.splits} folds; "
          f"{len(a.seeds) * a.splits} fold-fits per setting and model\n")

    raw, ref_keys = [], {}

    # ---- sweep 1: cap (window fixed) ----
    feats = window_features(train, kept, a.ref_window, STATS, pad="edge")
    caps = [*a.caps, "none"]
    for cap in caps:
        run_point("cap", cap, feats, names, None if cap == "none" else cap, a.seeds, a.splits, ref_keys, raw)
        print(f"  cap {cap} done ({time.perf_counter() - t_start:.0f}s)", flush=True)

    # ---- sweep 2: window (cap fixed) ----
    for w in a.windows:
        run_point("window", w, window_features(train, kept, w, STATS, pad="edge"), names, a.ref_cap, a.seeds, a.splits, ref_keys, raw)
        print(f"  window {w} done ({time.perf_counter() - t_start:.0f}s)", flush=True)
    # control: no padding (only rows with a full window; window <= 31 so every scored row still has features)
    run_point("window", f"{a.ref_window} (no padding)", window_features(train, kept, a.ref_window, STATS), names,
              a.ref_cap, a.seeds, a.splits, ref_keys, raw)

    raw = pd.DataFrame(raw); raw.to_csv(out / "sweep_raw.csv", index=False)
    cap_tbl = summarise(raw, "cap", str(a.ref_cap)); win_tbl = summarise(raw, "window", str(a.ref_window))

    # Unavoidable error from the cap: an oracle that predicts min(true RUL, cap) on exactly the scored rows.
    lab = add_rul(train)
    sc = lab[(lab.cycle >= 31) & lab.RUL.between(10, 150)]
    floor = {str(c): float(np.sqrt(np.mean((sc.RUL - (sc.RUL.clip(upper=c) if c != "none" else sc.RUL)) ** 2))) for c in caps}
    cap_tbl["oracle_floor_rmse"] = [floor[v] for _, v in cap_tbl.index]
    cap_tbl.to_csv(out / "sweep_cap.csv"); win_tbl.to_csv(out / "sweep_window.csv")

    cols = ["rmse", "rmse_sd_across_seeds", "rmse_sd_across_folds", "bias", "score_mean", "fit_s", "paired_rmse_vs_ref", "folds_better_than_ref"]
    print(f"\nCAP SWEEP (window {a.ref_window}, edge padding; reference = cap {a.ref_cap}; {len(a.seeds) * a.splits} folds per row)")
    print(cap_tbl[cols + ["oracle_floor_rmse"]].round(2).to_string())
    print(f"\nWINDOW SWEEP (cap {a.ref_cap}; reference = window {a.ref_window} with edge padding)")
    print(win_tbl[cols].round(2).to_string())

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    colors = {"ridge": "C1", "boosting": "C0"}
    numeric_caps = [str(c) for c in a.caps]
    for model in MODELS:
        t = cap_tbl.loc[model]
        ax[0].errorbar([int(v) for v in numeric_caps], t.loc[numeric_caps, "rmse"], yerr=t.loc[numeric_caps, "rmse_sd_across_seeds"],
                       fmt="o-", capsize=3, color=colors[model], label=model)
        ax[0].axhline(t.loc["none", "rmse"], color=colors[model], ls=":", lw=1)
        w = win_tbl.loc[model]
        wk = [str(x) for x in a.windows]
        ax[1].errorbar(a.windows, w.loc[wk, "rmse"], yerr=w.loc[wk, "rmse_sd_across_seeds"], fmt="o-", capsize=3, color=colors[model], label=model)
    ax[0].plot([int(v) for v in numeric_caps], [floor[v] for v in numeric_caps], "k--", label="oracle floor from cap")
    ax[0].set(xlabel="RUL cap used for the training target (cycles)", ylabel="CV RMSE (cycles)", title="Cap sweep (dotted = no cap)")
    ax[1].axvline(31, color="k", ls="--", lw=1); ax[1].text(32, ax[1].get_ylim()[1] * 0.99, "shortest test engine = 31", va="top", fontsize=8)
    ax[1].set(xlabel="window length N (cycles; edge-padded)", ylabel="CV RMSE (cycles)", title="Window sweep")
    for x in ax: x.grid(alpha=.3); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "sweeps.png", dpi=100)
    print(f"\nTotal time {time.perf_counter() - t_start:.0f}s. Test set not used.")


if __name__ == "__main__":
    main()
