"""Performance analysis of the RUL pipeline: timing, scaling, FLOPs, memory and model size.

Covers the brief's performance requirements:

  1  Scaling       measured fit/predict time against the FLOP model, over a range of rows and features
  2  Throughput    achieved GFLOP/s versus what this machine can sustain, and arithmetic intensity
  3  Profile       cProfile of the end-to-end pipeline: where the time actually goes
  4  Memory        peak allocation per stage, against the theoretical array footprint, and model size
  5  Comparison    baseline vs final vs operational model on accuracy, time, memory and size

Timings are machine-specific: rerun this on the machine the report quotes. The official test
set is not used.

    python scripts/run_profiling.py
"""
import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss.cv import cross_validate  # noqa: E402
from cmapss.decision import expected_cost  # noqa: E402
from cmapss.features import select_columns  # noqa: E402
from cmapss.loader import SENSOR_COLUMNS, SETTING_COLUMNS, add_rul, load_subset  # noqa: E402
from cmapss.models import RidgeRegressor, boosting_factory  # noqa: E402
from cmapss.profiling import (  # noqa: E402
    arithmetic_intensity,
    model_size_bytes,
    peak_memory,
    ridge_fit_flops,
    ridge_predict_bytes,
    ridge_predict_flops,
    time_callable,
)
from cmapss.windows import feature_names, window_features  # noqa: E402

ALL_COLS = SETTING_COLUMNS + SENSOR_COLUMNS
STATS = ("mean", "slope", "std")
CFG = dict(iters=200, leaves=16, min_leaf=100)
MB = 1024 ** 2


def machine_gflops(size: int = 1200, repeats: int = 3) -> float:
    """Sustained double-precision throughput on a dense matrix multiply, as a practical ceiling."""
    a, b = np.random.default_rng(0).normal(size=(size, size)), np.random.default_rng(1).normal(size=(size, size))
    t = time_callable(lambda: a @ b, repeats=repeats, warmup=1)
    return (2.0 * size ** 3) / t["min"] / 1e9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--cap", type=int, default=125)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--syn-features", type=int, nargs="+", default=[25, 50, 100, 200, 400],
                    help="feature counts for the synthetic wide-matrix scaling study")
    ap.add_argument("--c-early", type=float, default=1.0)
    ap.add_argument("--c-late", type=float, default=10.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

    data = load_subset("FD001", a.data_dir)
    train = data.train
    kept, _ = select_columns(train, ALL_COLS, min_trend=0.3)
    feats = window_features(train, kept, a.window, STATS, pad="edge")
    names = feature_names(kept, STATS)
    labelled = add_rul(feats, cap=a.cap)
    X, y = labelled[names].to_numpy(), labelled["RUL_capped"].to_numpy()
    n, p = X.shape
    print(f"Feature matrix: {n} rows x {p} features ({X.nbytes / MB:.1f} MB as float64)")
    peak = machine_gflops()
    print(f"Sustained dense-matmul throughput on this machine: {peak:.1f} GFLOP/s (practical ceiling)\n")

    # ---------------- 1. scaling ----------------
    rows = []
    for frac in (0.125, 0.25, 0.5, 1.0):
        m = max(100, int(n * frac))
        Xs, ys = X[:m], y[:m]
        fit = time_callable(lambda: RidgeRegressor(100.0).fit(Xs, ys), a.repeats)
        model = RidgeRegressor(100.0).fit(Xs, ys)
        pred = time_callable(lambda: model.predict(Xs), a.repeats)
        rows.append({"vary": "rows", "n": m, "p": p, "fit_ms": 1e3 * fit["min"], "fit_sd_ms": 1e3 * fit["std"],
                     "fit_GFLOPs": ridge_fit_flops(m, p) / fit["min"] / 1e9,
                     "pred_ms": 1e3 * pred["min"], "pred_GFLOPs": ridge_predict_flops(m, p) / pred["min"] / 1e9})
    for q in (7, 14, 28, 42):
        Xs = X[:, :q]
        fit = time_callable(lambda: RidgeRegressor(100.0).fit(Xs, y), a.repeats)
        model = RidgeRegressor(100.0).fit(Xs, y)
        pred = time_callable(lambda: model.predict(Xs), a.repeats)
        rows.append({"vary": "features", "n": n, "p": q, "fit_ms": 1e3 * fit["min"], "fit_sd_ms": 1e3 * fit["std"],
                     "fit_GFLOPs": ridge_fit_flops(n, q) / fit["min"] / 1e9,
                     "pred_ms": 1e3 * pred["min"], "pred_GFLOPs": ridge_predict_flops(n, q) / pred["min"] / 1e9})
    scaling = pd.DataFrame(rows)
    scaling.to_csv(out / "perf_scaling.csv", index=False)
    print("1. SCALING of the hand-written ridge (minimum of %d runs)" % a.repeats)
    print(scaling.round(3).to_string(index=False))
    r = scaling[scaling.vary == "rows"]
    slope = np.polyfit(np.log(r.n), np.log(r.fit_ms), 1)[0]
    f = scaling[scaling.vary == "features"]
    slope_p = np.polyfit(np.log(f.p), np.log(f.fit_ms), 1)[0]
    print(f"\n   fit time vs rows:     measured exponent {slope:.2f} (FLOP model predicts 1.0 for n >> p)")
    print(f"   fit time vs features: measured exponent {slope_p:.2f} (FLOP model predicts 2.0)")

    # The real data is too narrow (p <= 42) to show the asymptotic law, so widen it synthetically.
    rng = np.random.default_rng(0)
    wide, n_syn = [], 60000
    for q_ in a.syn_features:
        Xw, yw = rng.normal(size=(n_syn, q_)), rng.normal(size=n_syn)
        tw = time_callable(lambda: RidgeRegressor(1.0).fit(Xw, yw), 3)["min"]
        wide.append({"p": q_, "fit_ms": 1e3 * tw, "model_MFLOP": ridge_fit_flops(n_syn, q_) / 1e6,
                     "achieved_GFLOPs": ridge_fit_flops(n_syn, q_) / tw / 1e9,
                     "pct_of_ceiling": 100 * ridge_fit_flops(n_syn, q_) / tw / 1e9 / peak})
    wide = pd.DataFrame(wide)
    wide.to_csv(out / "perf_scaling_synthetic.csv", index=False)
    e_flops = np.polyfit(np.log(wide.p), np.log(wide.model_MFLOP), 1)[0]
    e_time = np.polyfit(np.log(wide.p), np.log(wide.fit_ms), 1)[0]
    e_rate = np.polyfit(np.log(wide.p), np.log(wide.achieved_GFLOPs), 1)[0]
    print(f"\n   Synthetic wide matrices (n = {n_syn} fixed) to reach the asymptotic regime:")
    print(wide.round(2).to_string(index=False))
    print(f"\n   work  grows as p^{e_flops:.2f}  (the FLOP model, essentially quadratic)")
    print(f"   rate  grows as p^{e_rate:.2f}  (wider matrices use cache and vector units far better)")
    print(f"   time  grows as p^{e_time:.2f}  = work / rate, which matches {e_flops - e_rate:.2f}")
    print("   So the FLOP count is sound; wall-clock time grows more slowly only because throughput improves.")

    # ---------------- 2. throughput and intensity ----------------
    fit_ai = arithmetic_intensity(ridge_fit_flops(n, p), 8.0 * (n + p) * p)
    pred_ai = arithmetic_intensity(ridge_predict_flops(n, p), ridge_predict_bytes(n, p))
    full = scaling[(scaling.vary == "rows") & (scaling.n == n)].iloc[0]
    print("\n2. THROUGHPUT AND ARITHMETIC INTENSITY at the full problem size")
    print(f"   fit:     {ridge_fit_flops(n, p) / 1e6:8.1f} MFLOP, {fit_ai:6.2f} FLOP/byte, achieved {full.fit_GFLOPs:.2f} GFLOP/s "
          f"({100 * full.fit_GFLOPs / peak:.1f}% of ceiling)")
    print(f"   predict: {ridge_predict_flops(n, p) / 1e6:8.1f} MFLOP, {pred_ai:6.2f} FLOP/byte, achieved {full.pred_GFLOPs:.2f} GFLOP/s "
          f"({100 * full.pred_GFLOPs / peak:.1f}% of ceiling)")
    print("   Prediction moves about 8 bytes per 2 FLOPs, so it is limited by memory bandwidth, not arithmetic:")
    print("   making the arithmetic cheaper would not make it faster. The fit is the compute-bound step,")
    print("   but at 42 features it still reaches only a few percent of the ceiling, because a tall narrow")
    print("   factorisation cannot fill the vector units the way a large square matrix multiply does.")

    # ---------------- 3. end-to-end profile ----------------
    def pipeline():
        d = load_subset("FD001", a.data_dir)
        k, _ = select_columns(d.train, ALL_COLS, min_trend=0.3)
        fe = window_features(d.train, k, a.window, STATS, pad="edge")
        lab = add_rul(fe, cap=a.cap)
        nm = feature_names(k, STATS)
        m = RidgeRegressor(100.0).fit(lab[nm].to_numpy(), lab["RUL_capped"].to_numpy())
        return m.predict(lab[nm].to_numpy())

    pr = cProfile.Profile(); pr.enable(); pipeline(); pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(12)
    text = buf.getvalue()
    (out / "perf_profile.txt").write_text(text)
    print("\n3. END-TO-END PROFILE (load -> screen -> features -> fit -> predict), top entries by cumulative time")
    print("\n".join(text.splitlines()[4:20]))

    # ---------------- 4. memory and model size ----------------
    def warm_peak(fn):
        """Peak allocation of fn, after one untimed call so first-call overhead is not counted."""
        fn()
        return peak_memory(fn)[1]

    mem_rows = []
    m_load = warm_peak(lambda: load_subset("FD001", a.data_dir))
    mem_rows.append({"stage": "load + validate 3 files", "peak_MB": m_load / MB,
                     "theoretical_MB": (train.memory_usage(deep=True).sum() + data.test.memory_usage(deep=True).sum()) / MB})
    m_feat = warm_peak(lambda: window_features(train, kept, a.window, STATS, pad="edge"))
    mem_rows.append({"stage": f"window features (N={a.window})", "peak_MB": m_feat / MB,
                     "theoretical_MB": 8 * len(train) * len(kept) * a.window / MB,
                     "note": "theoretical = windows as real data; the view avoids the copy"})
    m_ridge = warm_peak(lambda: RidgeRegressor(100.0).fit(X, y))
    mem_rows.append({"stage": "ridge fit", "peak_MB": m_ridge / MB, "theoretical_MB": 8 * (n + p) * p / MB})
    m_gbm = warm_peak(lambda: boosting_factory(**CFG)().fit(X, y))
    mem_rows.append({"stage": "boosting fit", "peak_MB": m_gbm / MB, "theoretical_MB": float("nan")})
    mem = pd.DataFrame(mem_rows)
    mem.to_csv(out / "perf_memory.csv", index=False)
    print("\n4. PEAK MEMORY PER STAGE (tracemalloc; theoretical = size of the arrays the step must hold)")
    print(mem.round(1).to_string(index=False))
    print(f"   Note the window step: holding every trailing window as real data would need "
          f"{8 * len(train) * len(kept) * a.window / MB:.0f} MB, but it peaks far below that because")
    print("   sliding_window_view returns a strided VIEW of the original array, so the windows are never copied.")
    print("   That is the single most important implementation choice for memory in this pipeline.")

    # ---------------- 5. baseline vs final vs operational ----------------
    snapshot = add_rul(train, cap=a.cap)
    variants = {
        "baseline: ridge, single snapshot": (train, kept, lambda: RidgeRegressor(1.0)),
        "final: boosting, windowed": (feats, names, boosting_factory(**CFG)),
        "operational: boosting q0.1, windowed": (feats, names, boosting_factory(**CFG, loss="quantile", quantile=0.1)),
    }
    comp = []
    for label, (frame, cols, factory) in variants.items():
        folds, preds = cross_validate(frame, cols, factory, a.cap, a.splits, a.seed)
        lab = snapshot if frame is train else labelled
        Xi, yi = lab[cols].to_numpy(), lab["RUL_capped"].to_numpy()
        fit_t = time_callable(lambda: factory().fit(Xi, yi), max(3, a.repeats // 2))
        model = factory().fit(Xi, yi)
        pred_t = time_callable(lambda: model.predict(Xi), a.repeats)
        mem_fit = warm_peak(lambda: factory().fit(Xi, yi))
        comp.append({"model": label, "features": len(cols), "rmse": folds.rmse.mean(),
                     "score_mean": folds.score_mean.mean(),
                     "cost": expected_cost(preds.y_true, preds.y_pred, a.c_early, a.c_late),
                     "pct_late": 100 * np.mean(preds.y_pred > preds.y_true),
                     "fit_s": fit_t["min"], "predict_us_per_1k": 1e6 * pred_t["min"] / (len(Xi) / 1000),
                     "fit_peak_MB": mem_fit / MB, "model_KB": model_size_bytes(model) / 1024})
    comp = pd.DataFrame(comp).set_index("model")
    comp.to_csv(out / "perf_comparison.csv")
    print("\n5. BASELINE vs FINAL vs OPERATIONAL (accuracy from %d-fold CV, seed %d; timings on the full training set)" % (a.splits, a.seed))
    print(comp.round(3).to_string())
    b, f_ = comp.iloc[0], comp.iloc[1]
    print(f"\n   Final model: {b.rmse - f_.rmse:.1f} cycles better RMSE for {f_.fit_s / b.fit_s:.0f}x the fit time, "
          f"{f_.predict_us_per_1k / b.predict_us_per_1k:.0f}x the prediction time and {f_.model_KB / b.model_KB:.0f}x the model size.")

    (out / "perf_summary.json").write_text(json.dumps({
        "rows": int(n), "features": int(p), "machine_GFLOPs_ceiling": round(peak, 2),
        "fit_MFLOP": round(ridge_fit_flops(n, p) / 1e6, 2), "predict_MFLOP": round(ridge_predict_flops(n, p) / 1e6, 2),
        "fit_intensity_FLOP_per_byte": round(fit_ai, 2), "predict_intensity_FLOP_per_byte": round(pred_ai, 3),
        "measured_fit_exponent_rows": round(float(slope), 2), "measured_fit_exponent_features": round(float(slope_p), 2),
        "synthetic_exponents": {"work_in_p": round(float(e_flops), 2), "rate_in_p": round(float(e_rate), 2),
                                "time_in_p": round(float(e_time), 2)},
        "note": "timings are machine-specific; test set not used"}, indent=1))
    print("\nWritten to results/perf_*.csv, perf_profile.txt and perf_summary.json. Test set not used.")


if __name__ == "__main__":
    main()
