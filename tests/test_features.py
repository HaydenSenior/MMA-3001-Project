"""Tests for cmapss.features (column screening and degradation onset).

Synthetic engines with known structure are used, so expected answers are
known by construction. One regression test runs on the real FD001 file if
present (CMAPSS_DATA_DIR, default ``data``).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cmapss.features import (
    constant_columns,
    degradation_onset,
    select_columns,
    trend_strength,
)


def make_engines(lengths, onset=None, slope=0.5, noise=0.2, seed=0):
    """Engines with: 'flat' (constant), 'noise' (no trend), 'wear' (ramps after onset),
    'fall' (ramps downward after onset). ``onset`` is a cycle number or None."""
    rng = np.random.default_rng(seed)
    frames = []
    for unit, n in enumerate(lengths, start=1):
        cycle = np.arange(1, n + 1)
        ramp = np.zeros(n) if onset is None else slope * np.clip(cycle - onset, 0, None)
        frames.append(
            pd.DataFrame(
                {
                    "unit": unit,
                    "cycle": cycle,
                    "flat": 5.0,
                    "noise": rng.normal(0, 1, n),
                    "wear": 10 + ramp + rng.normal(0, noise, n),
                    "fall": 20 - ramp + rng.normal(0, noise, n),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# ---------- constant_columns ----------

def test_constant_columns_finds_only_zero_variance():
    df = make_engines([50, 50], onset=20)
    assert constant_columns(df, ["flat", "noise", "wear"]) == ["flat"]


def test_constant_column_changes_in_one_engine_is_not_constant():
    df = make_engines([30, 30], onset=None)
    df.loc[df["unit"] == 2, "flat"] = 6.0   # constant per engine, different across engines
    assert constant_columns(df, ["flat"]) == []


# ---------- trend_strength ----------

def test_trend_strength_hand_checked_extremes():
    n = 40
    df = pd.DataFrame(
        {
            "unit": np.repeat([1, 2], n),
            "cycle": np.tile(np.arange(1, n + 1), 2),
            "up": np.tile(np.arange(n, dtype=float), 2),          # perfectly increasing
            "down": np.tile(-np.arange(n, dtype=float), 2),       # perfectly decreasing
            "flat": 3.0,
        }
    )
    ts = trend_strength(df, ["up", "down", "flat"])
    assert ts["up"] == pytest.approx(1.0)
    assert ts["down"] == pytest.approx(1.0)   # absolute value: direction does not matter
    assert ts["flat"] == 0.0


def test_trend_strength_noise_is_low_and_wear_is_high():
    df = make_engines([200] * 10, onset=50)
    ts = trend_strength(df, ["noise", "wear"])
    assert ts["noise"] < 0.2 < 0.6 < ts["wear"]


# ---------- select_columns ----------

def test_select_columns_keeps_trending_drops_flat_and_noise():
    df = make_engines([200] * 10, onset=50)
    kept, report = select_columns(df, ["flat", "noise", "wear", "fall"])
    assert kept == ["wear", "fall"]
    assert report.loc["flat", "reason"].startswith("constant")
    assert report.loc["noise", "reason"].startswith("no consistent trend")
    assert report["kept"].tolist() == [False, False, True, True]


def test_select_columns_threshold_is_respected():
    df = make_engines([200] * 10, onset=50)
    kept_loose, _ = select_columns(df, ["noise", "wear"], min_trend=0.0)
    kept_strict, _ = select_columns(df, ["noise", "wear"], min_trend=0.99)
    assert "noise" in kept_loose
    assert "wear" not in kept_strict


# ---------- degradation_onset ----------

def test_onset_recovers_known_onset_and_both_directions():
    df = make_engines([220] * 8, onset=100, slope=0.5, noise=0.2)
    out = degradation_onset(df, ["wear", "fall"])      # one column rises, one falls
    assert out["life"].tolist() == [220] * 8
    # Smoothing/persistence delay detection a little, never before the true onset.
    assert ((out["onset_cycle"] >= 100) & (out["onset_cycle"] <= 115)).all()
    assert (out["rul_at_onset"] == out["life"] - out["onset_cycle"]).all()


def test_onset_is_nan_for_engine_that_never_degrades():
    # Needs a degrading engine in the frame so the column signs are defined.
    degrading = make_engines([200] * 5, onset=80)
    healthy = make_engines([200], onset=None, seed=5)
    healthy["unit"] = 99
    out = degradation_onset(pd.concat([degrading, healthy], ignore_index=True), ["wear"])
    assert np.isnan(out.loc[99, "onset_cycle"])
    assert out.drop(index=99)["onset_cycle"].notna().all()


def test_onset_nan_when_engine_too_short():
    df = make_engines([25, 200, 200], onset=80)
    out = degradation_onset(df, ["wear"], baseline=30, persist=10)
    assert np.isnan(out.loc[1, "onset_cycle"]) and out.loc[2:, "onset_cycle"].notna().all()


def test_onset_later_true_onset_gives_later_estimate():
    early = degradation_onset(make_engines([250] * 5, onset=60), ["wear"])["onset_cycle"].median()
    late = degradation_onset(make_engines([250] * 5, onset=150), ["wear"])["onset_cycle"].median()
    assert late - early > 60


@pytest.mark.parametrize("kwargs", [{"baseline": 0}, {"window": 0}, {"persist": 0}])
def test_onset_rejects_nonpositive_settings(kwargs):
    with pytest.raises(ValueError):
        degradation_onset(make_engines([100, 100], onset=40), ["wear"], **kwargs)


# ---------- real data (skipped if not present) ----------

DATA_DIR = Path(os.environ.get("CMAPSS_DATA_DIR", "data"))
HAVE_DATA = (DATA_DIR / "train_FD001.txt").exists()


@pytest.mark.skipif(not HAVE_DATA, reason="train_FD001.txt not found in CMAPSS_DATA_DIR")
def test_real_fd001_screening_regression():
    """Regression test: records the screening result found in the EDA notebook."""
    from cmapss.loader import SENSOR_COLUMNS, SETTING_COLUMNS, load_raw

    df = load_raw(DATA_DIR / "train_FD001.txt")
    kept, _ = select_columns(df, SETTING_COLUMNS + SENSOR_COLUMNS, min_trend=0.3)
    assert kept == ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
