"""Tests for cmapss.splits, centred on the no-leakage guarantee.

Synthetic frames are used so the tests run without the real data; one test
runs on the real FD001 training file if it is present (CMAPSS_DATA_DIR,
default ``data``).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cmapss.splits import LeakageError, assert_no_unit_overlap, unit_kfold_splits


def make_frame(lengths):
    """Frame with a unit and cycle column; engine i has lengths[i-1] cycles."""
    units = np.repeat(np.arange(1, len(lengths) + 1), lengths)
    cycles = np.concatenate([np.arange(1, n + 1) for n in lengths])
    return pd.DataFrame({"unit": units, "cycle": cycles})


@pytest.fixture
def frame():
    rng = np.random.default_rng(1)
    return make_frame(rng.integers(120, 360, size=23))  # uneven lengths, 23 engines


# ---------- the leakage guarantee ----------

@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("n_splits", [2, 5, 23])
def test_no_unit_in_both_train_and_validation(frame, n_splits, shuffle):
    for train_idx, val_idx in unit_kfold_splits(frame, n_splits=n_splits, shuffle=shuffle, seed=3):
        train_units = set(frame["unit"].iloc[train_idx])
        val_units = set(frame["unit"].iloc[val_idx])
        assert train_units.isdisjoint(val_units)


def test_leakage_check_detects_a_row_level_split(frame):
    """The check must be able to fail: a naive row split of the same data leaks."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(frame))
    train_idx, val_idx = np.sort(idx[: len(idx) // 2]), np.sort(idx[len(idx) // 2:])
    with pytest.raises(LeakageError):
        assert_no_unit_overlap(frame, train_idx, val_idx)


def test_leakage_check_detects_single_shared_engine():
    df = make_frame([5, 5, 5])
    train_idx = np.arange(0, 10)          # engines 1 and 2
    val_idx = np.arange(9, 15)            # last row of engine 2, plus engine 3
    with pytest.raises(LeakageError, match="unit 2"):
        assert_no_unit_overlap(df, train_idx, val_idx)


def test_leakage_check_passes_on_clean_split():
    df = make_frame([5, 5, 5])
    assert_no_unit_overlap(df, np.arange(0, 10), np.arange(10, 15))


# ---------- split properties ----------

def test_every_row_is_validated_exactly_once(frame):
    folds = unit_kfold_splits(frame, n_splits=5)
    all_val = np.concatenate([v for _, v in folds])
    assert np.array_equal(np.sort(all_val), np.arange(len(frame)))


def test_train_and_validation_partition_each_fold(frame):
    for train_idx, val_idx in unit_kfold_splits(frame, n_splits=4):
        assert len(train_idx) + len(val_idx) == len(frame)
        assert np.intersect1d(train_idx, val_idx).size == 0


def test_whole_engines_stay_together(frame):
    """Every row of a validation engine is in validation, none in train."""
    counts = frame.groupby("unit").size()
    for _, val_idx in unit_kfold_splits(frame, n_splits=5):
        val_counts = frame.iloc[val_idx].groupby("unit").size()
        assert (val_counts == counts.loc[val_counts.index]).all()


def test_fold_engine_counts_differ_by_at_most_one(frame):
    sizes = [frame["unit"].iloc[v].nunique() for _, v in unit_kfold_splits(frame, n_splits=5)]
    assert sum(sizes) == 23 and max(sizes) - min(sizes) <= 1


def test_unshuffled_folds_hold_consecutive_units():
    df = make_frame([3] * 6)
    folds = unit_kfold_splits(df, n_splits=3, shuffle=False)
    assert [sorted(set(df["unit"].iloc[v])) for _, v in folds] == [[1, 2], [3, 4], [5, 6]]


def test_same_seed_reproducible_different_seed_differs(frame):
    a = unit_kfold_splits(frame, n_splits=5, seed=7)
    b = unit_kfold_splits(frame, n_splits=5, seed=7)
    c = unit_kfold_splits(frame, n_splits=5, seed=8)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    assert any(not np.array_equal(x[1], y[1]) for x, y in zip(a, c))


def test_indices_are_sorted_row_positions(frame):
    for train_idx, val_idx in unit_kfold_splits(frame, n_splits=5):
        assert (np.diff(train_idx) > 0).all() and (np.diff(val_idx) > 0).all()


# ---------- input validation ----------

@pytest.mark.parametrize("n_splits", [0, 1, 24])
def test_bad_n_splits(frame, n_splits):
    with pytest.raises(ValueError, match="n_splits"):
        unit_kfold_splits(frame, n_splits=n_splits)


def test_missing_unit_column():
    with pytest.raises(ValueError, match="unit"):
        unit_kfold_splits(pd.DataFrame({"cycle": [1, 2, 3]}))
    with pytest.raises(ValueError, match="unit"):
        assert_no_unit_overlap(pd.DataFrame({"cycle": [1, 2, 3]}), np.array([0]), np.array([1]))


# ---------- real data (skipped if not present) ----------

DATA_DIR = Path(os.environ.get("CMAPSS_DATA_DIR", "data"))
HAVE_DATA = (DATA_DIR / "train_FD001.txt").exists()


@pytest.mark.skipif(not HAVE_DATA, reason="train_FD001.txt not found in CMAPSS_DATA_DIR")
def test_real_fd001_five_fold_has_20_engines_per_fold_and_no_leakage():
    from cmapss.loader import load_raw

    df = load_raw(DATA_DIR / "train_FD001.txt")
    folds = unit_kfold_splits(df, n_splits=5, seed=0)
    for train_idx, val_idx in folds:
        assert df["unit"].iloc[val_idx].nunique() == 20
        assert df["unit"].iloc[train_idx].nunique() == 80
        assert_no_unit_overlap(df, train_idx, val_idx)
