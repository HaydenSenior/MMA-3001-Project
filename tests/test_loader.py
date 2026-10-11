"""Tests for cmapss.loader.

Most tests use tiny synthetic files written to a temp directory, so they run
without the real data. ``test_real_subset_*`` run only if the data directory
(env var ``CMAPSS_DATA_DIR``, default ``data``) contains the NASA files.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from cmapss.loader import (
    COLUMNS,
    EXPECTED_UNITS,
    SENSOR_COLUMNS,
    SENSOR_INFO,
    DataValidationError,
    add_rul,
    load_raw,
    load_rul,
    load_subset,
)


def write_engines(path, lengths, n_cols=26, trailing_space=True):
    """Write a synthetic C-MAPSS-style file: engine i has lengths[i-1] cycles."""
    lines = []
    for unit, n in enumerate(lengths, start=1):
        for cycle in range(1, n + 1):
            values = [unit, cycle] + [round(0.1 * cycle + j, 4) for j in range(n_cols - 2)]
            lines.append(" ".join(str(v) for v in values) + (" " if trailing_space else ""))
    Path(path).write_text("\n".join(lines) + "\n")


# ---------- constants ----------

def test_column_layout():
    assert len(COLUMNS) == 26
    assert len(SENSOR_COLUMNS) == 21
    assert set(SENSOR_INFO) == set(SENSOR_COLUMNS)


# ---------- load_raw ----------

def test_load_raw_parses_trailing_space_and_dtypes(tmp_path):
    f = tmp_path / "train.txt"
    write_engines(f, [3, 2])
    df = load_raw(f)
    assert df.shape == (5, 26)
    assert list(df.columns) == COLUMNS
    assert df["unit"].dtype == np.int64 and df["cycle"].dtype == np.int64
    assert df["unit"].tolist() == [1, 1, 1, 2, 2]
    assert df["cycle"].tolist() == [1, 2, 3, 1, 2]


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.txt")


def test_load_raw_empty_file(tmp_path):
    f = tmp_path / "e.txt"
    f.write_text("")
    with pytest.raises(DataValidationError, match="empty"):
        load_raw(f)


@pytest.mark.parametrize("n_cols", [25, 27])
def test_load_raw_wrong_column_count(tmp_path, n_cols):
    f = tmp_path / "w.txt"
    write_engines(f, [3], n_cols=n_cols)
    with pytest.raises(DataValidationError, match="columns"):
        load_raw(f)


def test_load_raw_non_numeric(tmp_path):
    f = tmp_path / "n.txt"
    write_engines(f, [3])
    f.write_text(f.read_text().replace("0.2", "abc", 1))
    with pytest.raises(DataValidationError, match="numeric"):
        load_raw(f)


def test_load_raw_nan_or_inf(tmp_path):
    f = tmp_path / "nan.txt"
    write_engines(f, [3])
    f.write_text(f.read_text().replace("0.2", "nan", 1))
    with pytest.raises(DataValidationError, match="NaN|infinite"):
        load_raw(f)


def test_load_raw_rejects_cycle_gap(tmp_path):
    f = tmp_path / "gap.txt"
    write_engines(f, [4])
    lines = f.read_text().splitlines()
    del lines[1]  # drop cycle 2
    f.write_text("\n".join(lines) + "\n")
    with pytest.raises(DataValidationError, match="cycles"):
        load_raw(f)


def test_load_raw_rejects_unit_gap(tmp_path):
    f = tmp_path / "ugap.txt"
    write_engines(f, [2, 2, 2])
    text = f.read_text().splitlines()
    kept = [ln for ln in text if not ln.startswith("2 ")]  # remove engine 2
    f.write_text("\n".join(kept) + "\n")
    with pytest.raises(DataValidationError, match="contiguous"):
        load_raw(f)


def test_load_raw_rejects_unsorted_units(tmp_path):
    f = tmp_path / "order.txt"
    write_engines(f, [2, 2])
    lines = f.read_text().splitlines()
    f.write_text("\n".join(lines[2:] + lines[:2]) + "\n")  # engine 2 rows first
    with pytest.raises(DataValidationError, match="start at 1|ascending"):
        load_raw(f)


# ---------- load_rul ----------

def test_load_rul_values_and_dtype(tmp_path):
    f = tmp_path / "rul.txt"
    f.write_text("112 \n98 \n69")  # trailing space and no final newline, as in real files
    rul = load_rul(f)
    assert rul.tolist() == [112, 98, 69]
    assert rul.dtype == np.int64


@pytest.mark.parametrize("content", ["5\n-3\n", "5\n2.5\n", "5\nabc\n", ""])
def test_load_rul_rejects_bad_values(tmp_path, content):
    f = tmp_path / "rul.txt"
    f.write_text(content)
    with pytest.raises(DataValidationError):
        load_rul(f)


# ---------- add_rul (hand-computed cases) ----------

def test_add_rul_hand_computed(tmp_path):
    f = tmp_path / "t.txt"
    write_engines(f, [4, 2])
    out = add_rul(load_raw(f))
    assert out["RUL"].tolist() == [3, 2, 1, 0, 1, 0]


def test_add_rul_cap_and_no_mutation(tmp_path):
    f = tmp_path / "t.txt"
    write_engines(f, [5])
    df = load_raw(f)
    out = add_rul(df, cap=2)
    assert out["RUL"].tolist() == [4, 3, 2, 1, 0]
    assert out["RUL_capped"].tolist() == [2, 2, 2, 1, 0]
    assert "RUL" not in df.columns  # input untouched


@pytest.mark.parametrize("cap", [0, -5, 12.5])
def test_add_rul_rejects_bad_cap(tmp_path, cap):
    f = tmp_path / "t.txt"
    write_engines(f, [3])
    with pytest.raises(ValueError):
        add_rul(load_raw(f), cap=cap)


# ---------- load_subset ----------

def make_subset(d, name, train_lengths, test_lengths, rul):
    write_engines(d / f"train_{name}.txt", train_lengths)
    write_engines(d / f"test_{name}.txt", test_lengths)
    (d / f"RUL_{name}.txt").write_text("\n".join(str(r) for r in rul) + "\n")


def test_load_subset_aligns_rul_to_test_units(tmp_path):
    make_subset(tmp_path, "FD001", [3, 4], [2, 3, 5], [10, 20, 30])
    data = load_subset("FD001", tmp_path, check_counts=False)
    assert data.test_rul.index.tolist() == [1, 2, 3]
    assert data.test_rul.tolist() == [10, 20, 30]
    assert data.train["unit"].nunique() == 2


def test_load_subset_rul_length_mismatch(tmp_path):
    make_subset(tmp_path, "FD001", [3], [2, 3], [10])
    with pytest.raises(DataValidationError, match="RUL values"):
        load_subset("FD001", tmp_path, check_counts=False)


def test_load_subset_count_check(tmp_path):
    make_subset(tmp_path, "FD001", [3], [2], [10])
    with pytest.raises(DataValidationError, match="expected"):
        load_subset("FD001", tmp_path)  # readme says 100/100


def test_load_subset_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="unknown subset"):
        load_subset("FD009", tmp_path)


def test_train_and_test_units_are_independent_namespaces(tmp_path):
    """Unit 1 of train and unit 1 of test are different engines (see module docstring)."""
    make_subset(tmp_path, "FD001", [10], [3], [7])
    data = load_subset("FD001", tmp_path, check_counts=False)
    assert len(data.train[data.train.unit == 1]) == 10
    assert len(data.test[data.test.unit == 1]) == 3


# ---------- real data (skipped if not present) ----------

DATA_DIR = Path(os.environ.get("CMAPSS_DATA_DIR", "data"))
HAVE_DATA = all((DATA_DIR / f"{k}_FD001.txt").exists() for k in ("train", "test", "RUL"))


@pytest.mark.skipif(not HAVE_DATA, reason="C-MAPSS files not found in CMAPSS_DATA_DIR")
@pytest.mark.parametrize("name", sorted(EXPECTED_UNITS))
def test_real_subset_loads_and_matches_known_counts(name):
    data = load_subset(name, DATA_DIR)
    assert (data.train["unit"].nunique(), data.test["unit"].nunique()) == EXPECTED_UNITS[name]
    assert (data.test_rul >= 0).all()
    # Every train engine must end at RUL 0 after labelling.
    assert (add_rul(data.train).groupby("unit")["RUL"].min() == 0).all()


@pytest.mark.skipif(not HAVE_DATA, reason="C-MAPSS files not found in CMAPSS_DATA_DIR")
def test_real_fd001_first_rul_matches_published_value():
    # RUL_FD001.txt line 1 is 112 (engine 1), line 100 is 20.
    data = load_subset("FD001", DATA_DIR)
    assert data.test_rul.loc[1] == 112 and data.test_rul.loc[100] == 20
