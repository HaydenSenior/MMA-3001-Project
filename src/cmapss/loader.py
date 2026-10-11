"""Loading and validation of the NASA C-MAPSS turbofan degradation data.

The data come as plain-text files with 26 whitespace-separated numeric columns
and no header. Each row is one operating cycle (one flight) of one engine::

    unit, cycle, op1, op2, op3, s1 ... s21

Per subset (FD001 to FD004) there are three files:

* ``train_FDxxx.txt``: engines run to failure.
* ``test_FDxxx.txt``: engines cut off some time before failure.
* ``RUL_FDxxx.txt``: one true remaining-useful-life value per test engine, in
  the order the engines first appear in the test file.

Unit numbers restart at 1 in every file, so they identify an engine only
*within* one file. Never join train and test data on ``unit``.

Example:
    >>> data = load_subset("FD001", "data/")           # doctest: +SKIP
    >>> train = add_rul(data.train, cap=125)           # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

N_SENSORS = 21
SETTING_COLUMNS = ["op1", "op2", "op3"]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, N_SENSORS + 1)]
COLUMNS = ["unit", "cycle"] + SETTING_COLUMNS + SENSOR_COLUMNS

# Sensor symbols, units and meaning from Table 2 of Saxena et al. (2008).
# The mapping s1..s21 -> Table 2 order is the usual convention for this data;
# the dataset readme does not state it. Magnitudes in FD001 are consistent
# with it (e.g. s1 ~ 518.67 degR fan-inlet temperature, s8 ~ 2388 rpm fan speed).
SENSOR_INFO = {
    "s1": ("T2", "degR", "Total temperature at fan inlet"),
    "s2": ("T24", "degR", "Total temperature at LPC outlet"),
    "s3": ("T30", "degR", "Total temperature at HPC outlet"),
    "s4": ("T50", "degR", "Total temperature at LPT outlet"),
    "s5": ("P2", "psia", "Pressure at fan inlet"),
    "s6": ("P15", "psia", "Total pressure in bypass duct"),
    "s7": ("P30", "psia", "Total pressure at HPC outlet"),
    "s8": ("Nf", "rpm", "Physical fan speed"),
    "s9": ("Nc", "rpm", "Physical core speed"),
    "s10": ("epr", "-", "Engine pressure ratio (P50/P2)"),
    "s11": ("Ps30", "psia", "Static pressure at HPC outlet"),
    "s12": ("phi", "pps/psi", "Ratio of fuel flow to Ps30"),
    "s13": ("NRf", "rpm", "Corrected fan speed"),
    "s14": ("NRc", "rpm", "Corrected core speed"),
    "s15": ("BPR", "-", "Bypass ratio"),
    "s16": ("farB", "-", "Burner fuel-air ratio"),
    "s17": ("htBleed", "-", "Bleed enthalpy"),
    "s18": ("Nf_dmd", "rpm", "Demanded fan speed"),
    "s19": ("PCNfR_dmd", "rpm", "Demanded corrected fan speed"),
    "s20": ("W31", "lbm/s", "HPT coolant bleed"),
    "s21": ("W32", "lbm/s", "LPT coolant bleed"),
}

# Engine counts (train, test) per subset. These follow the dataset readme except
# FD004, where the readme lists 248 train / 249 test but the files (and the 248
# values in RUL_FD004.txt) contain 249 train / 248 test engines, so the readme
# appears to have the two numbers swapped. Checked against the files, not the readme.
EXPECTED_UNITS = {
    "FD001": (100, 100),
    "FD002": (260, 259),
    "FD003": (100, 100),
    "FD004": (249, 248),
}


class DataValidationError(ValueError):
    """Raised when a C-MAPSS file does not have the structure we rely on."""


@dataclass(frozen=True)
class CMAPSSData:
    """One C-MAPSS subset.

    Attributes:
        name: Subset name, e.g. ``"FD001"``.
        train: Run-to-failure cycles, columns as in ``COLUMNS``.
        test: Truncated cycles, columns as in ``COLUMNS``.
        test_rul: True RUL in cycles after each test engine's last cycle,
            indexed by test ``unit`` (int64, >= 0).
    """

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.Series


def load_raw(path: PathLike) -> pd.DataFrame:
    """Read one ``train_*.txt`` or ``test_*.txt`` file and validate it.

    Args:
        path: Path to the whitespace-separated text file.

    Returns:
        DataFrame with columns ``COLUMNS``. ``unit`` and ``cycle`` are int64,
        all other columns float64. Rows are in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        DataValidationError: If the file is empty, does not have 26 numeric
            columns, contains NaN/inf, or if units are not numbered 1..N in
            order, or if any unit's cycles are not exactly 1..n_cycles.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"C-MAPSS file not found: {path}")

    try:
        # \s+ also absorbs the trailing space on every line of the raw files.
        raw = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"{path.name}: file is empty") from exc
    except ValueError as exc:  # non-numeric token or ragged rows
        raise DataValidationError(f"{path.name}: cannot parse as numeric table ({exc})") from exc

    if raw.shape[1] != len(COLUMNS):
        raise DataValidationError(
            f"{path.name}: expected {len(COLUMNS)} columns, found {raw.shape[1]}"
        )
    raw.columns = COLUMNS

    if not np.isfinite(raw.to_numpy()).all():
        raise DataValidationError(f"{path.name}: contains NaN or infinite values")

    for col in ("unit", "cycle"):
        if not (raw[col] == raw[col].round()).all():
            raise DataValidationError(f"{path.name}: column '{col}' has non-integer values")
        raw[col] = raw[col].astype(np.int64)

    _check_unit_and_cycle_structure(raw, path.name)
    return raw


def _check_unit_and_cycle_structure(df: pd.DataFrame, label: str) -> None:
    """Check units are 1..N in file order and each unit's cycles are 1..n."""
    unit = df["unit"].to_numpy()
    if unit[0] != 1 or (np.diff(unit) < 0).any():
        raise DataValidationError(f"{label}: units must start at 1 and be grouped in ascending order")
    unique_units = np.unique(unit)
    if not np.array_equal(unique_units, np.arange(1, len(unique_units) + 1)):
        raise DataValidationError(f"{label}: unit numbers are not contiguous 1..N")

    # Within a unit the cycle must step by exactly 1 starting at 1.
    expected_cycle = df.groupby("unit").cumcount().to_numpy() + 1
    if not np.array_equal(df["cycle"].to_numpy(), expected_cycle):
        bad = int(df.loc[df["cycle"].to_numpy() != expected_cycle, "unit"].iloc[0])
        raise DataValidationError(
            f"{label}: unit {bad} has missing, repeated or out-of-order cycles"
        )


def load_rul(path: PathLike) -> np.ndarray:
    """Read a ``RUL_*.txt`` file.

    Args:
        path: Path to the file with one RUL value per line.

    Returns:
        1-D int64 array. Element ``i`` is the true RUL (cycles) of test engine
        ``i + 1``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        DataValidationError: If the file is empty, non-numeric, non-integer
            or contains negative values.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"RUL file not found: {path}")
    try:
        values = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"{path.name}: file is empty") from exc
    except ValueError as exc:
        raise DataValidationError(f"{path.name}: cannot parse as numeric ({exc})") from exc

    if values.shape[1] != 1:
        raise DataValidationError(f"{path.name}: expected 1 column, found {values.shape[1]}")
    arr = values.iloc[:, 0].to_numpy()
    if not np.isfinite(arr).all() or (arr != np.round(arr)).any() or (arr < 0).any():
        raise DataValidationError(f"{path.name}: RUL values must be finite non-negative integers")
    return arr.astype(np.int64)


def add_rul(df: pd.DataFrame, cap: Optional[int] = None) -> pd.DataFrame:
    """Add run-to-failure RUL labels to a *training* frame.

    ``RUL = (last cycle of the unit) - cycle``, so the final row of every
    unit has RUL 0. This is only valid for run-to-failure data; test engines
    are truncated before failure, so their last cycle is not the failure cycle.

    Args:
        df: Training data as returned by :func:`load_raw`.
        cap: If given, also add ``RUL_capped = min(RUL, cap)``, the
            piecewise-linear target often used because early-life
            degradation is not visible in the sensors. Must be a positive int.

    Returns:
        A copy of ``df`` with ``RUL`` (and optionally ``RUL_capped``) columns.

    Raises:
        ValueError: If ``cap`` is not a positive integer.
    """
    if cap is not None and (int(cap) != cap or cap <= 0):
        raise ValueError(f"cap must be a positive integer, got {cap!r}")
    out = df.copy()
    out["RUL"] = out.groupby("unit")["cycle"].transform("max") - out["cycle"]
    if cap is not None:
        out["RUL_capped"] = out["RUL"].clip(upper=int(cap))
    return out


def load_subset(name: str, data_dir: PathLike, check_counts: bool = True) -> CMAPSSData:
    """Load train, test and true-RUL files of one subset.

    Args:
        name: One of ``"FD001"`` to ``"FD004"``.
        data_dir: Directory containing ``train_<name>.txt``,
            ``test_<name>.txt`` and ``RUL_<name>.txt``.
        check_counts: If True, check engine counts against ``EXPECTED_UNITS``.

    Returns:
        A :class:`CMAPSSData` with ``test_rul`` indexed by test unit.

    Raises:
        ValueError: If ``name`` is not a known subset.
        FileNotFoundError: If any of the three files is missing.
        DataValidationError: If a file is malformed, the number of RUL values
            differs from the number of test engines, any true RUL is
            inconsistent with the file, or (when ``check_counts``) engine
            counts differ from the readme.
    """
    if name not in EXPECTED_UNITS:
        raise ValueError(f"unknown subset {name!r}; expected one of {sorted(EXPECTED_UNITS)}")
    data_dir = Path(data_dir)

    train = load_raw(data_dir / f"train_{name}.txt")
    test = load_raw(data_dir / f"test_{name}.txt")
    rul = load_rul(data_dir / f"RUL_{name}.txt")

    n_train, n_test = train["unit"].nunique(), test["unit"].nunique()
    if len(rul) != n_test:
        raise DataValidationError(
            f"{name}: {len(rul)} RUL values but {n_test} test engines"
        )
    if check_counts and (n_train, n_test) != EXPECTED_UNITS[name]:
        raise DataValidationError(
            f"{name}: found {n_train} train / {n_test} test engines, "
            f"expected {EXPECTED_UNITS[name]}"
        )

    test_rul = pd.Series(rul, index=pd.Index(np.arange(1, n_test + 1), name="unit"), name="RUL")
    return CMAPSSData(name=name, train=train, test=test, test_rul=test_rul)
