"""Load a plain CSV of ``(case id, parameters..., label[, split])`` into arrays.

This is the entire data contract NullBench asks of a benchmark maintainer: one
CSV, headers, one row per case. No mesh, no field, no point cloud -- nothing
this module reads can depend on simulation output, which is the whole reason
the covariate null is cheap to compute. Uses the standard library ``csv``
module and numpy only, matching the rest of the repository's tabular-data
convention (``scripts/covariate_null_crossbench.py``); no pandas dependency.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Table:
    """A loaded covariate-null input table.

    Attributes
    ----------
    ids : list[str]
        Case identifiers, row order preserved from the source CSV.
    feature_names : list[str]
        Names of the parameter columns, in the order they appear in ``X``.
    X : numpy.ndarray, shape (n, p)
        Raw parameter matrix (no intercept column -- ``design_matrix`` adds one).
    y : numpy.ndarray, shape (n,)
        The label column.
    split : numpy.ndarray[str] | None
        Per-case split tag (e.g. ``"train"``/``"test"``), or ``None`` if the
        CSV carried no split column, in which case K-fold is the only
        available protocol.
    source : str
        Path the table was read from, kept for provenance in output records.
    """

    ids: list[str]
    feature_names: list[str]
    X: np.ndarray
    y: np.ndarray
    split: np.ndarray | None
    source: str

    @property
    def n_cases(self) -> int:
        return len(self.ids)

    def has_split(self) -> bool:
        return self.split is not None

    def split_mask(self, values: tuple[str, ...]) -> np.ndarray:
        if self.split is None:
            raise ValueError("this table has no split column")
        return np.isin(self.split, list(values))


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    # strip stray whitespace in headers/values, a recurring trait of these
    # benchmarks' published CSVs (see scripts/covariate_null_crossbench.py)
    clean = []
    for r in rows:
        clean.append({(k.strip() if k else k): (v.strip() if isinstance(v, str) else v)
                      for k, v in r.items()})
    return clean


def load_table(
    path: str | Path,
    id_col: str,
    label_col: str,
    feature_cols: list[str] | None = None,
    split_col: str | None = None,
) -> Table:
    """Load ``path`` into a :class:`Table`.

    Parameters
    ----------
    id_col, label_col : str
        Column names for the case id and the label to be predicted.
    feature_cols : list[str] | None
        Parameter columns to use. If ``None``, every column except
        ``id_col``, ``label_col`` and ``split_col`` is used, in file order
        (this is the "use everything published" default).
    split_col : str | None
        Column carrying a split tag. If absent from the CSV or not given,
        the table has no split and only K-fold is available.
    """
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"{path}: no rows")
    header = list(rows[0].keys())
    if id_col not in header:
        raise ValueError(f"{path}: id column {id_col!r} not found; have {header}")
    if label_col not in header:
        raise ValueError(f"{path}: label column {label_col!r} not found; have {header}")
    if feature_cols is None:
        skip = {id_col, label_col}
        if split_col:
            skip.add(split_col)
        feature_cols = [c for c in header if c not in skip]
    missing = [c for c in feature_cols if c not in header]
    if missing:
        raise ValueError(f"{path}: feature column(s) {missing} not found; have {header}")
    has_split = bool(split_col) and split_col in header

    ids = [r[id_col] for r in rows]
    X = np.array([[float(r[c]) for c in feature_cols] for r in rows], dtype=float)
    y = np.array([float(r[label_col]) for r in rows], dtype=float)
    split = np.array([r[split_col] for r in rows]) if has_split else None

    return Table(ids=ids, feature_names=list(feature_cols), X=X, y=y, split=split,
                 source=str(path))


def design_matrix(X: np.ndarray) -> np.ndarray:
    """Prepend an intercept column of ones. Every fit in NullBench uses this
    -- the intercept-only model (X sliced to zero feature columns) is the
    metric floor by construction (see ``stats.py``)."""
    n = X.shape[0]
    return np.hstack([np.ones((n, 1)), X])


def intercept_only(n: int) -> np.ndarray:
    return np.ones((n, 1))
