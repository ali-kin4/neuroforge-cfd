"""Fitting and split protocols for the covariate null.

The null model is deliberately the cheapest thing that can use the published
metadata: ordinary least squares, with an intercept column supplied by the
caller (``io.build_design`` always prepends one). No regularisation, no
hyperparameters, nothing that could be tuned against the test set.

Two out-of-sample protocols, matching ``scripts/covariate_null*.py``:

* ``official_split``: fit on a caller-supplied train set, score on a
  caller-supplied test set. Used whenever the benchmark publishes a split.
* ``kfold_oos``: fit is repeated K times, each time excluding one fold, so
  every prediction comes from a model that never saw that case. Used when no
  canonical split exists. Every output record states which protocol ran.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    """Ordinary least squares, fit on ``(X_train, y_train)``, applied to ``X_eval``."""
    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    return X_eval @ beta


def kfold_oos(X: np.ndarray, y: np.ndarray, k: int = 10, seed: int = 0) -> np.ndarray:
    """Out-of-sample predictions by K-fold OLS. Every prediction is made by a
    fold that excluded its own case."""
    n = len(y)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, k)
    pred = np.empty(n)
    for f in folds:
        mask = np.ones(n, dtype=bool)
        mask[f] = False
        pred[f] = fit_predict(X[mask], y[mask], X[f])
    return pred


@dataclass
class SplitResult:
    protocol: str  # "official_split" | "kfold_oos_k{k}"
    n_fit: int
    n_score: int
    y_true: np.ndarray            # labels the OOS predictions are scored against
    y_pred_oos: np.ndarray        # out-of-sample predictions, same order as y_true
    y_true_in_sample: np.ndarray  # labels the in-sample predictions are scored against
    y_pred_in_sample: np.ndarray


def run_official_split(X_train, y_train, X_test, y_test) -> SplitResult:
    pred_test = fit_predict(X_train, y_train, X_test)
    pred_train = fit_predict(X_train, y_train, X_train)
    return SplitResult(
        protocol="official_split",
        n_fit=int(len(y_train)),
        n_score=int(len(y_test)),
        y_true=np.asarray(y_test, dtype=float),
        y_pred_oos=pred_test,
        y_true_in_sample=np.asarray(y_train, dtype=float),
        y_pred_in_sample=pred_train,
    )


def run_kfold(X, y, k: int = 10, seed: int = 0) -> SplitResult:
    oos = kfold_oos(X, y, k=k, seed=seed)
    ins = fit_predict(X, y, X)
    y = np.asarray(y, dtype=float)
    return SplitResult(
        protocol=f"kfold_oos_k{k}",
        n_fit=int(len(y)),
        n_score=int(len(y)),
        y_true=y,
        y_pred_oos=oos,
        y_true_in_sample=y,
        y_pred_in_sample=ins,
    )
