"""The harmonised, cross-benchmark-comparable view of the metadata null.

**Why this module exists, separately from `benchmarks.py`.** The five
worked examples in `benchmarks.py` each use the benchmark's OWN published
split and metric, because that is what makes a verdict against that
benchmark's own leaderboard admissible. But that means they are **not
comparable to each other**: AirfRANS is scored there in Spearman, not R2;
DrivAerNet++'s number there zero-fills 595 test designs with no published
parameters at all; DrivAerML's official split is drag-sorted, which inflates
R2 relative to a random holdout. A single cross-benchmark table built from
those numbers would mix a rank correlation with variance ratios computed
under different inflation -- precisely the reporting error this project
exists to flag (``docs/paper/review/null_mechanism.md`` Sec 1). This module
is the answer: **one protocol, applied identically to all five benchmarks**.

Protocol (fixed, named here so it is machine-readable, not only prose):
10-fold out-of-sample OLS, seed 0, R2 (sklearn convention), constant
reference area where both conventions are published. Every harmonised CSV
under ``nullbench/data/harmonised/`` mirrors
``scripts/null_mechanism.py``'s own ``load_airfrans``/``load_simple``/
``load_drivaernet`` loaders byte-for-byte in row order and column set (built
by ``scripts/nullbench_build_data.py``'s ``build_harmonised_*`` functions),
so :func:`run_harmonised` reproduces ``results/review/null_mechanism.json``'s
``null_r2_linear`` to floating-point precision -- pinned in
``tests/test_nullbench.py``.

**No published comparison here, by design, and the JSON says so.** The
published entries in ``benchmarks.py`` were scored under each benchmark's
own protocol; comparing them against a harmonised-protocol null would
reintroduce the exact mixing error this module exists to avoid. Every
:class:`HarmonisedResult` carries ``published_comparison: None`` with a note
saying so, so a reader of the JSON alone (not just this docstring) sees the
reason.

**The flexible ceiling is sourced, not recomputed.** ``null_mechanism.md``
Sec 3 shows a linear null alone conflates two different failures -- AirfRANS
(0.6833) and AhmedML (0.6842) have statistically indistinguishable *linear*
nulls with opposite causes, separated only by how much higher a flexible
model can go (0.946 vs 0.775) -- and recommends reporting the pair. This
release reads that already-computed, already-verified flexible-ceiling
number from ``nullbench/data/harmonised/flexible_ceiling_sourced.json`` (a
small distillation of ``results/review/null_mechanism.json``, committed with
this package so it survives a ``pip install``) rather than reimplementing
the quadratic-expansion + ridge-path + kNN/local-linear model family that
produced it. That family is a genuinely larger unit of work than packaging
-- its own numerical-stability edge cases (near-collinear columns, as
WindsorML's `frontal_area` demonstrates) would need their own test suite --
and is declined for this release; see
``docs/paper/review/nullbench_release.md`` Sec 3 for the explicit scope call.
Every sourced value is field-labelled ``*_sourced`` so it is never mistaken
for something this harness computed, and a transcription-guard test asserts
it still matches ``null_mechanism.json`` verbatim.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from . import stats
from .fit import kfold_oos
from .io import Table, design_matrix, load_table

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "harmonised")

PROTOCOL = ("10-fold out-of-sample OLS, seed 0, R2 (sklearn convention), constant "
           "reference area where both conventions are published")
PROTOCOL_SOURCE = ("docs/paper/review/null_mechanism.md; scripts/null_mechanism.py "
                   "(pre-registration commit ffbec98)")

NO_COMPARISON_NOTE = (
    "published entries were scored on each benchmark's OWN protocol (see "
    "benchmarks.py); comparing them against a harmonised-protocol null would "
    "repeat the exact mixing error this artifact exists to flag, so none is "
    "attempted here"
)


@dataclass
class HarmonisedConfig:
    """One benchmark under the harmonised protocol.

    ``display_name`` must match a key in ``results/review/null_mechanism.json``
    / ``flexible_ceiling_sourced.json`` so the sourced flexible ceiling can be
    looked up; it is not necessarily the same string as the registry key.
    """

    name: str
    display_name: str
    csv_path: str  # relative to DATA_DIR
    id_col: str
    feature_cols: list[str]
    label_col: str
    flexible_ceiling_key: str  # target key inside flexible_ceiling_sourced.json
    metric: stats.Metric = "r2"
    folds: int = 10
    seed: int = 0
    note: str = ""

    def path(self) -> str:
        return os.path.join(DATA_DIR, self.csv_path)

    def load(self) -> Table:
        return load_table(self.path(), id_col=self.id_col, label_col=self.label_col,
                          feature_cols=self.feature_cols)


AIRFRANS_H = HarmonisedConfig(
    name="airfrans", display_name="AirfRANS",
    csv_path="airfrans.csv", id_col="case_id",
    feature_cols=["U", "alpha", "naca_0", "naca_1", "naca_2", "naca_3"],
    label_col="cd", flexible_ceiling_key="cd",
    metric="r2",  # harmonised protocol scores in R2, NOT Spearman -- see module docstring
    note=("Official full/test 200 cases (the cached official labels); NO alpha^2 term "
         "(unlike benchmarks.AIRFRANS) -- this mirrors null_mechanism.py's feature set "
         "exactly, not the worked-example one."),
)

AHMEDML_H = HarmonisedConfig(
    name="ahmedml", display_name="AhmedML",
    csv_path="ahmedml.csv", id_col="run_id",
    feature_cols=["body-length", "body-height", "body-width", "front-arc-diameter",
                 "slant-angle-length", "slant-angle-height", "slant-surface-length",
                 "slant-angle-degrees"],
    label_col="cd", flexible_ceiling_key="cd",
)

WINDSORML_H = HarmonisedConfig(
    name="windsorml", display_name="WindsorML",
    csv_path="windsorml.csv", id_col="run_id",
    feature_cols=["ratio_length_back_fast", "ratio_height_nose_windshield",
                 "ratio_height_fast_back", "side_taper", "clearance",
                 "bottom_taper_angle", "frontal_area"],
    label_col="cd", flexible_ceiling_key="cd",
    note="6 published parameters + 1 derived (frontal_area); see benchmarks.WINDSORML.",
)

DRIVAERML_H = HarmonisedConfig(
    name="drivaerml", display_name="DrivAerML",
    csv_path="drivaerml.csv", id_col="run_id",
    feature_cols=["Vehicle_Length", "Vehicle_Width", "Vehicle_Height", "Front_Overhang",
                 "Front_Planview", "Hood_Angle", "Approach_Angle", "Windscreen_Angle",
                 "Greenhouse_Tapering", "Backlight_Angle", "Decklid_Height",
                 "Rearend_tapering", "Rear_Overhang", "Rear_Diffusor_Angle",
                 "Vehicle_Ride_Height", "Vehicle_Pitch"],
    label_col="cd", flexible_ceiling_key="cd",
    note=("CONSTANT reference area (force_mom_constref_all.csv) and plain 10-fold OOS -- "
         "NOT the PhysicsNeMo drag-sorted 436/48 split benchmarks.DRIVAERML uses."),
)

DRIVAERNET_H = HarmonisedConfig(
    name="drivaernet", display_name="DrivAerNet++",
    csv_path="drivaernet.csv", id_col="design_id",
    # feature_cols intentionally omitted here; set by _drivaernet_feature_cols() below,
    # since the family-dummy column count depends on how many families exist in the
    # PARAMETRIC POOL (not the official split's families) -- see build_harmonised_drivaernet.
    feature_cols=[],
    label_col="cd", flexible_ceiling_key="cd",
    note=("PARAMETRIC POOL only (4165 designs with published parameters; excludes the "
         "595 test-split v1-fastbacks benchmarks.DRIVAERNET zero-fills), family dummies "
         "fit on ALL parametric designs (not train-only)."),
)


def _drivaernet_feature_cols() -> list[str]:
    """The DrivAerNet++ harmonised CSV's header depends on how many family levels
    exist in the parametric pool (read once, not hard-coded, so a CSV rebuild with a
    different pool size cannot silently desync from this list)."""
    path = DRIVAERNET_H.path()
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    return [c for c in header if c not in (DRIVAERNET_H.id_col, DRIVAERNET_H.label_col)]


REGISTRY: dict[str, HarmonisedConfig] = {
    "airfrans": AIRFRANS_H,
    "ahmedml": AHMEDML_H,
    "windsorml": WINDSORML_H,
    "drivaerml": DRIVAERML_H,
    "drivaernet": DRIVAERNET_H,
}


@dataclass
class HarmonisedResult:
    benchmark: str
    protocol: str
    metric: stats.Metric
    n_cases: int
    metadata_null_r2: float
    metadata_null_r2_ci95: tuple[float, float]
    metadata_null_mse: float
    metadata_null_mse_ci95: tuple[float, float]
    flexible_ceiling_r2_sourced: float | None
    flexible_ceiling_source: str
    flexible_ceiling_best_model_sourced: str | None
    linearity_share_sourced: float | None
    note: str
    published_comparison: None = field(default=None)
    published_comparison_note: str = NO_COMPARISON_NOTE

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "protocol": self.protocol,
            "metric": self.metric,
            "n_cases": self.n_cases,
            "metadata_null_r2": self.metadata_null_r2,
            "metadata_null_r2_ci95": list(self.metadata_null_r2_ci95),
            "metadata_null_mse": self.metadata_null_mse,
            "metadata_null_mse_ci95": list(self.metadata_null_mse_ci95),
            "flexible_ceiling_r2_sourced": self.flexible_ceiling_r2_sourced,
            "flexible_ceiling_source": self.flexible_ceiling_source,
            "flexible_ceiling_best_model_sourced": self.flexible_ceiling_best_model_sourced,
            "linearity_share_sourced": self.linearity_share_sourced,
            "note": self.note,
            "published_comparison": self.published_comparison,
            "published_comparison_note": self.published_comparison_note,
        }


def _load_sourced_flexible_ceiling() -> dict:
    path = os.path.join(DATA_DIR, "flexible_ceiling_sourced.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run_harmonised(cfg: HarmonisedConfig, n_boot: int = 10000) -> HarmonisedResult:
    """Run the harmonised metadata null for one benchmark. Computed by this
    harness (metadata_null_*); the flexible ceiling is read, not computed --
    see the module docstring."""
    feature_cols = cfg.feature_cols or (
        _drivaernet_feature_cols() if cfg is DRIVAERNET_H else None)
    table = load_table(cfg.path(), id_col=cfg.id_col, label_col=cfg.label_col,
                       feature_cols=feature_cols)
    Xd = design_matrix(table.X)
    pred = kfold_oos(Xd, table.y, k=cfg.folds, seed=cfg.seed)
    r2 = stats.r2_score(table.y, pred)
    r2_lo, r2_hi, _ = stats.bootstrap_ci(table.y, pred, "r2", n_boot=n_boot, seed=cfg.seed)
    mse = stats.mse_score(table.y, pred)
    mse_lo, mse_hi, _ = stats.bootstrap_ci_raw(table.y, pred, stats.mse_score,
                                               n_boot=n_boot, seed=cfg.seed)

    sourced = _load_sourced_flexible_ceiling()
    b = sourced["benchmarks"].get(cfg.display_name, {}).get(cfg.flexible_ceiling_key, {})

    return HarmonisedResult(
        benchmark=cfg.display_name,
        protocol=PROTOCOL,
        metric=cfg.metric,
        n_cases=table.n_cases,
        metadata_null_r2=r2,
        metadata_null_r2_ci95=(r2_lo, r2_hi),
        metadata_null_mse=mse,
        metadata_null_mse_ci95=(mse_lo, mse_hi),
        flexible_ceiling_r2_sourced=b.get("flexible_ceiling_r2_sourced"),
        flexible_ceiling_source=sourced["source"],
        flexible_ceiling_best_model_sourced=b.get("flexible_ceiling_best_model"),
        linearity_share_sourced=b.get("linearity_share_sourced"),
        note=cfg.note,
    )


def run_all_harmonised(n_boot: int = 10000) -> dict[str, HarmonisedResult]:
    return {name: run_harmonised(cfg, n_boot=n_boot) for name, cfg in REGISTRY.items()}
