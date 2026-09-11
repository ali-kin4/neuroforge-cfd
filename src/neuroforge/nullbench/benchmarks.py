"""The five worked-example benchmark configurations.

Every published entry below is copied from a source already read at the
primary document (proceedings PDF or the paper's own full text) and recorded
in ``docs/paper/review/published_baselines_verified.md`` (AirfRANS) or
``docs/paper/review/null_travels.md`` (DrivAerML, DrivAerNet++, AhmedML,
WindsorML). No entry here was invented or transcribed from a secondary
source; where no verified published baseline exists (AhmedML, and every row
but one on WindsorML) that is stated rather than filled in.

Each config's CSV under ``nullbench/data/<name>/`` was distilled by
``scripts/nullbench_build_data.py`` from committed or locally-cached
per-case metadata -- see that script's docstring for provenance -- and its
feature set is the *richest* metadata block ``null_travels.md`` reports for
that benchmark, so ``run_benchmark`` reproduces the headline number in
``results/review/covariate_null*.json`` (within bootstrap-seed tolerance;
see ``tests/test_nullbench.py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .harness import NullResult, PublishedEntry, run_null
from .io import Table, load_table
from .stats import Metric

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@dataclass
class Target:
    """One label column within a benchmark, with its own metric and published table."""

    label_col: str
    metric: Metric
    published: list[PublishedEntry] = field(default_factory=list)
    note: str = ""


@dataclass
class BenchmarkConfig:
    name: str
    csv_path: str  # relative to DATA_DIR
    id_col: str
    targets: dict[str, Target]
    split_col: str | None = None
    train_values: tuple[str, ...] = ("train",)
    test_values: tuple[str, ...] = ("test",)
    feature_cols: list[str] | None = None
    folds: int = 10
    seed: int = 0
    description: str = ""
    verified_in: str = ""

    def path(self) -> str:
        return os.path.join(DATA_DIR, self.csv_path)

    def load(self, target: str) -> Table:
        if target not in self.targets:
            raise KeyError(f"{self.name}: unknown target {target!r}; have {list(self.targets)}")
        return load_table(self.path(), id_col=self.id_col,
                          label_col=self.targets[target].label_col,
                          feature_cols=self.feature_cols, split_col=self.split_col)


def run_benchmark(cfg: BenchmarkConfig, target: str, n_boot: int = 10000,
                  seed: int | None = None) -> NullResult:
    tgt = cfg.targets[target]
    table = cfg.load(target)
    return run_null(table, metric=tgt.metric, train_values=cfg.train_values,
                    test_values=cfg.test_values, folds=cfg.folds, n_boot=n_boot,
                    seed=cfg.seed if seed is None else seed, published=tgt.published)


# --------------------------------------------------------------------------------------
# 1. AirfRANS -- Bonnet, Mazari, Cinnella, Gallinari, NeurIPS 2022 D&B (arXiv 2212.07564),
#    Table 3. Official 800/200 `full`-task split. Verified at source 2026-09-11; see
#    docs/paper/review/published_baselines_verified.md.
# --------------------------------------------------------------------------------------
AIRFRANS_PARAM_COLS = ["U", "alpha", "alpha2", "naca_0", "naca_1", "naca_2", "naca_3"]

AIRFRANS = BenchmarkConfig(
    name="AirfRANS",
    csv_path=os.path.join("airfrans", "airfrans_full_800_200.csv"),
    id_col="case_id",
    split_col="split",
    train_values=("train",),
    test_values=("test",),
    feature_cols=AIRFRANS_PARAM_COLS,  # explicit: the CSV also carries the OTHER target (cl/cd)
    description=("2-D RANS airfoils, official `full` task. Parameters: inlet speed U, "
                 "angle of attack alpha (+ alpha^2), and the NACA digits parsed from the "
                 "case file name -- exactly the geometry a surrogate is given as input."),
    verified_in="docs/paper/review/published_baselines_verified.md",
    targets={
        "cl": Target("cl", "spearman", [
            PublishedEntry("MLP", 0.913, 0.018, "NeurIPS 2022 D&B Table 3"),
            PublishedEntry("GraphSAGE", 0.965, 0.011, "NeurIPS 2022 D&B Table 3"),
            PublishedEntry("PointNet", 0.938, 0.023, "NeurIPS 2022 D&B Table 3"),
            PublishedEntry("Graph U-Net", 0.967, 0.019, "NeurIPS 2022 D&B Table 3"),
            PublishedEntry("Transolver", 0.9978, None,
                          "Wu et al. ICML 2024, arXiv 2402.02366",
                          note="no drag column reported; lift only"),
        ]),
        "cd": Target("cd", "spearman", [
            PublishedEntry("MLP", -0.117, 0.256, "NeurIPS 2022 D&B Table 3"),
            PublishedEntry("GraphSAGE", -0.303, 0.124, "NeurIPS 2022 D&B Table 3"),
            PublishedEntry("PointNet", -0.022, 0.097, "NeurIPS 2022 D&B Table 3"),
            PublishedEntry("Graph U-Net", -0.138, 0.258, "NeurIPS 2022 D&B Table 3"),
        ], note=("Every published rho_D is negative; three of four 95% intervals span "
                 "zero. The covariate-null fraction is undefined for all four rows -- "
                 "see stats.py's floor convention -- because the published metric never "
                 "exceeds its own floor.")),
    },
)


# --------------------------------------------------------------------------------------
# 2. DrivAerML -- NVIDIA PhysicsNeMo-CFD benchmarking framework (arXiv 2507.10747),
#    Tables 4/6/7. 436-run train / 48-run validation split, drag FORCE label (newtons).
#    Preprint, not peer reviewed; no seed spread reported.
# --------------------------------------------------------------------------------------
DRIVAERML = BenchmarkConfig(
    name="DrivAerML",
    csv_path=os.path.join("drivaerml", "drivaerml_pn_436_48.csv"),
    id_col="run_id",
    split_col="split",
    train_values=("train",),
    test_values=("test",),  # PhysicsNeMo's own "validation" split; see note below
    description=("Automotive aerodynamics, DrivAer variants. 16 published morph "
                 "parameters (arXiv 2408.11969 geo_parameters_all.csv), scored on "
                 "PhysicsNeMo-CFD's own drag-force label and 436/48 split -- an exact "
                 "head-to-head, same target, same split, as the published models."),
    verified_in="docs/paper/review/null_travels.md#21-drivaerml--the-clean-kill",
    targets={
        "drag_force_N": Target("drag_force_N", "r2", [
            PublishedEntry("X-MeshGraphNet", 0.92, source="arXiv 2507.10747 Table 6 (surface mesh)"),
            PublishedEntry("FIGConvNet", 0.97, source="arXiv 2507.10747 Table 6 (surface mesh)"),
            PublishedEntry("DoMINO", 0.98, source="arXiv 2507.10747 Table 6 (surface mesh)"),
        ], note=("PhysicsNeMo's validation set is built by sorting on drag and taking the "
                 "top/bottom deciles, which RAISES its variance relative to a random 10% "
                 "and inflates R2 for every model scored on it equally -- including the "
                 "null. The head-to-head is unaffected; the absolute R2 is not comparable "
                 "to a random-split R2. Published values are given to 2 decimals.")),
        "drag_force_N_rank": Target("drag_force_N", "spearman", [
            PublishedEntry("X-MeshGraphNet", 0.96, source="arXiv 2507.10747 Table 4"),
            PublishedEntry("FIGConvNet", 0.99, source="arXiv 2507.10747 Table 4"),
            PublishedEntry("DoMINO", 0.99, source="arXiv 2507.10747 Table 4"),
        ], note="same split and label as drag_force_N, scored in Spearman (Table 4)."),
    },
)


# --------------------------------------------------------------------------------------
# 3. DrivAerNet++ -- NeurIPS 2024 D&B Table 4 (dataset-paper baselines) + TripNet
#    (arXiv 2503.17400 Table 5) + PointNet2D+BiLSTM (arXiv 2601.02112 Table 1, preprint)
#    as current SOTA. Official split, scored on the FULL 1154-design test set.
# --------------------------------------------------------------------------------------
DRIVAERNET = BenchmarkConfig(
    name="DrivAerNet++",
    csv_path=os.path.join("drivaernet", "drivaernet_pp_5819_1154.csv"),
    id_col="design_id",
    split_col="split",
    train_values=("train",),
    test_values=("test",),
    description=("8121 car designs; category tokens (rear/underbody/wheel/mirror family, "
                 "one-hot, levels fit on TRAIN only) + the 23 published design parameters "
                 "where they exist (zero-filled for the ~half of designs -- the "
                 "DrivAerNet-v1 fastbacks -- whose 50-parameter table is unpublished). "
                 "This is a LOWER bound on what published metadata supports: it is "
                 "strictly weaker than a null with every design's parameters."),
    verified_in="docs/paper/review/null_travels.md#22-drivaernet--the-null-beats-what-the-field-cites-not-what-the-fields-best-is",
    targets={
        "cd": Target("cd", "r2", [
            PublishedEntry("PointNet", 0.643, source="NeurIPS 2024 D&B Table 4"),
            PublishedEntry("GCNN", 0.596, source="NeurIPS 2024 D&B Table 4"),
            PublishedEntry("RegDGCNN", 0.641, source="NeurIPS 2024 D&B Table 4"),
            PublishedEntry("TripNet", 0.957, source="arXiv 2503.17400 Table 5 (current SOTA)"),
            PublishedEntry("PointNet2D+BiLSTM", 0.9528,
                          source="arXiv 2601.02112 Table 1 (preprint)"),
        ], note=("The dataset paper's own NeurIPS checklist answers \"[No]\" to error "
                 "bars, so these rows carry no seed spread.")),
    },
)


# --------------------------------------------------------------------------------------
# 4. AhmedML -- arXiv 2407.20801. No recommended split (K-fold), no published ML drag
#    baseline found anywhere (search recorded in null_travels.md Sec 2.3).
# --------------------------------------------------------------------------------------
AHMEDML_PARAM_COLS = [
    "body-length", "body-height", "body-width", "front-arc-diameter",
    "slant-angle-length", "slant-angle-height", "slant-surface-length", "slant-angle-degrees",
]

AHMEDML = BenchmarkConfig(
    name="AhmedML",
    csv_path=os.path.join("ahmedml", "ahmedml_500.csv"),
    id_col="run_id",
    split_col=None,
    # Explicit, not auto-detected: the CSV carries BOTH force targets (cd, cl) as
    # columns so a single file serves both benchmarks.targets entries below. Auto
    # feature-detection (every non-id/label/split column) would silently leak the
    # OTHER target into the regressor -- this list is the 8 published shape
    # parameters only.
    feature_cols=AHMEDML_PARAM_COLS,
    description=("500 Ahmed-body variants, 8 published shape parameters "
                 "(geo_parameters_all.csv), force_mom_all.csv (CONSTANT reference area "
                 "0.112 m^2, so no area term enters the label). No published split -> "
                 "10-fold out-of-sample."),
    verified_in="docs/paper/review/null_travels.md#23-ahmedml--a-bar-nobody-has-cleared",
    targets={
        "cd": Target("cd", "r2", [], note=(
            "No published ML drag baseline found: searched the dataset paper "
            "(arXiv 2407.20801, which reports no ML results), NeuralCFD/GP-UPT "
            "(arXiv 2502.09692, whose drag result is on DrivAerML not AhmedML), "
            "PhysicsNeMo-CFD (DrivAerML only), and FIGConvNet (its \"Ahmed body\" is a "
            "different dataset). Not adjudicable: a benchmark cannot be said to be beaten "
            "by a model whose number was never published. R2 0.684 [0.638, 0.727] "
            "(const-area) is reported here as the bar any future AhmedML drag surrogate "
            "must clear.")),
        "cl": Target("cl", "r2", [], note="no published ML lift baseline found."),
    },
)


# --------------------------------------------------------------------------------------
# 5. WindsorML -- arXiv 2407.19320. No published split id lists (recommended 60/20/20;
#    K-fold used here). SI D.2 publishes one MeshGraphNet result, as an MSE BOUND.
# --------------------------------------------------------------------------------------
WINDSORML_PARAM_COLS = [
    "ratio_length_back_fast", "ratio_height_nose_windshield", "ratio_height_fast_back",
    "side_taper", "clearance", "bottom_taper_angle", "frontal_area",
]

WINDSORML = BenchmarkConfig(
    name="WindsorML",
    csv_path=os.path.join("windsorml", "windsorml_355.csv"),
    id_col="run_id",
    split_col=None,
    feature_cols=WINDSORML_PARAM_COLS,  # see AHMEDML_PARAM_COLS note: the CSV also carries cl
    description=("355 Windsor-body variants, 6 published shape parameters PLUS a derived "
                 "frontal_area column (7 CSV columns total; ordinary covariates here because "
                 "force_mom_all.csv uses a CONSTANT reference area). The WindsorML paper "
                 "(arXiv 2407.19320) itself describes SEVEN design parameters; the published "
                 "geo_parameters_all.csv is missing `ratio_length_front_rear`, and the "
                 "frontal_area column it ships instead is 97.1% explained by `clearance` "
                 "alone -- see docs/paper/review/null_mechanism.md Sec 4, which argues this "
                 "metadata gap, not the body's physics, most likely explains why this is the "
                 "one benchmark where the null decisively fails (Sec 4 Finding 4: a single "
                 "withheld parameter is empirically worth up to R2 0.71 elsewhere). No "
                 "published split id lists here -> 10-fold out-of-sample."),
    verified_in="docs/paper/review/null_travels.md#24-windsorml--the-null-does-not-travel-and-i-am-reporting-that-plainly",
    targets={
        "cd": Target("cd", "r2", [
            # The paper publishes an MSE bound, not an R2 point estimate: SI D.2,
            # verbatim, "using a 60/20/20 split ... it is possible to obtain a MSE of
            # less than 0.00028 for the drag coefficient". Converting requires the
            # target's own variance; done in `windsor_implied_r2_floor` below rather
            # than hand-coded here so the conversion is auditable and reproducible.
        ], note=("The one published number here is a BOUND, not a point estimate -- see "
                 "`windsor_implied_r2_floor`. This is the benchmark where the null "
                 "decisively fails: R2 0.104 [-0.143, 0.267], while the published bound "
                 "implies the MeshGraphNet attains R2 >= 0.79.")),
    },
)

WINDSOR_PUBLISHED_CD_MSE_BOUND = 2.8e-4  # SI D.2, "MSE of less than 0.00028"


def windsor_implied_r2_floor(mse_bound: float = WINDSOR_PUBLISHED_CD_MSE_BOUND) -> PublishedEntry:
    """Convert WindsorML's published MSE bound into an implied R2 LOWER bound
    (``published R2 >= this``), using the label's own variance as the SST
    proxy -- the same conversion ``scripts/covariate_null_crossbench.py``
    performs. A diagnostic conversion, not itself independently verified at
    source (only the MSE bound is); returned as an ``is_bound`` entry with
    ``bound_kind="lower"`` so it is adjudicated by
    :func:`neuroforge.nullbench.stats.verdict_bound_higher` and never assigned
    a covariate-null fraction.
    """
    table = WINDSORML.load("cd")
    sst_proxy = float(np.var(table.y, ddof=0))
    floor = 1.0 - mse_bound / sst_proxy
    return PublishedEntry(
        name="MeshGraphNet (direct KPI head)",
        value=floor,
        source="WindsorML arXiv 2407.19320 SI D.2 (MSE bound, converted to implied R2)",
        note=(f"SI D.2 verbatim: 'MSE of less than {mse_bound:g} for the drag "
              f"coefficient' on a 60/20/20 split (no id lists published). Implied R2 "
              f"floor uses the label's population variance ({sst_proxy:.3e}) as the SST "
              f"proxy: 1 - {mse_bound:g}/{sst_proxy:.3e} = {floor:.4f}."),
        is_bound=True,
        bound_kind="lower",
    )


REGISTRY: dict[str, BenchmarkConfig] = {
    "airfrans": AIRFRANS,
    "drivaerml": DRIVAERML,
    "drivaernet": DRIVAERNET,
    "ahmedml": AHMEDML,
    "windsorml": WINDSORML,
}
