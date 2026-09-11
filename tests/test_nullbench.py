"""Tests for ``neuroforge.nullbench``: the metadata-null harness.

Naming (``docs/paper/review/naming_and_positioning.md``): the protocol is
"the metadata null"; its headline statistic is metadata-only R2
(``R2_meta``, emitted as ``metadata_null_r2``). The former "covariate-null
fraction" is now :func:`~neuroforge.nullbench.stats.published_relative_ratio`,
a demoted comparison aid, not the headline.

Four kinds of test:

* the statistic itself (floor/ceiling convention, undefined cases, bootstrap
  sanity, rounding sensitivity) and a no-drift check against the
  pre-registered verdict rule in ``scripts/covariate_null.py`` /
  ``scripts/covariate_null_crossbench.py``;
* the generic CSV harness (official split vs K-fold, protocol labelling,
  a permutation test that drives the null to its floor);
* reproduction of the committed numbers in ``results/review/covariate_null*
  .json`` (per-benchmark, own protocol) AND ``results/review/
  null_mechanism.json`` (harmonised, cross-benchmark protocol) from the
  shipped worked-example CSVs, at the SAME bootstrap seed and count used to
  produce them, so the check is exact rather than approximate;
* a transcription guard on the sourced (not recomputed) flexible-ceiling
  numbers.
"""

from __future__ import annotations

import csv
import json
import math
import os

import numpy as np
import pytest

from neuroforge.nullbench import (
    PublishedEntry,
    load_table,
    published_relative_ratio,
    r2_score,
    run_null,
    spearman_score,
    verdict_bound_lower,
    verdict_higher,
    verdict_lower,
)
from neuroforge.nullbench.benchmarks import (
    AHMEDML,
    DRIVAERML,
    DRIVAERNET,
    REGISTRY,
    WINDSORML,
    run_benchmark,
    windsor_implied_r2_floor,
)
from neuroforge.nullbench.harness import permutation_check

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# statistic: floor / ceiling convention
# --------------------------------------------------------------------------- #
def test_r2_intercept_only_is_the_floor():
    y = np.array([1.0, 2.0, 3.0, 5.0, -1.0])
    pred = np.full_like(y, y.mean())
    assert r2_score(y, pred) == pytest.approx(0.0, abs=1e-12)


def test_r2_perfect_prediction_is_the_ceiling():
    y = np.array([1.0, 2.0, 3.0, 5.0, -1.0])
    assert r2_score(y, y) == pytest.approx(1.0)


def test_spearman_constant_prediction_is_undefined_not_floor():
    """A constant prediction has no ranks, so Spearman is NaN -- it must not
    be silently mapped to 0 (the floor for a NON-degenerate ranking)."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.ones_like(y)
    assert math.isnan(spearman_score(y, pred))


def test_spearman_perfect_rank_is_the_ceiling():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman_score(y, y) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# published-relative ratio: the undefined / flagged cases
# --------------------------------------------------------------------------- #
def test_ratio_basic():
    r = published_relative_ratio("r2", null_point=0.5, null_ci95=(0.4, 0.6), published=1.0)
    assert r.value == pytest.approx(0.5)
    assert r.value_ci95 == pytest.approx((0.4, 0.6))
    assert r.flags == []


def test_ratio_undefined_when_published_at_or_below_floor():
    """This is the AirfRANS drag case: every published rho_D is negative."""
    r = published_relative_ratio("spearman", null_point=0.87, null_ci95=(0.8, 0.9),
                                 published=-0.117, published_std=0.256)
    assert r.value is None
    assert r.value_ci95 is None
    assert "published_at_or_below_floor" in r.flags
    # published_std still carried, even though no ratio is reported
    assert r.published_std == 0.256


def test_ratio_undefined_when_published_not_reported():
    r = published_relative_ratio("r2", null_point=0.68, null_ci95=(0.6, 0.7), published=None)
    assert r.value is None
    assert r.flags == ["published_not_reported"]


def test_ratio_flags_when_null_beats_published():
    """The DrivAerML case: the null's CI covers a published model."""
    r = published_relative_ratio("r2", null_point=0.973, null_ci95=(0.958, 0.983), published=0.92)
    assert r.value > 1.0
    assert "null_exceeds_published" in r.flags


def test_ratio_flags_when_null_at_or_below_its_own_floor():
    r = published_relative_ratio("r2", null_point=-0.02, null_ci95=(-0.1, 0.05), published=0.5)
    assert r.value <= 0
    assert "null_at_or_below_floor" in r.flags


def test_ratio_never_clips_a_value_above_one_or_below_zero():
    """Both directions are reported, not hidden -- see the module docstring."""
    over = published_relative_ratio("r2", 0.99, (0.9, 1.0), 0.5)
    under = published_relative_ratio("r2", -0.5, (-0.6, -0.4), 0.5)
    assert over.value > 1.0
    assert under.value < 0.0


# --------------------------------------------------------------------------- #
# rounding sensitivity (coordinator-requested check, not just an argument):
# is `published_relative_ratio` as sensitive to published-value rounding as
# the null-normalised gain `G` the positioning review demonstrated a 6x swing
# for on DrivAerML? Checked directly, both the safe case and the dangerous one.
# --------------------------------------------------------------------------- #
def test_rounding_sensitivity_drivaerml_domino_is_small_not_6x():
    """DoMINO published 0.98 to 2 decimals; the positioning review shows a
    DIFFERENT ratio (null-normalised gain, denominator 1-R2_meta=0.0269)
    swings 6x under this same rounding. This ratio's denominator is the
    published value itself (~0.98, far from the floor), so the swing should
    be small. Computed, not assumed."""
    null_point, ci = 0.9730580198150222, (0.9580772227747316, 0.9826207956902174)
    r = published_relative_ratio("r2", null_point, ci, published=0.98, published_precision=2)
    assert r.rounding_interval is not None
    width = r.rounding_interval[1] - r.rounding_interval[0]
    relative_swing = width / r.value
    assert relative_swing < 0.05, "expected a small swing, not the 6x G shows"
    assert "rounding_sensitive" not in r.flags


def test_rounding_sensitivity_flags_a_published_value_near_the_floor():
    """The dangerous regime is the OPPOSITE end from where G fails: a
    published value reported to few decimals AND close to the metric's own
    floor, where a small absolute rounding band is a large relative one."""
    r = published_relative_ratio("r2", null_point=0.3, null_ci95=(0.25, 0.35),
                                 published=0.05, published_precision=2)
    assert r.rounding_interval is not None
    width = r.rounding_interval[1] - r.rounding_interval[0]
    assert width / r.value > 0.15
    assert "rounding_sensitive" in r.flags


def test_rounding_interval_crosses_floor_is_flagged_not_computed_wrong():
    """If the rounding band itself straddles the floor, no rounding interval
    is defensible (dividing by ~0 blows up); flagged instead of silently
    producing a huge or infinite number."""
    r = published_relative_ratio("r2", null_point=0.1, null_ci95=(0.05, 0.15),
                                 published=0.005, published_precision=2)
    assert r.rounding_interval is None
    assert "rounding_interval_crosses_floor" in r.flags


# --------------------------------------------------------------------------- #
# no-drift check against the pre-registered decision rule
# --------------------------------------------------------------------------- #
def _grid():
    rng = np.random.default_rng(0)
    for _ in range(200):
        published = rng.uniform(-1, 1)
        null_point = rng.uniform(-1, 1)
        lo, hi = sorted(rng.uniform(-1, 1, size=2))
        yield published, null_point, lo, hi


def test_verdict_higher_matches_scripts_covariate_null():
    from scripts.covariate_null import verdict as scripts_verdict

    for published, null_point, lo, hi in _grid():
        assert verdict_higher(published, null_point, lo, hi) == \
            scripts_verdict(published, null_point, lo, hi)
    assert verdict_higher(None, 0.5, 0.4, 0.6) == scripts_verdict(None, 0.5, 0.4, 0.6)


def test_verdict_lower_matches_scripts_covariate_null_crossbench():
    from scripts.covariate_null_crossbench import verdict_lower as scripts_verdict_lower

    for published, null_point, lo, hi in _grid():
        assert verdict_lower(published, null_point, lo, hi) == \
            scripts_verdict_lower(published, null_point, lo, hi)


def test_verdict_bound_lower_matches_scripts_covariate_null_crossbench():
    from scripts.covariate_null_crossbench import verdict_bound_lower as scripts_bound

    for published, null_point, lo, hi in _grid():
        assert verdict_bound_lower(published, null_point, lo, hi) == \
            scripts_bound(published, null_point, lo, hi)


# --------------------------------------------------------------------------- #
# generic CSV harness
# --------------------------------------------------------------------------- #
def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _synthetic_official_split_csv(path, n_train=60, n_test=20, seed=0):
    """y = 2*x1 - x2 + noise; enough signal for OLS to clearly beat the floor."""
    rng = np.random.default_rng(seed)
    rows = []
    n = n_train + n_test
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    y = 2 * x1 - x2 + rng.normal(0, 0.05, n)
    for i in range(n):
        tag = "train" if i < n_train else "test"
        rows.append([f"case{i}", tag, x1[i], x2[i], y[i]])
    _write_csv(path, ["id", "split", "x1", "x2", "y"], rows)


def test_official_split_protocol_used_when_split_col_given(tmp_path):
    path = tmp_path / "cases.csv"
    _synthetic_official_split_csv(path)
    table = load_table(str(path), id_col="id", label_col="y", split_col="split")
    result = run_null(table, metric="r2", n_boot=200, seed=0)
    assert result.protocol == "official_split"
    assert result.n_fit == 60
    assert result.n_score == 20
    assert result.out_of_sample > 0.8  # strong synthetic signal


def test_kfold_protocol_used_when_no_split_col(tmp_path):
    path = tmp_path / "cases.csv"
    _synthetic_official_split_csv(path)
    # explicit feature_cols: the fixture CSV also carries a text "split" column, which
    # auto-detection would otherwise (correctly) try to use as a feature and fail on --
    # exactly the leak `AIRFRANS_PARAM_COLS`/`AHMEDML_PARAM_COLS` guard against for the
    # shipped benchmarks (see benchmarks.py).
    table = load_table(str(path), id_col="id", label_col="y", feature_cols=["x1", "x2"])
    assert not table.has_split()
    result = run_null(table, metric="r2", folds=5, n_boot=200, seed=0)
    assert result.protocol == "kfold_oos_k5"
    assert result.n_fit == result.n_score == 80
    assert result.out_of_sample > 0.7


def test_auto_feature_detection_excludes_id_split_and_label(tmp_path):
    path = tmp_path / "cases.csv"
    _synthetic_official_split_csv(path)
    table = load_table(str(path), id_col="id", label_col="y", split_col="split")
    assert table.feature_names == ["x1", "x2"]


def test_permutation_check_collapses_to_the_floor(tmp_path):
    path = tmp_path / "cases.csv"
    _synthetic_official_split_csv(path, n_train=200, n_test=50, seed=1)
    table = load_table(str(path), id_col="id", label_col="y", split_col="split")
    real = run_null(table, metric="r2", n_boot=200, seed=0).out_of_sample
    permuted = permutation_check(table, metric="r2", seed=1)
    assert real > 0.8
    assert permuted < 0.1  # nowhere near the real fit; machinery isn't manufacturing signal


def test_null_result_emits_the_settled_headline_field_names(tmp_path):
    """naming_and_positioning.md: the primary output is `metadata_null_r2`
    (metric-tagged), alongside `metadata_null_mse` -- always, regardless of
    which metric was used to score the run."""
    path = tmp_path / "cases.csv"
    _synthetic_official_split_csv(path)
    table = load_table(str(path), id_col="id", label_col="y", split_col="split")
    result = run_null(table, metric="r2", n_boot=200, seed=0)
    d = result.to_dict()
    assert d["metadata_null_r2"] == pytest.approx(result.out_of_sample)
    assert d["metadata_null_r2_ci95"] == pytest.approx(list(result.ci95))
    assert d["metadata_null_mse"] == pytest.approx(result.mse_out_of_sample)
    assert d["metadata_null_mse_ci95"] == pytest.approx(list(result.mse_ci95))
    assert "covariate_null_fraction" not in d
    assert "metadata_null_spearman" not in d  # only the metric actually run is present


def test_published_comparison_verdict_and_ratio(tmp_path):
    path = tmp_path / "cases.csv"
    _synthetic_official_split_csv(path)
    table = load_table(str(path), id_col="id", label_col="y", split_col="split")
    result = run_null(table, metric="r2", n_boot=500, seed=0, published=[
        PublishedEntry("weak_model", value=0.1, std=0.02, source="fixture"),
        PublishedEntry("strong_model", value=0.999, std=0.001, source="fixture"),
        PublishedEntry("no_number", value=None),
    ])
    by_name = {c["model"]: c for c in result.comparisons}
    assert by_name["weak_model"]["verdict"] == "BELOW THE NULL"
    assert by_name["weak_model"]["published_relative_ratio"]["published_relative_ratio"] > 1.0
    assert by_name["strong_model"]["verdict"] == "clears"
    assert by_name["no_number"]["verdict"] == "not reported"
    # a ratio record is still emitted (uniform schema), but its value is None and the
    # reason is flagged -- see stats.published_relative_ratio
    assert by_name["no_number"]["published_relative_ratio"]["published_relative_ratio"] is None
    assert "published_not_reported" in by_name["no_number"]["published_relative_ratio"]["flags"]


def test_missing_column_raises_clear_error(tmp_path):
    path = tmp_path / "cases.csv"
    _synthetic_official_split_csv(path)
    with pytest.raises(ValueError, match="not found"):
        load_table(str(path), id_col="id", label_col="does_not_exist")


# --------------------------------------------------------------------------- #
# reproduction (per-benchmark, own protocol): shipped worked examples vs
# committed results/review/covariate_null*.json
# --------------------------------------------------------------------------- #
def _load_committed(name):
    with open(os.path.join(REPO, "results", "review", name), encoding="utf-8") as fh:
        return json.load(fh)


def test_drivaerml_reproduces_committed_numbers():
    committed = _load_committed("covariate_null_crossbench.json")
    dm = committed["benchmarks"]["DrivAerML_benchmark_split"]["targets"]["drag_force"]["params_linear"]

    result = run_benchmark(DRIVAERML, "drag_force_N", n_boot=10000)

    assert result.protocol == "official_split"
    assert result.n_fit == dm["n_fit"]
    assert result.n_score == dm["n_score"]
    assert result.out_of_sample == pytest.approx(dm["out_of_sample"]["r2"], rel=1e-9)
    assert result.ci95[0] == pytest.approx(dm["ci95"]["r2"][0], rel=1e-9)
    assert result.ci95[1] == pytest.approx(dm["ci95"]["r2"][1], rel=1e-9)


def test_drivaernet_reproduces_committed_numbers():
    committed = _load_committed("covariate_null_crossbench.json")
    dn = committed["benchmarks"]["DrivAerNet++"]["targets"]["cd"]["category_params_where_published"]

    result = run_benchmark(DRIVAERNET, "cd", n_boot=10000)

    assert result.protocol == "official_split"
    assert result.n_fit == dn["n_fit"]
    assert result.n_score == dn["n_score"]
    assert result.out_of_sample == pytest.approx(dn["out_of_sample"]["r2"], rel=1e-9)
    assert result.ci95[0] == pytest.approx(dn["ci95"]["r2"][0], rel=1e-9)
    assert result.ci95[1] == pytest.approx(dn["ci95"]["r2"][1], rel=1e-9)


def test_ahmedml_reproduces_committed_numbers():
    committed = _load_committed("covariate_null_crossbench.json")
    a = committed["benchmarks"]["AhmedML"]["targets"]["cd"]["params_linear"]

    result = run_benchmark(AHMEDML, "cd", n_boot=10000)

    assert result.protocol == "kfold_oos_k10"
    assert result.out_of_sample == pytest.approx(a["out_of_sample"]["r2"], rel=1e-9)
    assert result.ci95[0] == pytest.approx(a["ci95"]["r2"][0], rel=1e-9)
    assert result.ci95[1] == pytest.approx(a["ci95"]["r2"][1], rel=1e-9)


def test_windsorml_reproduces_committed_numbers_and_the_counterexample_verdict():
    committed = _load_committed("covariate_null_crossbench.json")
    w = committed["benchmarks"]["WindsorML"]["targets"]["cd"]["params_linear"]

    result = run_benchmark(WINDSORML, "cd", n_boot=10000)

    assert result.protocol == "kfold_oos_k10"
    assert result.out_of_sample == pytest.approx(w["out_of_sample"]["r2"], rel=1e-9)
    assert result.ci95[0] == pytest.approx(w["ci95"]["r2"][0], rel=1e-9)
    assert result.ci95[1] == pytest.approx(w["ci95"]["r2"][1], rel=1e-9)

    # the published entry is a bound, converted to an implied R2 floor, and the null
    # decisively fails to clear it -- the one counterexample among the five benchmarks
    bound_entry = windsor_implied_r2_floor()
    assert bound_entry.value > result.ci95[1]
    from neuroforge.nullbench.stats import verdict_bound_higher
    v = verdict_bound_higher(bound_entry.value, result.out_of_sample, *result.ci95)
    assert v.startswith("clears")


_AIRFRANS_CSV = os.path.join(
    REPO, "src", "neuroforge", "nullbench", "data", "airfrans", "airfrans_full_800_200.csv")


@pytest.mark.skipif(not os.path.exists(_AIRFRANS_CSV),
                    reason="AirfRANS worked-example CSV not yet built")
def test_airfrans_reproduces_committed_numbers():
    committed = _load_committed("covariate_null_trainfit.json")
    cl = committed["targets"]["cl"]["U_alpha_alpha2_naca"]
    cd = committed["targets"]["cd"]["U_alpha_alpha2_naca"]

    from neuroforge.nullbench.benchmarks import AIRFRANS

    r_cl = run_benchmark(AIRFRANS, "cl", n_boot=10000)
    r_cd = run_benchmark(AIRFRANS, "cd", n_boot=10000)

    assert r_cl.protocol == "official_split"
    assert r_cl.n_fit == committed["n_train"]
    assert r_cl.n_score == committed["n_test"]
    assert r_cl.out_of_sample == pytest.approx(cl["spearman_train_fit_test_score"], rel=1e-9)
    assert r_cl.ci95[0] == pytest.approx(cl["ci95"][0], rel=1e-9)
    assert r_cl.ci95[1] == pytest.approx(cl["ci95"][1], rel=1e-9)

    assert r_cd.out_of_sample == pytest.approx(cd["spearman_train_fit_test_score"], rel=1e-9)
    assert r_cd.ci95[0] == pytest.approx(cd["ci95"][0], rel=1e-9)
    assert r_cd.ci95[1] == pytest.approx(cd["ci95"][1], rel=1e-9)

    # every published drag rho is negative -> undefined ratio for every row (see stats.py)
    for c in r_cd.comparisons:
        if c["published"] is not None and c["published"] <= 0:
            assert c["published_relative_ratio"]["published_relative_ratio"] is None
            assert "published_at_or_below_floor" in c["published_relative_ratio"]["flags"]


# --------------------------------------------------------------------------- #
# reproduction (harmonised, cross-benchmark protocol): shipped harmonised
# CSVs vs committed results/review/null_mechanism.json
# --------------------------------------------------------------------------- #
_HARMONISED_DIR = os.path.join(
    REPO, "src", "neuroforge", "nullbench", "data", "harmonised")


@pytest.mark.skipif(not os.path.isdir(_HARMONISED_DIR),
                    reason="harmonised worked-example CSVs not yet built")
@pytest.mark.parametrize("name,display_name", [
    ("airfrans", "AirfRANS"), ("ahmedml", "AhmedML"), ("windsorml", "WindsorML"),
    ("drivaerml", "DrivAerML"), ("drivaernet", "DrivAerNet++"),
])
def test_harmonised_reproduces_null_mechanism_json(name, display_name):
    from neuroforge.nullbench.harmonised import REGISTRY as HARM_REGISTRY
    from neuroforge.nullbench.harmonised import run_harmonised

    committed = _load_committed("null_mechanism.json")
    target = committed["benchmarks"][display_name]["targets"]["cd"]["null_r2_linear"]

    result = run_harmonised(HARM_REGISTRY[name], n_boot=200)  # point estimate is boot-independent

    assert result.benchmark == display_name
    assert result.metadata_null_r2 == pytest.approx(target, rel=1e-9)


def test_harmonised_n_cases_match_null_mechanism_json():
    from neuroforge.nullbench.harmonised import REGISTRY as HARM_REGISTRY
    from neuroforge.nullbench.harmonised import run_harmonised

    committed = _load_committed("null_mechanism.json")
    for name, cfg in HARM_REGISTRY.items():
        result = run_harmonised(cfg, n_boot=50)
        expected = committed["benchmarks"][cfg.display_name]["n_cases"]
        assert result.n_cases == expected, name


def test_harmonised_ranking_matches_null_mechanism_md():
    """The published ranking (docs/paper/review/null_mechanism.md Sec 1):
    DrivAerML > DrivAerNet++ > AhmedML > AirfRANS > WindsorML."""
    from neuroforge.nullbench.leaderboard import build_harmonised_table

    table = build_harmonised_table(n_boot=100)
    order = [name for name, _ in table["ranking_by_metadata_null_r2"]]
    assert order == ["DrivAerML", "DrivAerNet++", "AhmedML", "AirfRANS", "WindsorML"]


def test_harmonised_carries_no_published_comparison_and_says_why():
    from neuroforge.nullbench.harmonised import REGISTRY as HARM_REGISTRY
    from neuroforge.nullbench.harmonised import run_harmonised

    result = run_harmonised(HARM_REGISTRY["drivaerml"], n_boot=50)
    d = result.to_dict()
    assert d["published_comparison"] is None
    assert "own protocol" in d["published_comparison_note"].lower()


# --------------------------------------------------------------------------- #
# flexible-ceiling: sourced, not recomputed -- transcription guard only
# --------------------------------------------------------------------------- #
def test_flexible_ceiling_matches_null_mechanism_json_verbatim():
    from neuroforge.nullbench.harmonised import REGISTRY as HARM_REGISTRY
    from neuroforge.nullbench.harmonised import run_harmonised

    committed = _load_committed("null_mechanism.json")
    for name, cfg in HARM_REGISTRY.items():
        result = run_harmonised(cfg, n_boot=50)
        expected = committed["benchmarks"][cfg.display_name]["targets"]["cd"][
            "C1_flex_r2_lower_bound"]
        assert result.flexible_ceiling_r2_sourced == pytest.approx(expected, rel=1e-12), name


def test_flexible_ceiling_at_or_above_linear_null_everywhere():
    """Sanity on the sourced pair itself: a flexible model, chosen by
    validation performance, cannot score below the linear one it contains as
    a special case (within numerical tolerance of the model-selection noise)."""
    from neuroforge.nullbench.harmonised import run_all_harmonised

    for name, result in run_all_harmonised(n_boot=50).items():
        assert result.flexible_ceiling_r2_sourced >= result.metadata_null_r2 - 1e-6, name


# --------------------------------------------------------------------------- #
# all five configs are registered and loadable end to end (smoke)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_registry_configs_load_and_run(name):
    cfg = REGISTRY[name]
    for target in cfg.targets:
        if name == "airfrans" and not os.path.exists(cfg.path()):
            pytest.skip("AirfRANS worked-example CSV not yet built")
        result = run_benchmark(cfg, target, n_boot=100)
        assert result.n_score > 0
        assert result.protocol in ("official_split",) or result.protocol.startswith("kfold_oos")
