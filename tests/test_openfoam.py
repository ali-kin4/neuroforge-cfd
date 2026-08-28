"""Tests for the OpenFOAM (WSL2) full-case backend.

Everything here runs **without OpenFOAM installed**: the case writer, the field
parser, the log parser and the path translation are pure Python and are checked
against golden text. The one test that actually invokes ``simpleFoam`` is marked
``slow`` and skipped unless an installation is found.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from neuroforge.core.types import FlowCase, FlowField
from neuroforge.solver import openfoam as of


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def small_case() -> FlowCase:
    return FlowCase.from_airfoil(
        airfoil="naca2412", aoa=4.0, reynolds=1.0e6, u_inf=1.0, resolution=24, n_surface=80
    )


@pytest.fixture
def warm_field(small_case) -> FlowField:
    from neuroforge.geometry.sdf import solid_mask

    ny, nx = small_case.domain.shape
    rng = np.random.default_rng(0)
    mask = solid_mask(small_case.geometry, small_case.domain).astype(np.float32)
    return FlowField(
        domain=small_case.domain,
        u=(1.0 + 0.05 * rng.standard_normal((ny, nx))).astype(np.float32) * mask,
        v=(0.05 * rng.standard_normal((ny, nx))).astype(np.float32) * mask,
        p=(0.01 * rng.standard_normal((ny, nx))).astype(np.float32) * mask,
        nut=(1e-4 * np.abs(rng.standard_normal((ny, nx)))).astype(np.float32) * mask,
        mask=mask,
    )


# --------------------------------------------------------------------------- #
# Path translation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "win, posix",
    [
        (r"D:\Codes\Github\neuroforge-cfd", "/mnt/d/Codes/Github/neuroforge-cfd"),
        (r"C:\Users\x\AppData\Local\Temp", "/mnt/c/Users/x/AppData/Local/Temp"),
        ("/home/ali/case", "/home/ali/case"),
        ("relative/path", "relative/path"),
        ("", ""),
    ],
)
def test_to_wsl_path(win, posix):
    assert of.to_wsl_path(win) == posix


def test_to_wsl_path_lowercases_drive():
    assert of.to_wsl_path(r"E:\a\b").startswith("/mnt/e/")


# --------------------------------------------------------------------------- #
# Availability probes never raise
# --------------------------------------------------------------------------- #


def test_openfoam_available_is_false_when_nothing_is_found(monkeypatch):
    monkeypatch.setattr(of, "detect_openfoam", lambda distro=None: None)
    assert of.openfoam_available() is False


def test_openfoam_available_is_true_when_env_is_located(monkeypatch):
    env = of.OpenFOAMEnv(distro="Ubuntu", bashrc="/opt/openfoam/etc/bashrc")
    monkeypatch.setattr(of, "detect_openfoam", lambda distro=None: env)
    assert of.openfoam_available() is True


def test_detect_openfoam_returns_none_when_the_probe_fails(monkeypatch):
    """A broken/absent WSL must degrade to None, never propagate an OSError."""
    def boom(*_a, **_k):
        raise OSError("wsl.exe is not installed")

    monkeypatch.setattr(of, "_run_bash", boom)
    monkeypatch.setitem(of._ENV_CACHE, "probe-distro", None)
    assert of.detect_openfoam("probe-distro", refresh=True) is None


@pytest.mark.slow
def test_list_wsl_distros_returns_list():
    # Shells out to wsl.exe, so it is kept out of the fast suite.
    assert isinstance(of.list_wsl_distros(), list)


def test_require_openfoam_message_has_install_guidance(monkeypatch):
    monkeypatch.setattr(of, "detect_openfoam", lambda distro=None: None)
    with pytest.raises(of.OpenFOAMUnavailable) as exc:
        of.require_openfoam()
    assert "add-debian-repo.sh" in str(exc.value)


# --------------------------------------------------------------------------- #
# Case writing
# --------------------------------------------------------------------------- #


def test_write_case_cold_creates_expected_tree(tmp_path, small_case):
    d = of.write_case(small_case, str(tmp_path / "cold"), n_iter=50)
    for rel in [
        "system/controlDict", "system/fvSchemes", "system/fvSolution",
        "system/blockMeshDict", "system/topoSetDict",
        "constant/transportProperties", "constant/turbulenceProperties",
        "0/U", "0/p", "0/nut", "0/nuTilda", "0/cellId", "neuroforge.json",
    ]:
        assert os.path.isfile(os.path.join(d, rel)), f"missing {rel}"


def test_written_dicts_use_lf_endings(tmp_path, small_case):
    """OpenFOAM's dictionary parser trips over CRLF; we are on Windows."""
    d = of.write_case(small_case, str(tmp_path / "lf"), n_iter=10)
    with open(os.path.join(d, "system", "controlDict"), "rb") as fh:
        assert b"\r\n" not in fh.read()


def test_cold_start_is_uniform_freestream(tmp_path, small_case):
    d = of.write_case(small_case, str(tmp_path / "cold"), n_iter=10)
    text = open(os.path.join(d, "0", "U"), encoding="utf-8").read()
    assert "internalField   uniform" in text
    u = of.read_volfield(os.path.join(d, "0", "U"))
    aoa = np.deg2rad(small_case.bc.aoa_deg)
    assert u[0, 0] == pytest.approx(small_case.bc.u_inf * np.cos(aoa), rel=1e-6)
    assert u[0, 1] == pytest.approx(small_case.bc.u_inf * np.sin(aoa), rel=1e-6)


def test_warm_start_writes_nonuniform_fields(tmp_path, small_case, warm_field):
    d = of.write_case(small_case, str(tmp_path / "warm"), initial=warm_field, n_iter=10)
    ny, nx = small_case.domain.shape
    u = of.read_volfield(os.path.join(d, "0", "U"))
    assert u.shape == (ny * nx, 3)
    # Written in flat (j*nx + i) order and round-tripped through the parser.
    np.testing.assert_allclose(u[:, 0], np.asarray(warm_field.u).ravel(), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(u[:, 1], np.asarray(warm_field.v).ravel(), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(u[:, 2], 0.0)
    p = of.read_volfield(os.path.join(d, "0", "p"))
    np.testing.assert_allclose(p, np.asarray(warm_field.p).ravel(), rtol=1e-6, atol=1e-9)


def test_warm_start_nutilda_is_bounded_below_by_freestream(tmp_path, small_case, warm_field):
    d = of.write_case(small_case, str(tmp_path / "warm"), initial=warm_field, n_iter=10)
    nutilda = of.read_volfield(os.path.join(d, "0", "nuTilda"))
    nu = small_case.fluid.kinematic_viscosity
    assert nutilda.min() >= of.NUTILDA_FREESTREAM_RATIO * nu - 1e-12


def test_cell_id_marker_is_the_flat_grid_index(tmp_path, small_case):
    d = of.write_case(small_case, str(tmp_path / "ids"), n_iter=10)
    ny, nx = small_case.domain.shape
    ids = of.read_volfield(os.path.join(d, "0", "cellId"))
    np.testing.assert_array_equal(ids, np.arange(ny * nx))


def test_block_mesh_centres_match_the_grid(tmp_path, small_case):
    """The block is extended half a cell so cell centres land on grid points."""
    d = of.write_case(small_case, str(tmp_path / "mesh"), n_iter=10)
    text = open(os.path.join(d, "system", "blockMeshDict"), encoding="utf-8").read()
    dom = small_case.domain
    xmin, xmax, ymin, ymax = dom.bounds
    assert f"({dom.nx} {dom.ny} 1)" in text
    verts = [
        tuple(float(v) for v in line.strip(" ()").split())
        for line in text.split("vertices")[1].split(");")[0].splitlines()
        if line.strip().startswith("(") and line.strip(" ()").strip()
    ]
    assert len(verts) == 8
    xs = sorted({v[0] for v in verts})
    ys = sorted({v[1] for v in verts})
    assert xs[0] == pytest.approx(xmin - 0.5 * dom.dx)
    assert xs[-1] == pytest.approx(xmax + 0.5 * dom.dx)
    assert ys[0] == pytest.approx(ymin - 0.5 * dom.dy)
    assert ys[-1] == pytest.approx(ymax + 0.5 * dom.dy)


def test_topo_set_selects_exactly_the_fluid_cells(tmp_path, small_case):
    from neuroforge.geometry.sdf import solid_mask

    d = of.write_case(small_case, str(tmp_path / "topo"), n_iter=10)
    text = open(os.path.join(d, "system", "topoSetDict"), encoding="utf-8").read()
    body = text.split("value")[1]
    ids = [int(t) for t in body.replace("(", " ").replace(")", " ").split() if t.isdigit()]
    # geometry.solid_mask is a FLUID indicator: 1.0 in the fluid, 0.0 in the body.
    fluid = solid_mask(small_case.geometry, small_case.domain) > 0.5
    np.testing.assert_array_equal(np.asarray(ids), np.flatnonzero(fluid.ravel()))
    assert len(ids) < small_case.domain.nx * small_case.domain.ny  # body was carved out


def test_body_is_actually_carved_out_at_production_resolution(tmp_path):
    """Regression: `geometry.solid_mask` is a *fluid* indicator.

    Reading it as a solid indicator inverts the cut-out -- the mesh keeps the
    airfoil and deletes the flow. Both guards in `write_case` survive that
    inversion (a handful of cells land inside the body, so neither "no fluid"
    nor "no solid" trips), so it has to be caught here.
    """
    from neuroforge.geometry.sdf import solid_mask

    case = FlowCase.from_airfoil(airfoil="naca2412", aoa=4.0, resolution=128)
    d = of.write_case(case, str(tmp_path / "prod"), n_iter=10)
    text = open(os.path.join(d, "system", "topoSetDict"), encoding="utf-8").read()
    ids = [
        int(t)
        for t in text.split("value")[1].replace("(", " ").replace(")", " ").split()
        if t.isdigit()
    ]
    n_cells = case.domain.nx * case.domain.ny
    n_solid = n_cells - len(ids)
    # A NACA section on a 3-chord domain at res 128 occupies O(100) cells --
    # a small minority. Under the inversion this number would be ~16 000.
    assert 20 < n_solid < 0.05 * n_cells
    fluid = solid_mask(case.geometry, case.domain) > 0.5
    np.testing.assert_array_equal(np.asarray(ids), np.flatnonzero(fluid.ravel()))
    report = json.loads(open(os.path.join(d, "neuroforge.json"), encoding="utf-8").read())
    assert report["geometry_check"]["components"] == 1
    assert report["solid_cells"] == n_solid


def test_transport_properties_carry_the_case_viscosity(tmp_path, small_case):
    d = of.write_case(small_case, str(tmp_path / "nu"), n_iter=10)
    text = open(os.path.join(d, "constant", "transportProperties"), encoding="utf-8").read()
    assert f"{small_case.fluid.kinematic_viscosity:.9g}" in text


def test_turbulence_model_is_spalart_allmaras(tmp_path, small_case):
    """AirfRANS is SA, not k-omega SST -- the 4-channel spec has one nut field."""
    d = of.write_case(small_case, str(tmp_path / "sa"), n_iter=10)
    for name in ("turbulenceProperties", "momentumTransport"):
        text = open(os.path.join(d, "constant", name), encoding="utf-8").read()
        assert "SpalartAllmaras" in text
        assert "kOmega" not in text


def test_control_dict_iteration_cap(tmp_path, small_case):
    d = of.write_case(small_case, str(tmp_path / "iters"), n_iter=1234)
    text = open(os.path.join(d, "system", "controlDict"), encoding="utf-8").read()
    assert "endTime         1234;" in text
    assert "application     simpleFoam;" in text


def test_write_case_rejects_mismatched_warm_field(tmp_path, small_case, warm_field):
    bad = FlowField(
        domain=small_case.domain,
        u=np.zeros((8, 8), np.float32), v=np.zeros((8, 8), np.float32),
        p=np.zeros((8, 8), np.float32), mask=np.ones((8, 8), np.float32),
    )
    with pytest.raises(ValueError):
        of.write_case(small_case, str(tmp_path / "bad"), initial=bad)


def test_write_case_overwrites_existing_dir(tmp_path, small_case):
    d = str(tmp_path / "twice")
    of.write_case(small_case, d, n_iter=10)
    stray = os.path.join(d, "stray.txt")
    open(stray, "w").close()
    of.write_case(small_case, d, n_iter=10)
    assert not os.path.exists(stray)


# --------------------------------------------------------------------------- #
# Field parsing
# --------------------------------------------------------------------------- #


def test_read_volfield_uniform_scalar(tmp_path):
    p = tmp_path / "f"
    p.write_text("internalField   uniform 3.5;\n", encoding="utf-8")
    np.testing.assert_allclose(of.read_volfield(str(p)), [3.5])


def test_read_volfield_nonuniform_scalar(tmp_path):
    p = tmp_path / "f"
    p.write_text("internalField   nonuniform List<scalar>\n3\n(\n1\n2\n3\n)\n;\n", encoding="utf-8")
    np.testing.assert_allclose(of.read_volfield(str(p)), [1.0, 2.0, 3.0])


def test_read_volfield_nonuniform_vector(tmp_path):
    p = tmp_path / "f"
    p.write_text(
        "internalField   nonuniform List<vector>\n2\n(\n(1 2 0)\n(3 4 0)\n)\n;\n",
        encoding="utf-8",
    )
    np.testing.assert_allclose(of.read_volfield(str(p)), [[1, 2, 0], [3, 4, 0]])


def test_read_volfield_missing_entry_raises(tmp_path):
    p = tmp_path / "f"
    p.write_text("nothing here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        of.read_volfield(str(p))


# --------------------------------------------------------------------------- #
# Time-directory selection
# --------------------------------------------------------------------------- #


def _touch_time(root, name):
    os.makedirs(os.path.join(root, name), exist_ok=True)
    open(os.path.join(root, name, "U"), "w").close()


def test_latest_time_ignores_the_zero_directory(tmp_path):
    """0/ always exists and, warm-started, holds the *input* -- never a result."""
    _touch_time(str(tmp_path), "0")
    assert of._latest_time(str(tmp_path)) is None


def test_latest_time_picks_the_largest_written_step(tmp_path):
    for name in ("0", "10", "200", "50"):
        _touch_time(str(tmp_path), name)
    assert of._latest_time(str(tmp_path)) == "200"


def test_latest_time_skips_dirs_without_a_field(tmp_path):
    _touch_time(str(tmp_path), "10")
    os.makedirs(os.path.join(str(tmp_path), "20"))  # no U written
    assert of._latest_time(str(tmp_path)) == "10"


# --------------------------------------------------------------------------- #
# Geometry sanity
# --------------------------------------------------------------------------- #


def test_check_solid_region_accepts_a_solid_blob():
    solid = np.zeros((20, 20), bool)
    solid[6:14, 6:14] = True
    report = of.check_solid_region(solid)
    assert report["components"] == 1
    assert report["max_thickness"] == 8
    assert report["thin_column_fraction"] == 0.0


def test_check_solid_region_warns_on_disconnected_specks():
    solid = np.zeros((20, 20), bool)
    solid[3:6, 3:6] = True
    solid[12:15, 12:15] = True
    with pytest.warns(UserWarning, match="disconnected"):
        report = of.check_solid_region(solid)
    assert report["components"] == 2
    assert report["solid_cells"] == 18


def test_check_solid_region_warns_when_the_body_is_a_thin_staircase():
    solid = np.zeros((20, 20), bool)
    solid[9:11, 4:12] = True   # 2 cells thick at most
    solid[9, 12:16] = True     # tapering to a one-cell tail
    with pytest.warns(UserWarning, match="staircase"):
        report = of.check_solid_region(solid)
    assert report["max_thickness"] == 2
    assert report["thin_column_fraction"] > 0.0


def test_check_solid_region_reports_the_tapering_tail_without_warning_when_thick():
    solid = np.zeros((30, 30), bool)
    solid[10:20, 4:20] = True  # 10 cells thick
    solid[14, 20:26] = True    # one-cell trailing edge, as every airfoil has
    report = of.check_solid_region(solid)
    assert report["max_thickness"] == 10
    assert 0.0 < report["thin_column_fraction"] < 0.5


def test_check_solid_region_on_empty_input():
    report = of.check_solid_region(np.zeros((8, 8), bool))
    assert report["solid_cells"] == 0


# --------------------------------------------------------------------------- #
# Log parsing
# --------------------------------------------------------------------------- #

_LOG = """\
Starting time loop

Time = 1

smoothSolver:  Solving for Ux, Initial residual = 1, Final residual = 0.05, No Iterations 3
smoothSolver:  Solving for Uy, Initial residual = 1, Final residual = 0.04, No Iterations 3
GAMG:  Solving for p, Initial residual = 1, Final residual = 0.009, No Iterations 8
smoothSolver:  Solving for nuTilda, Initial residual = 0.9, Final residual = 0.02, No Iterations 2
ExecutionTime = 0.31 s  ClockTime = 1 s

Time = 2

smoothSolver:  Solving for Ux, Initial residual = 0.002, Final residual = 1e-05, No Iterations 3
smoothSolver:  Solving for Uy, Initial residual = 0.003, Final residual = 2e-05, No Iterations 3
GAMG:  Solving for p, Initial residual = 5e-06, Final residual = 4e-08, No Iterations 6
smoothSolver:  Solving for nuTilda, Initial residual = 4e-07, Final residual = 1e-09, No Iterations 2
ExecutionTime = 0.62 s  ClockTime = 2 s

SIMPLE solution converged in 2 iterations

End
"""


def test_parse_log_iterations_and_convergence():
    info = of.parse_simple_foam_log(_LOG)
    assert info["iterations"] == 2
    assert info["converged"] is True
    assert info["converged_at"] == 2


def test_parse_log_residual_history():
    info = of.parse_simple_foam_log(_LOG)
    assert set(info["residuals"]) == {"Ux", "Uy", "p", "nuTilda"}
    assert info["residuals"]["p"] == [1.0, 5e-06]
    assert info["final_residual"]["Ux"] == pytest.approx(0.002)


def test_parse_log_timings():
    info = of.parse_simple_foam_log(_LOG)
    assert info["execution_time"] == pytest.approx(0.62)
    assert info["clock_time"] == pytest.approx(2.0)


def test_parse_log_unconverged_run():
    truncated = _LOG.replace("SIMPLE solution converged in 2 iterations", "")
    info = of.parse_simple_foam_log(truncated)
    assert info["converged"] is False
    assert info["converged_at"] is None
    assert info["iterations"] == 2


def test_parse_empty_log_is_not_a_crash():
    info = of.parse_simple_foam_log("")
    assert info["iterations"] == 0
    assert info["converged"] is False
    assert info["residuals"] == {}


# A real case runs `nNonOrthogonalCorrectors 2`, so pressure is solved three
# times per outer iteration. Flattening those into one list made the `p` history
# three times longer than `Ux`'s, and any metric that compares fields at a common
# index then read pressure from a third of the way back through the run.
_LOG_NON_ORTHOGONAL = """\
Time = 1

smoothSolver:  Solving for Ux, Initial residual = 1, Final residual = 0.05, No Iterations 3
smoothSolver:  Solving for Uy, Initial residual = 1, Final residual = 0.04, No Iterations 3
GAMG:  Solving for p, Initial residual = 1, Final residual = 0.01, No Iterations 8
GAMG:  Solving for p, Initial residual = 0.01, Final residual = 0.0001, No Iterations 5
GAMG:  Solving for p, Initial residual = 0.0001, Final residual = 1e-06, No Iterations 4
ExecutionTime = 0.31 s  ClockTime = 1 s

Time = 2

smoothSolver:  Solving for Ux, Initial residual = 0.002, Final residual = 1e-05, No Iterations 3
smoothSolver:  Solving for Uy, Initial residual = 0.003, Final residual = 2e-05, No Iterations 3
GAMG:  Solving for p, Initial residual = 0.004, Final residual = 4e-05, No Iterations 6
GAMG:  Solving for p, Initial residual = 4e-05, Final residual = 5e-07, No Iterations 4
GAMG:  Solving for p, Initial residual = 5e-07, Final residual = 8e-09, No Iterations 3
ExecutionTime = 0.62 s  ClockTime = 2 s

End
"""


def test_parse_log_keeps_one_entry_per_outer_iteration():
    info = of.parse_simple_foam_log(_LOG_NON_ORTHOGONAL)
    res = info["residuals"]
    assert len(res["p"]) == len(res["Ux"]) == 2
    # The corrector sub-solves (0.01, 1e-4, ...) are not outer-iteration values.
    assert res["p"] == [1.0, 0.004]
    assert info["final_residual"]["p"] == pytest.approx(0.004)


def test_iterations_to_threshold_is_not_fooled_by_correctors():
    res = of.parse_simple_foam_log(_LOG_NON_ORTHOGONAL)["residuals"]
    # Flattening put p's 1e-4 corrector at index 2, which would have declared
    # the run below 1e-2 at iteration 2 even though its pressure sits at 4e-3.
    assert of.iterations_to_threshold(res, 1e-2) == 2
    assert of.iterations_to_threshold(res, 1e-3) is None


def test_parse_log_pads_a_field_missing_from_a_block():
    # A run killed mid-iteration leaves a block with only the fields it reached.
    truncated = _LOG_NON_ORTHOGONAL[: _LOG_NON_ORTHOGONAL.index("Solving for Uy, Initial residual = 0.003")]
    res = of.parse_simple_foam_log(truncated)["residuals"]
    assert len(res["Ux"]) == len(res["Uy"]) == 2
    assert res["Uy"][-1] == float("inf")
    # inf never satisfies a threshold, so a partial iteration cannot be counted.
    assert of.iterations_to_threshold(res, 1e-1) is None


# --------------------------------------------------------------------------- #
# Threshold-based convergence metric
# --------------------------------------------------------------------------- #

# Shape taken from a real run: residuals fall, then stagnate at a nonzero floor.
_STAGNATING = {
    "Ux": [1.0, 0.5, 0.02, 9e-4, 6.2e-4, 6.2e-4, 6.2e-4],
    "Uy": [1.0, 0.6, 0.03, 8e-4, 6.1e-4, 6.1e-4, 6.1e-4],
    "p": [1.0, 0.9, 0.05, 2e-3, 1.0e-3, 1.0e-3, 1.0e-3],
}


def test_iterations_to_threshold_is_one_based():
    assert of.iterations_to_threshold(_STAGNATING, 1e-1) == 3
    assert of.iterations_to_threshold(_STAGNATING, 1e-2) == 4


def test_iterations_to_threshold_requires_every_field():
    """p lags U, so the answer is set by the slowest field, not the fastest."""
    assert of.iterations_to_threshold(_STAGNATING, 2e-3) == 4
    assert of.iterations_to_threshold({"Ux": _STAGNATING["Ux"]}, 2e-3, fields=("Ux",)) == 4


def test_iterations_to_threshold_below_the_floor_returns_none():
    """A threshold under the stagnation floor is unreachable -- refuse, don't lie."""
    assert of.iterations_to_threshold(_STAGNATING, 1e-6) is None


def test_iterations_to_threshold_on_empty_history():
    assert of.iterations_to_threshold({}, 1e-3) is None


def test_iterations_to_threshold_handles_ragged_histories():
    ragged = {"Ux": [1.0, 1e-4], "Uy": [1.0, 1e-4, 1e-4], "p": [1.0, 1e-4]}
    assert of.iterations_to_threshold(ragged, 1e-3) == 2


def test_residual_floor_reports_the_stagnation_level():
    floor = of.residual_floor(_STAGNATING, window=3)
    assert floor == pytest.approx(1.0e-3)  # p is the worst field


def test_residual_floor_on_empty_history_is_nan():
    assert np.isnan(of.residual_floor({}))


def test_result_exposes_the_metric():
    res = of.OpenFOAMResult(
        field=None, iterations=7, converged=False, wall_time=1.0, execution_time=1.0,
        start="cold", case_dir="x", residuals=_STAGNATING,
    )
    assert res.iterations_to(1e-2) == 4
    # The property uses the default 50-iteration window; this toy history is
    # only 7 long, so the median spans the descent as well as the plateau.
    assert res.residual_floor == pytest.approx(2.0e-3)
    assert of.residual_floor(res.residuals, window=3) == pytest.approx(1.0e-3)


# --------------------------------------------------------------------------- #
# End-to-end (needs a real OpenFOAM installation)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_solve_case_end_to_end(tmp_path, small_case):
    # Probed inside the body, not in a skipif: a skipif condition is evaluated at
    # collection time even when `slow` is deselected, and the probe spins up WSL.
    if not of.openfoam_available():
        pytest.skip("OpenFOAM/WSL not installed")
    res = of.solve_case(
        small_case, case_dir=str(tmp_path / "e2e"), n_iter=25, timeout=1800.0
    )
    assert res.field.shape == small_case.domain.shape
    assert res.iterations > 0
    assert res.start == "cold"
    # Solid cells are outside the mesh and stay at the scatter fill value.
    assert np.all(np.isfinite(res.field.u))
    assert res.meta["fluid_cells"] < small_case.domain.nx * small_case.domain.ny


# --------------------------------------------------------------------------- #
# Crash recovery
# --------------------------------------------------------------------------- #


def _fake_case(root, *, log_tail="End\n", time_dir="100"):
    os.makedirs(os.path.join(root, time_dir), exist_ok=True)
    open(os.path.join(root, time_dir, "U"), "w").close()
    with open(os.path.join(root, "log.simpleFoam"), "w", encoding="utf-8") as fh:
        fh.write(_LOG.replace("End\n", "") + log_tail)
    return root


def test_completed_run_reads_a_finished_solve(tmp_path):
    info = of.completed_run(_fake_case(str(tmp_path / "a")))
    assert info is not None
    assert info["reused"] is True
    assert info["iterations"] == 2


def test_completed_run_rejects_a_truncated_log(tmp_path):
    """A machine that lost power mid-solve leaves a log with no End marker."""
    assert of.completed_run(_fake_case(str(tmp_path / "b"), log_tail="Time = 3\n")) is None


def test_completed_run_rejects_a_case_with_no_written_step(tmp_path):
    root = str(tmp_path / "c")
    os.makedirs(os.path.join(root, "0"), exist_ok=True)
    open(os.path.join(root, "0", "U"), "w").close()
    with open(os.path.join(root, "log.simpleFoam"), "w", encoding="utf-8") as fh:
        fh.write(_LOG)
    assert of.completed_run(root) is None


def test_completed_run_rejects_an_unconverged_run_that_stopped_short(tmp_path):
    """Never mix budgets when resuming: 2 iterations cannot stand in for 500."""
    root = str(tmp_path / "d")
    os.makedirs(os.path.join(root, "100"), exist_ok=True)
    open(os.path.join(root, "100", "U"), "w").close()
    short = _LOG.replace("SIMPLE solution converged in 2 iterations", "")
    with open(os.path.join(root, "log.simpleFoam"), "w", encoding="utf-8") as fh:
        fh.write(short)
    assert of.completed_run(root, n_iter=2) is not None
    assert of.completed_run(root, n_iter=500) is None


def test_completed_run_keeps_a_converged_short_run(tmp_path):
    """Converged before the cap is complete, however short."""
    root = _fake_case(str(tmp_path / "e"))
    assert of.completed_run(root, n_iter=10_000) is not None


def test_completed_run_on_a_missing_case(tmp_path):
    assert of.completed_run(str(tmp_path / "nope")) is None


def test_completed_run_rejects_the_other_arm(tmp_path):
    """One directory reused across arms must not hand back the wrong result."""
    import json as _json

    root = _fake_case(str(tmp_path / "f"))
    with open(os.path.join(root, "neuroforge.json"), "w", encoding="utf-8") as fh:
        _json.dump({"start": "cold"}, fh)
    assert of.completed_run(root, start="cold") is not None
    assert of.completed_run(root, start="warm") is None


def test_completed_run_without_metadata_is_rejected_when_an_arm_is_required(tmp_path):
    root = _fake_case(str(tmp_path / "g"))
    assert of.completed_run(root) is not None          # no arm requested: fine
    assert of.completed_run(root, start="cold") is None  # cannot prove the arm


# --------------------------------------------------------------------------- #
# Force coefficients: a convergence metric that does not live on the floor
# --------------------------------------------------------------------------- #


def test_force_coeffs_directions_follow_the_freestream():
    text = of._force_coeffs(np.cos(np.pi / 6), np.sin(np.pi / 6), span=0.1)
    # 30 degrees: drag along the flow, lift perpendicular and to its left.
    assert "dragDir         (0.866025404 0.5 0)" in text
    assert "liftDir         (-0.5 0.866025404 0)" in text
    assert "magUInf         1" in text
    assert "Aref            0.1" in text        # chord 1 x span 0.1
    assert "rhoInf          1" in text          # pressure is kinematic


def test_control_dict_carries_the_function_object():
    text = of._control_dict(100, 100, functions=of._force_coeffs(1.0, 0.0, span=0.1))
    assert "endTime         100" in text
    assert "type            forceCoeffs;" in text
    # The functions block must come after the run controls, not inside them.
    assert text.index("endTime") < text.index("functions")


_COEFF_FILE = """\
# Force coefficients
# dragDir : (1 0 0)
# Time Cd Cd(f) Cd(r) Cl Cl(f) Cl(r) CmPitch
1 0.5 0.25 0.25 0.9 0.45 0.45 0.01
2 0.2 0.1 0.1 1.05 0.5 0.55 0.01
3 0.11 0.05 0.06 1.002 0.5 0.502 0.01
4 0.1005 0.05 0.0505 1.0 0.5 0.5 0.01
5 0.1 0.05 0.05 1.0 0.5 0.5 0.01
"""


def _write_coeffs(root, text=_COEFF_FILE, start="0"):
    d = os.path.join(root, "postProcessing", "forceCoeffs", start)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "coefficient.dat"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return root


def test_read_force_coeffs_uses_the_files_own_header(tmp_path):
    data = of.read_force_coeffs(_write_coeffs(str(tmp_path / "c")))
    assert data["Time"].tolist() == [1, 2, 3, 4, 5]
    assert data["Cd"][-1] == pytest.approx(0.1)
    assert data["Cl"][0] == pytest.approx(0.9)


def test_read_force_coeffs_on_a_run_without_the_function_object(tmp_path):
    (tmp_path / "bare").mkdir()
    assert of.read_force_coeffs(str(tmp_path / "bare")) == {}


def test_read_force_coeffs_concatenates_a_restart_keeping_the_later_value(tmp_path):
    root = _write_coeffs(str(tmp_path / "r"))
    _write_coeffs(root, "# Time Cd Cl\n5 0.9 9\n6 0.1 1\n", start="5")
    data = of.read_force_coeffs(root)
    assert data["Time"].tolist() == [1, 2, 3, 4, 5, 6]
    assert data["Cd"][4] == pytest.approx(0.9)   # the restart's value at t=5 wins


def test_iterations_to_force_band_finds_where_it_settles(tmp_path):
    data = of.read_force_coeffs(_write_coeffs(str(tmp_path / "b")))
    # Cd reaches 0.1005 at t=4, which is within 1% of the final 0.1.
    assert of.iterations_to_force_band(data["Time"], data["Cd"], tol=0.01) == 4
    assert of.iterations_to_force_band(data["Time"], data["Cd"], tol=0.2) == 3


def test_iterations_to_force_band_requires_it_to_stay():
    time = np.arange(1, 8)
    # Sweeps through the final value at t=2 on its way past, then comes back.
    values = np.array([2.0, 1.0, 0.4, 0.7, 0.95, 1.0, 1.0])
    assert of.iterations_to_force_band(time, values, tol=0.01) == 6


def test_iterations_to_force_band_on_a_run_that_never_settles():
    time = np.arange(1, 6)
    assert of.iterations_to_force_band(time, np.array([1.0, 2.0, 1.0, 2.0, 1.0]),
                                       tol=0.01, reference=10.0) is None


def test_iterations_to_force_band_takes_a_shared_reference():
    time = np.arange(1, 6)
    values = np.array([5.0, 2.0, 1.02, 1.01, 1.0])
    # Against its own final it settles at t=3; against a target of 2.0 it never does.
    assert of.iterations_to_force_band(time, values, tol=0.05) == 3
    assert of.iterations_to_force_band(time, values, tol=0.05, reference=2.0) is None


def test_iterations_to_force_band_on_empty_input():
    assert of.iterations_to_force_band(np.array([]), np.array([])) is None


# --------------------------------------------------------------------------- #
# Under-relaxation
# --------------------------------------------------------------------------- #


def test_written_cases_use_the_relaxation_that_does_not_stall(tmp_path, small_case):
    """0.9 is SIMPLEC's usual pairing and it limit-cycles here; see RELAX_U."""
    assert (of.RELAX_U, of.RELAX_NUT) == (0.7, 0.4)
    d = of.write_case(small_case, str(tmp_path / "relax"), n_iter=10)
    text = open(os.path.join(d, "system", "fvSolution"), encoding="utf-8").read()
    assert "U               0.7;" in text
    # nuTilda is relaxed harder: it is the laggard once momentum is relaxed.
    assert "nuTilda         0.4;" in text
    assert "consistent      yes;" in text


def test_fv_solution_relaxation_is_overridable():
    text = of._fv_solution(1e-6, 1e-7, n_non_orth=2, relax=0.5, relax_nut=0.3)
    assert "U               0.5;" in text
    assert "nuTilda         0.3;" in text
