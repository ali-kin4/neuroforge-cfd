"""OpenFOAM (WSL2) backend: full-case, warm-startable classical RANS solves.

This module drives a real steady incompressible RANS solve (``simpleFoam`` with
the Spalart-Allmaras closure -- the AirfRANS solver) on the *whole* case, from a
Windows/WSL2 host. Its purpose is the Paper-2 experiment: compare a **cold**
start (uniform freestream) against a **warm** start seeded from a NeuroForge
prediction, and measure iterations-to-convergence and wall-clock.

Why full-case and not a region patch
------------------------------------
:class:`~neuroforge.solver.fallback.ClassicalFallback` patches a flagged
sub-region using the surrounding prediction as Dirichlet data. That is a local
boundary-value problem with *approximate* boundary data, and
``scripts/probe_patch_acceptance.py`` measured that it never lowers true field
error (0/140 trials, and 0 again when handed exact boundary data) -- a better
solver inside the box cannot fix wrong data on the box border. A full-case solve
has no such seam: the steady solution is fixed by the *physical* boundary
conditions, so it converges to the same answer from any initial guess. The
prediction therefore buys **cost**, not accuracy, which is exactly the claim
``docs/ROADMAP_paper2.md`` targets.

Mesh
----
The case mesh is a uniform 2-D block over :attr:`Domain.bounds`, built so that
**cell centres coincide exactly with the NeuroForge grid points** (the block is
extended by half a cell on each side and divided into ``nx x ny`` cells). Cells
inside the body are then removed with ``topoSet`` + ``subsetMesh``, exposing an
``airfoil`` wall patch. Because the fields are written *before* the subset,
``subsetMesh`` maps them for us; a marker field ``cellId`` rides along so the
grid<->mesh correspondence is read back from the mesh rather than assumed.

This is deliberately not a body-fitted C-grid: it is the cheapest mesh on which
warm and cold starts are compared *on identical footing*, which is all the
warm-start claim needs. A snappyHexMesh path can replace :func:`write_case`
later without touching the runner or the log parser.

The module imports fine with no WSL and no OpenFOAM installed; every entry point
degrades to a clear :class:`OpenFOAMUnavailable`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field as _dc_field

import numpy as np

from neuroforge.core.types import DTYPE, FlowCase, FlowField

__all__ = [
    "OpenFOAMUnavailable",
    "OpenFOAMEnv",
    "OpenFOAMResult",
    "to_wsl_path",
    "list_wsl_distros",
    "detect_openfoam",
    "openfoam_available",
    "require_openfoam",
    "run_openfoam",
    "write_case",
    "mesh_case",
    "run_simple_foam",
    "parse_simple_foam_log",
    "iterations_to_threshold",
    "residual_floor",
    "read_volfield",
    "check_solid_region",
    "solve_case",
]

# Environment overrides (both optional).
ENV_DISTRO = "NEUROFORGE_WSL_DISTRO"
ENV_BASHRC = "NEUROFORGE_OPENFOAM_BASHRC"

# Spalart-Allmaras freestream ratio nuTilda_inf / nu -- the standard "3 to 5"
# range; 3 is the usual low-turbulence external-aero choice.
NUTILDA_FREESTREAM_RATIO = 3.0

_BASHRC_PATTERNS = (
    "/usr/lib/openfoam/openfoam*/etc/bashrc",
    "/opt/openfoam*/etc/bashrc",
    "/usr/lib/openfoam*/etc/bashrc",
    "$HOME/OpenFOAM/OpenFOAM-*/etc/bashrc",
)


class OpenFOAMUnavailable(RuntimeError):
    """Raised when WSL or an OpenFOAM installation could not be located."""


# --------------------------------------------------------------------------- #
# WSL plumbing
# --------------------------------------------------------------------------- #


def to_wsl_path(path: str | os.PathLike) -> str:
    """Translate a Windows path to its ``/mnt/<drive>`` WSL equivalent.

    POSIX-looking paths pass through unchanged, so this is a no-op when the
    package runs natively on Linux.
    """
    text = str(path)
    if not text:
        return text
    text = text.replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", text)
    if m:
        drive, rest = m.groups()
        return f"/mnt/{drive.lower()}/{rest}"
    return text


def _wsl_exe() -> str | None:
    """Path to ``wsl.exe``, or ``None`` when not running under Windows."""
    return shutil.which("wsl.exe") or shutil.which("wsl")


def list_wsl_distros() -> list[str]:
    """Installed WSL distribution names (empty list when WSL is absent)."""
    exe = _wsl_exe()
    if exe is None:
        return []
    try:
        out = subprocess.run([exe, "-l", "-q"], capture_output=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    # wsl.exe emits UTF-16LE; decode leniently and strip NULs either way.
    try:
        text = out.decode("utf-16-le")
    except UnicodeDecodeError:
        text = out.decode("utf-8", "replace")
    return [ln.strip() for ln in text.replace("\x00", "").splitlines() if ln.strip()]


@dataclass(frozen=True)
class OpenFOAMEnv:
    """A located OpenFOAM installation reachable from this host."""

    distro: str | None          # WSL distro name; None when running natively
    bashrc: str                 # POSIX path to the OpenFOAM etc/bashrc
    version: str = "unknown"    # release tag parsed from the install path
    native: bool = False        # True when invoked without wsl.exe

    @property
    def available(self) -> bool:
        return bool(self.bashrc)


_ENV_CACHE: dict[str | None, "OpenFOAMEnv | None"] = {}


def _shell_prefix(env: OpenFOAMEnv) -> str:
    """``source``-the-OpenFOAM-environment prefix for a login-less ``bash -c``."""
    # OpenFOAM's bashrc is chatty and on some builds returns non-zero; silence it
    # and swallow its status so the real command decides the exit code.
    return f'source "{env.bashrc}" >/dev/null 2>&1 || true; '


def _run_bash(
    env: OpenFOAMEnv, script: str, timeout: float | None
) -> subprocess.CompletedProcess:
    """Run ``script`` in bash, inside the WSL distro when there is one."""
    if env.native:
        argv = ["bash", "-c", script]
    else:
        exe = _wsl_exe()
        if exe is None:
            raise OpenFOAMUnavailable("wsl.exe not found on PATH")
        argv = [exe, "-d", env.distro or "Ubuntu", "--", "bash", "-c", script]
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, errors="replace"
    )


def _version_from_bashrc(bashrc: str) -> str:
    """Release tag from the install path, e.g. ``.../openfoam2606/etc/bashrc``.

    ``$WM_PROJECT_VERSION`` is not readable back through ``wsl.exe`` (see the
    note in :func:`detect_openfoam`), and the path carries the same information.
    """
    m = re.search(r"(?:openfoam|OpenFOAM-)([\w.+-]+?)/etc/bashrc$", bashrc)
    return m.group(1) if m else "unknown"


def detect_openfoam(distro: str | None = None, refresh: bool = False) -> OpenFOAMEnv | None:
    """Locate an OpenFOAM installation, or return ``None``.

    Search order: ``$NEUROFORGE_OPENFOAM_BASHRC`` if set, then the usual ESI /
    Foundation / source-build prefixes, newest version last. On Windows the
    search runs inside ``$NEUROFORGE_WSL_DISTRO`` (or the WSL default); on Linux
    it runs natively. Cached per distro -- pass ``refresh=True`` after
    installing OpenFOAM.
    """
    distro = distro or os.environ.get(ENV_DISTRO) or None
    if not refresh and distro in _ENV_CACHE:
        return _ENV_CACHE[distro]

    native = _wsl_exe() is None
    probe = OpenFOAMEnv(distro=distro, bashrc="", native=native)

    override = os.environ.get(ENV_BASHRC)
    if override:
        find_cmd = f'ls -d "{override}" 2>/dev/null'
    else:
        # NB: no shell variables. Argument passing through wsl.exe drops
        # user-defined variable references ("$f" arrives empty), so a
        # `for f in ...; do ... "$f"` probe silently finds nothing. Every
        # command this module builds must be variable-free for that reason.
        globs = " ".join(_BASHRC_PATTERNS)
        find_cmd = f"ls -d {globs} 2>/dev/null | sort -V | tail -1"

    try:
        found = _run_bash(probe, find_cmd, timeout=300)
    except (OSError, subprocess.SubprocessError, OpenFOAMUnavailable):
        _ENV_CACHE[distro] = None
        return None

    lines = [ln.strip() for ln in (found.stdout or "").splitlines() if ln.strip()]
    bashrc = lines[-1] if lines else ""
    if not bashrc:
        _ENV_CACHE[distro] = None
        return None

    env = OpenFOAMEnv(distro=distro, bashrc=bashrc, native=native)
    try:
        ver = _run_bash(env, _shell_prefix(env) + "command -v simpleFoam", timeout=300)
    except (OSError, subprocess.SubprocessError):
        _ENV_CACHE[distro] = None
        return None

    out = [ln.strip() for ln in (ver.stdout or "").splitlines() if ln.strip()]
    if not any(ln.endswith("simpleFoam") for ln in out):
        _ENV_CACHE[distro] = None
        return None

    env = OpenFOAMEnv(
        distro=distro, bashrc=bashrc, version=_version_from_bashrc(bashrc), native=native
    )
    _ENV_CACHE[distro] = env
    return env


def openfoam_available(distro: str | None = None) -> bool:
    """``True`` when a usable ``simpleFoam`` was found. Never raises."""
    return detect_openfoam(distro) is not None


def require_openfoam(distro: str | None = None) -> OpenFOAMEnv:
    """Return the located environment, or raise with install guidance."""
    env = detect_openfoam(distro)
    if env is not None:
        return env
    distros = list_wsl_distros()
    where = f"WSL distros: {distros}" if distros else "no WSL distributions found"
    raise OpenFOAMUnavailable(
        f"OpenFOAM not found ({where}). Install it in WSL2 with:\n"
        "  curl -fsSL https://dl.openfoam.com/add-debian-repo.sh | sudo bash\n"
        "  sudo apt-get update && sudo apt-get install -y openfoam2506-default\n"
        f"Then set ${ENV_DISTRO} / ${ENV_BASHRC} if it lives somewhere unusual."
    )


def run_openfoam(
    command: str,
    case_dir: str,
    *,
    distro: str | None = None,
    timeout: float | None = 3600.0,
    log_name: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run ``command`` with the OpenFOAM environment sourced, inside ``case_dir``.

    ``case_dir`` is a host path and is translated to WSL form. When ``log_name``
    is given the combined output is also written to ``<case_dir>/<log_name>``.
    """
    env = require_openfoam(distro)
    wsl_case = to_wsl_path(os.path.abspath(case_dir))
    script = _shell_prefix(env) + f'cd "{wsl_case}" || exit 3; ' + command
    proc = _run_bash(env, script, timeout=timeout)
    if log_name:
        with open(os.path.join(case_dir, log_name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(proc.stdout or "")
            if proc.stderr:
                fh.write("\n--- stderr ---\n")
                fh.write(proc.stderr)
    if check and proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-25:])
        raise RuntimeError(
            f"OpenFOAM command failed (exit {proc.returncode}): {command}\n"
            f"--- last output ---\n{tail}\n"
            f"--- stderr ---\n{(proc.stderr or '')[-2000:]}"
        )
    return proc


# --------------------------------------------------------------------------- #
# Case writing
# --------------------------------------------------------------------------- #

_BANNER = """\
/*--------------------------------*- C++ -*----------------------------------*\\
|  Written by neuroforge.solver.openfoam -- do not edit by hand.              |
\\*---------------------------------------------------------------------------*/
"""


def _header(cls: str, obj: str, location: str | None = None) -> str:
    loc = f'    location    "{location}";\n' if location else ""
    return (
        _BANNER
        + "FoamFile\n{\n"
        + "    version     2.0;\n    format      ascii;\n"
        + f"    class       {cls};\n{loc}"
        + f"    object      {obj};\n}}\n"
        + "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
    )


def _num(x: float) -> str:
    """Format a float with enough digits to round-trip float32 exactly."""
    return f"{float(x):.9g}"


def _write(path: str, text: str) -> None:
    """Write ``text`` with LF endings (OpenFOAM chokes on CRLF in dicts)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _scalar_list(values: np.ndarray) -> str:
    """``nonuniform List<scalar>`` body."""
    vals = np.asarray(values, dtype=np.float64).ravel()
    body = "\n".join(_num(v) for v in vals)
    return f"nonuniform List<scalar>\n{vals.size}\n(\n{body}\n)\n"


def _vector_list(ux: np.ndarray, uy: np.ndarray) -> str:
    """``nonuniform List<vector>`` body for a 2-D field (w = 0)."""
    a = np.asarray(ux, dtype=np.float64).ravel()
    b = np.asarray(uy, dtype=np.float64).ravel()
    body = "\n".join(f"({_num(p)} {_num(q)} 0)" for p, q in zip(a, b))
    return f"nonuniform List<vector>\n{a.size}\n(\n{body}\n)\n"


def _boundary_field(entries: dict[str, str]) -> str:
    out = ["boundaryField\n{"]
    for name, spec in entries.items():
        out.append(f"    {name}\n    {{\n{spec}    }}")
    out.append("}\n")
    return "\n".join(out)


def _freestream(case: FlowCase) -> tuple[float, float]:
    """Inlet velocity components from ``u_inf`` and angle of attack."""
    aoa = np.deg2rad(float(case.bc.aoa_deg))
    u = float(case.bc.u_inf) * float(np.cos(aoa))
    v = float(case.bc.u_inf) * float(np.sin(aoa))
    return u, v


def _solid_mask(case: FlowCase, field: FlowField | None) -> np.ndarray:
    """Boolean ``(ny, nx)`` map of cells inside the body."""
    ny, nx = case.domain.shape
    if field is not None and field.mask is not None:
        return np.asarray(field.mask) < 0.5
    from neuroforge.geometry.sdf import solid_mask

    # NB: geometry.solid_mask is a *fluid* indicator -- 1.0 in the fluid, 0.0
    # inside the body (see its docstring). Same convention as FlowField.mask.
    return np.asarray(solid_mask(case.geometry, case.domain)) < 0.5


def check_solid_region(solid: np.ndarray) -> dict:
    """Sanity-check the rasterised body before it becomes a mesh cut-out.

    At low resolution a NACA section is thinner than a cell near the trailing
    edge, so the solid set can break into disconnected specks or develop a slit
    the flow leaks through. Neither is caught by "are there any solid cells"; both
    make the resulting mesh a different geometry from the one the case names.

    Returns a report dict and emits a :class:`UserWarning` for each problem --
    a warning rather than an error, so a deliberately coarse run still proceeds.
    """
    import warnings

    solid = np.asarray(solid) > 0.5
    rows = np.flatnonzero(solid.any(axis=1))
    cols = np.flatnonzero(solid.any(axis=0))
    report = {
        "solid_cells": int(solid.sum()),
        "bbox_cells": [int(rows.size), int(cols.size)],
        "components": 1,
        "max_thickness": 0,
        "thin_column_fraction": 1.0,
    }
    if report["solid_cells"] == 0:
        return report

    # An airfoil always tapers to a single cell at the trailing edge, so the
    # *thinnest* column is 1 at every resolution and says nothing. The thickest
    # column is the resolution indicator; the fraction of one-cell columns says
    # how much of the body is a fin the solver will see as a slit.
    counts = solid.sum(axis=0)
    nonzero = counts[counts > 0]
    report["max_thickness"] = int(nonzero.max())
    report["thin_column_fraction"] = float((nonzero < 2).mean())

    try:
        from scipy import ndimage

        _labels, n = ndimage.label(solid)
        report["components"] = int(n)
    except ImportError:  # pragma: no cover - scipy is a hard dep in practice
        pass

    if report["components"] > 1:
        warnings.warn(
            f"rasterised body has {report['components']} disconnected components at this "
            f"resolution; the mesh cut-out will not be the named airfoil",
            UserWarning,
            stacklevel=3,
        )
    if report["max_thickness"] < 4:
        warnings.warn(
            f"rasterised body is at most {report['max_thickness']} cells thick; the mesh "
            f"cut-out is a coarse staircase. Use resolution >= 128 for a NACA section on "
            f"the default 3-chord domain.",
            UserWarning,
            stacklevel=3,
        )
    return report


def _block_mesh_dict(case: FlowCase) -> str:
    """Uniform 2-D block whose cell centres land on the NeuroForge grid points."""
    dom = case.domain
    xmin, xmax, ymin, ymax = (float(b) for b in dom.bounds)
    # Extend by half a cell so the nx x ny cell centres coincide with the grid.
    hx, hy = 0.5 * dom.dx, 0.5 * dom.dy
    x0, x1 = xmin - hx, xmax + hx
    y0, y1 = ymin - hy, ymax + hy
    z0, z1 = 0.0, float(dom.dx)  # unit-ish span; 2-D via `empty` front/back

    v = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    verts = "\n".join(f"    ({_num(a)} {_num(b)} {_num(c)})" for a, b, c in v)
    return (
        _header("dictionary", "blockMeshDict", "system")
        + "scale   1;\n\nvertices\n(\n"
        + verts
        + "\n);\n\nblocks\n(\n"
        + f"    hex (0 1 2 3 4 5 6 7) ({dom.nx} {dom.ny} 1) simpleGrading (1 1 1)\n"
        + ");\n\nedges\n(\n);\n\nboundary\n(\n"
        + "    inlet\n    {\n        type patch;\n"
        + "        faces\n        (\n            (0 4 7 3)\n            (1 5 4 0)\n"
        + "            (3 7 6 2)\n        );\n    }\n"
        + "    outlet\n    {\n        type patch;\n"
        + "        faces\n        (\n            (2 6 5 1)\n        );\n    }\n"
        # Declared empty here; subsetMesh moves the exposed body faces into it.
        + "    airfoil\n    {\n        type wall;\n        faces\n        (\n        );\n    }\n"
        + "    frontAndBack\n    {\n        type empty;\n"
        + "        faces\n        (\n            (0 3 2 1)\n            (4 5 6 7)\n"
        + "        );\n    }\n"
        + ");\n\nmergePatchPairs\n(\n);\n"
    )


def _topo_set_dict(fluid_ids: np.ndarray) -> str:
    """Select the fluid cells by explicit label, for ``subsetMesh``."""
    ids = "\n".join(str(int(i)) for i in fluid_ids)
    return (
        _header("dictionary", "topoSetDict", "system")
        + "actions\n(\n    {\n        name    fluidCells;\n"
        + "        type    cellSet;\n        action  new;\n"
        + "        source  labelToCell;\n"
        + f"        value\n        (\n{ids}\n        );\n"
        + "    }\n);\n"
    )


def _control_dict(n_iter: int, write_interval: int) -> str:
    return (
        _header("dictionary", "controlDict", "system")
        + "application     simpleFoam;\n"
        + "startFrom       startTime;\nstartTime       0;\n"
        + "stopAt          endTime;\n"
        + f"endTime         {int(n_iter)};\n"
        + "deltaT          1;\nwriteControl    timeStep;\n"
        + f"writeInterval   {int(write_interval)};\n"
        + "purgeWrite      0;\nwriteFormat     ascii;\nwritePrecision  10;\n"
        + "writeCompression off;\ntimeFormat      general;\ntimePrecision   6;\n"
        + "runTimeModifiable false;\n"
    )


_FV_SCHEMES = (
    "ddtSchemes\n{\n    default         steadyState;\n}\n\n"
    "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
    "divSchemes\n{\n    default         none;\n"
    "    div(phi,U)      bounded Gauss linearUpwind grad(U);\n"
    "    div(phi,nuTilda) bounded Gauss upwind;\n"
    "    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n}\n\n"
    "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
    "interpolationSchemes\n{\n    default         linear;\n}\n\n"
    "snGradSchemes\n{\n    default         corrected;\n}\n\n"
    "wallDist\n{\n    method          meshWave;\n}\n"
)


def _fv_solution(tol_p: float, tol_u: float, n_non_orth: int = 0) -> str:
    return (
        _header("dictionary", "fvSolution", "system")
        + "solvers\n{\n"
        + "    p\n    {\n        solver          GAMG;\n"
        + "        smoother        GaussSeidel;\n"
        + "        tolerance       1e-09;\n        relTol          0.01;\n    }\n\n"
        + '    "(U|nuTilda)"\n    {\n        solver          smoothSolver;\n'
        + "        smoother        symGaussSeidel;\n"
        + "        tolerance       1e-09;\n        relTol          0.1;\n    }\n}\n\n"
        + "SIMPLE\n{\n"
        + f"    nNonOrthogonalCorrectors {int(n_non_orth)};\n"
        + "    consistent      yes;\n"
        + "    residualControl\n    {\n"
        + f"        p               {_num(tol_p)};\n"
        + f"        U               {_num(tol_u)};\n"
        + f"        nuTilda         {_num(tol_u)};\n    }}\n}}\n\n"
        + "relaxationFactors\n{\n    equations\n    {\n"
        + "        U               0.9;\n        nuTilda         0.9;\n    }\n}\n"
    )


def write_case(
    case: FlowCase,
    case_dir: str,
    *,
    initial: FlowField | None = None,
    n_iter: int = 3000,
    tol_p: float = 1e-5,
    tol_u: float = 1e-6,
) -> str:
    """Write a complete ``simpleFoam`` case directory for ``case``.

    Parameters
    ----------
    case : FlowCase
        Geometry, boundary conditions, fluid properties and grid.
    case_dir : str
        Host directory to create (overwritten if it exists).
    initial : FlowField, optional
        Warm start. When given, its ``u, v, p, nut`` are written as the
        ``0/`` internal fields (``nuTilda`` is seeded from ``nut``, whose SA
        inverse is monotone and well approximated by ``nut`` itself away from
        the wall). When ``None`` the case is a **cold** start: uniform
        freestream velocity, zero pressure, freestream ``nuTilda``.
    n_iter : int
        SIMPLE iteration cap (``endTime``).
    tol_p, tol_u
        ``residualControl`` thresholds that define convergence.

    Returns
    -------
    str
        ``case_dir``, for chaining.
    """
    dom = case.domain
    ny, nx = dom.shape
    n_cells = nx * ny
    u_inf, v_inf = _freestream(case)
    nu = float(case.fluid.kinematic_viscosity)
    nut_inf = NUTILDA_FREESTREAM_RATIO * nu

    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(case_dir, exist_ok=True)

    solid = _solid_mask(case, initial)
    if solid.shape != (ny, nx):
        raise ValueError(f"solid mask {solid.shape} does not match domain {(ny, nx)}")
    fluid_ids = np.flatnonzero(~solid.ravel())
    if fluid_ids.size == 0:
        raise ValueError("no fluid cells: the body fills the whole domain")
    if fluid_ids.size == n_cells:
        raise ValueError("no solid cells: the body does not intersect the domain grid")
    geom_report = check_solid_region(solid)

    # --- system/ ---------------------------------------------------------- #
    _write(os.path.join(case_dir, "system", "controlDict"),
           _control_dict(n_iter, n_iter))
    _write(os.path.join(case_dir, "system", "fvSchemes"),
           _header("dictionary", "fvSchemes", "system") + _FV_SCHEMES)
    _write(os.path.join(case_dir, "system", "fvSolution"), _fv_solution(tol_p, tol_u))
    _write(os.path.join(case_dir, "system", "blockMeshDict"), _block_mesh_dict(case))
    _write(os.path.join(case_dir, "system", "topoSetDict"), _topo_set_dict(fluid_ids))

    # --- constant/ -------------------------------------------------------- #
    _write(
        os.path.join(case_dir, "constant", "transportProperties"),
        _header("dictionary", "transportProperties", "constant")
        + f"transportModel  Newtonian;\nnu              {_num(nu)};\n",
    )
    turb = (
        "simulationType  RAS;\n\nRAS\n{\n    RASModel        SpalartAllmaras;\n"
        "    turbulence      on;\n    printCoeffs     on;\n}\n"
    )
    # ESI reads turbulenceProperties, Foundation v11+ reads momentumTransport.
    # Writing both keeps the case portable across forks; the unused one is inert.
    _write(os.path.join(case_dir, "constant", "turbulenceProperties"),
           _header("dictionary", "turbulenceProperties", "constant") + turb)
    _write(os.path.join(case_dir, "constant", "momentumTransport"),
           _header("dictionary", "momentumTransport", "constant") + turb)

    # --- 0/ (written on the FULL block; subsetMesh maps them) -------------- #
    if initial is not None:
        if tuple(initial.shape) != (ny, nx):
            raise ValueError(
                f"warm-start field {tuple(initial.shape)} does not match domain {(ny, nx)}"
            )
        u_int = _vector_list(initial.u, initial.v)
        p_int = _scalar_list(initial.p)
        nut_src = initial.nut if initial.nut is not None else np.full((ny, nx), nut_inf)
        nut_arr = np.maximum(np.asarray(nut_src, dtype=np.float64), 0.0)
        nut_int = _scalar_list(nut_arr)
        # SA transports nuTilda; for nut/nu >> 1 the two coincide, and near the
        # wall the fv damping makes nuTilda >= nut. Seeding nuTilda := nut is the
        # standard warm-start approximation and is bounded below by freestream.
        nutilda_int = _scalar_list(np.maximum(nut_arr, nut_inf))
        start = "warm"
    else:
        u_int = f"uniform ({_num(u_inf)} {_num(v_inf)} 0)\n"
        p_int = "uniform 0\n"
        nut_int = f"uniform {_num(nut_inf)}\n"
        nutilda_int = f"uniform {_num(nut_inf)}\n"
        start = "cold"

    _write(
        os.path.join(case_dir, "0", "U"),
        _header("volVectorField", "U", "0")
        + "dimensions      [0 1 -1 0 0 0 0];\n\ninternalField   "
        + u_int
        + ";\n\n"
        + _boundary_field({
            "inlet": f"        type            fixedValue;\n"
                     f"        value           uniform ({_num(u_inf)} {_num(v_inf)} 0);\n",
            "outlet": "        type            zeroGradient;\n",
            "airfoil": "        type            noSlip;\n",
            "frontAndBack": "        type            empty;\n",
        }),
    )
    _write(
        os.path.join(case_dir, "0", "p"),
        _header("volScalarField", "p", "0")
        + "dimensions      [0 2 -2 0 0 0 0];\n\ninternalField   "
        + p_int
        + ";\n\n"
        + _boundary_field({
            "inlet": "        type            zeroGradient;\n",
            "outlet": "        type            fixedValue;\n"
                      "        value           uniform 0;\n",
            "airfoil": "        type            zeroGradient;\n",
            "frontAndBack": "        type            empty;\n",
        }),
    )
    _write(
        os.path.join(case_dir, "0", "nuTilda"),
        _header("volScalarField", "nuTilda", "0")
        + "dimensions      [0 2 -1 0 0 0 0];\n\ninternalField   "
        + nutilda_int
        + ";\n\n"
        + _boundary_field({
            "inlet": f"        type            fixedValue;\n"
                     f"        value           uniform {_num(nut_inf)};\n",
            "outlet": "        type            zeroGradient;\n",
            "airfoil": "        type            fixedValue;\n"
                       "        value           uniform 0;\n",
            "frontAndBack": "        type            empty;\n",
        }),
    )
    _write(
        os.path.join(case_dir, "0", "nut"),
        _header("volScalarField", "nut", "0")
        + "dimensions      [0 2 -1 0 0 0 0];\n\ninternalField   "
        + nut_int
        + ";\n\n"
        + _boundary_field({
            "inlet": "        type            calculated;\n"
                     f"        value           uniform {_num(nut_inf)};\n",
            "outlet": "        type            calculated;\n"
                      f"        value           uniform {_num(nut_inf)};\n",
            "airfoil": "        type            nutUSpaldingWallFunction;\n"
                       "        value           uniform 0;\n",
            "frontAndBack": "        type            empty;\n",
        }),
    )
    # Marker field: rides through subsetMesh so the mesh->grid map is *read*,
    # never assumed. Values are flat (j * nx + i) indices into the (ny, nx) grid.
    _write(
        os.path.join(case_dir, "0", "cellId"),
        _header("volScalarField", "cellId", "0")
        + "dimensions      [0 0 0 0 0 0 0];\n\ninternalField   "
        + _scalar_list(np.arange(n_cells, dtype=np.float64))
        + ";\n\n"
        + _boundary_field({
            "inlet": "        type            zeroGradient;\n",
            "outlet": "        type            zeroGradient;\n",
            "airfoil": "        type            zeroGradient;\n",
            "frontAndBack": "        type            empty;\n",
        }),
    )

    _write(
        os.path.join(case_dir, "neuroforge.json"),
        __import__("json").dumps(
            {
                "case": case.name,
                "start": start,
                "nx": nx, "ny": ny,
                "bounds": [float(b) for b in dom.bounds],
                "nu": nu, "u_inf": u_inf, "v_inf": v_inf,
                "nut_freestream": nut_inf,
                "n_iter": int(n_iter),
                "fluid_cells": int(fluid_ids.size),
                "solid_cells": int(n_cells - fluid_ids.size),
                "geometry_check": geom_report,
            },
            indent=2,
        )
        + "\n",
    )
    return case_dir


# --------------------------------------------------------------------------- #
# Field reading
# --------------------------------------------------------------------------- #

_INTERNAL_RE = re.compile(r"internalField\s+(.*?);", re.DOTALL)


def read_volfield(path: str) -> np.ndarray:
    """Parse the ``internalField`` of an ASCII OpenFOAM volField.

    Returns ``(n,)`` for a scalar field and ``(n, 3)`` for a vector field.
    Uniform fields are broadcast to the list length recorded in the file, or to
    a single element when the length is unknown.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    m = _INTERNAL_RE.search(text)
    if m is None:
        raise ValueError(f"no internalField entry in {path}")
    body = m.group(1).strip()

    if body.startswith("uniform"):
        vals = re.findall(r"[-+0-9.eE]+", body[len("uniform"):])
        arr = np.asarray([float(v) for v in vals], dtype=np.float64)
        return arr if arr.size != 3 else arr.reshape(1, 3)

    # nonuniform List<scalar|vector> N ( ... )
    open_paren = body.index("(")
    count = int(re.findall(r"(\d+)\s*$", body[:open_paren].strip())[-1])
    payload = body[open_paren + 1 : body.rindex(")")]
    is_vector = "List<vector>" in body[:open_paren]
    nums = np.fromstring(payload.replace("(", " ").replace(")", " "), sep=" ")
    if is_vector:
        return nums.reshape(count, 3)
    if nums.size != count:
        raise ValueError(f"{path}: expected {count} values, parsed {nums.size}")
    return nums


def _latest_time(case_dir: str) -> str | None:
    """Name of the largest **non-zero** numeric time directory holding a ``U``.

    ``0/`` is excluded on purpose. It always exists and, on a warm start, holds
    the neural prediction itself -- so accepting it would silently return the
    input as the "solve result" whenever ``simpleFoam`` failed to write a step.
    """
    times = []
    for entry in os.listdir(case_dir):
        try:
            t = float(entry)
        except ValueError:
            continue
        if t > 0.0 and os.path.isfile(os.path.join(case_dir, entry, "U")):
            times.append((t, entry))
    if not times:
        return None
    return max(times)[1]


def _scatter_to_grid(
    values: np.ndarray, cell_ids: np.ndarray, shape: tuple[int, int], fill: float = 0.0
) -> np.ndarray:
    """Place mesh-ordered ``values`` back on the ``(ny, nx)`` grid via ``cell_ids``."""
    out = np.full(int(shape[0]) * int(shape[1]), fill, dtype=np.float64)
    out[cell_ids] = values
    return out.reshape(shape).astype(DTYPE)


# --------------------------------------------------------------------------- #
# Log parsing
# --------------------------------------------------------------------------- #

_TIME_RE = re.compile(r"^Time = (\d+(?:\.\d+)?)\s*$", re.MULTILINE)
_RES_RE = re.compile(
    r"Solving for (\w+), Initial residual = ([-\d.eE+]+), "
    r"Final residual = ([-\d.eE+]+), No Iterations (\d+)"
)
_CONVERGED_RE = re.compile(r"SIMPLE solution converged in (\d+) iterations")
_CLOCK_RE = re.compile(r"ExecutionTime = ([\d.]+) s\s+ClockTime = (\d+) s")


def parse_simple_foam_log(text: str) -> dict:
    """Extract iteration count, convergence and the residual history from a log.

    Returns a dict with ``iterations`` (last ``Time`` reached), ``converged``
    (whether ``residualControl`` was satisfied), ``converged_at`` (the iteration
    OpenFOAM reported, or ``None``), ``execution_time`` / ``clock_time`` in
    seconds, and ``residuals``: ``{field: [initial residual per outer
    iteration]}`` for ``Ux, Uy, p, nuTilda``.
    """
    times = _TIME_RE.findall(text)
    iterations = int(times[-1]) if times else 0

    residuals: dict[str, list[float]] = {}
    for name, initial, _final, _n in _RES_RE.findall(text):
        residuals.setdefault(name, []).append(float(initial))

    conv = _CONVERGED_RE.search(text)
    clocks = _CLOCK_RE.findall(text)
    exec_t = float(clocks[-1][0]) if clocks else float("nan")
    clock_t = float(clocks[-1][1]) if clocks else float("nan")

    return {
        "iterations": iterations,
        "converged": conv is not None,
        "converged_at": int(conv.group(1)) if conv else None,
        "execution_time": exec_t,
        "clock_time": clock_t,
        "residuals": residuals,
        "final_residual": {k: v[-1] for k, v in residuals.items() if v},
    }


def iterations_to_threshold(
    residuals: dict[str, list[float]],
    threshold: float,
    fields: tuple[str, ...] = ("Ux", "Uy", "p"),
) -> int | None:
    """First outer iteration where every named field's initial residual is below
    ``threshold``, or ``None`` if that never happens.

    This is the metric the warm-start experiment reports, and it exists because
    ``residualControl`` is not usable here. Steady SIMPLE stagnates at a nonzero
    residual floor set by the mesh and the unsteadiness of the flow -- measured
    on this uniform mesh at Re 1e4, ``Ux`` sits at 6.2e-4 and ``p`` at 1.0e-3,
    bit-identical from iteration 500 to 1500. A convergence *flag* therefore
    never fires and iterations-to-convergence is undefined, while
    iterations-to-a-reachable-threshold is well defined for both arms and is the
    standard way the warm-start literature reports this.

    Pick a threshold above the floor (see :func:`residual_floor`); a threshold
    below it returns ``None`` for every arm, which is a refusal to measure
    rather than a saving of zero.
    """
    present = [f for f in fields if residuals.get(f)]
    if not present:
        return None
    n = min(len(residuals[f]) for f in present)
    for i in range(n):
        if all(residuals[f][i] <= threshold for f in present):
            return i + 1  # OpenFOAM's Time counter is 1-based
    return None


def residual_floor(
    residuals: dict[str, list[float]],
    fields: tuple[str, ...] = ("Ux", "Uy", "p"),
    window: int = 50,
) -> float:
    """Stagnation level: the largest per-field median over the last ``window``
    iterations. Any useful threshold must sit above this.
    """
    vals = []
    for f in fields:
        hist = residuals.get(f) or []
        if hist:
            vals.append(float(np.median(hist[-window:])))
    return max(vals) if vals else float("nan")


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def completed_run(case_dir: str, *, n_iter: int | None = None) -> dict | None:
    """Parsed log of an already-finished solve in ``case_dir``, or ``None``.

    A solve writes ``log.simpleFoam`` and its time directories to disk as it
    goes, so a run interrupted by a power cut leaves everything except the
    in-flight case recoverable. This reads that back so an experiment can resume
    instead of re-solving, and so results can be reconstructed from a
    ``runs/`` tree alone.

    A run counts as complete only if the log ends with OpenFOAM's ``End`` marker
    *and* a time directory greater than zero holds a ``U`` field -- a truncated
    log from a machine that lost power does not qualify. When ``n_iter`` is
    given, a run that stopped short of it is rejected too, so a resumed
    experiment never mixes budgets.
    """
    log = os.path.join(case_dir, "log.simpleFoam")
    if not os.path.isfile(log):
        return None
    with open(log, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    if not text.rstrip().endswith("End"):
        return None
    if _latest_time(case_dir) is None:
        return None
    info = parse_simple_foam_log(text)
    if n_iter is not None and not info["converged"] and info["iterations"] < n_iter:
        return None
    info["wall_time"] = float("nan")  # not recoverable from the log
    info["reused"] = True
    return info


def mesh_case(case_dir: str, *, distro: str | None = None, timeout: float = 1800.0) -> None:
    """Build the mesh: ``blockMesh`` -> ``topoSet`` -> ``subsetMesh``.

    ``subsetMesh`` carries the ``0/`` fields (including the ``cellId`` marker)
    onto the carved mesh, so the warm start survives the body cut-out.
    """
    run_openfoam("blockMesh", case_dir, distro=distro, timeout=timeout, log_name="log.blockMesh")
    run_openfoam("topoSet", case_dir, distro=distro, timeout=timeout, log_name="log.topoSet")
    run_openfoam(
        "subsetMesh fluidCells -patch airfoil -overwrite",
        case_dir,
        distro=distro,
        timeout=timeout,
        log_name="log.subsetMesh",
    )


def run_simple_foam(
    case_dir: str, *, distro: str | None = None, timeout: float = 7200.0
) -> dict:
    """Run ``simpleFoam`` in ``case_dir`` and return the parsed log."""
    t0 = time.perf_counter()
    proc = run_openfoam(
        "simpleFoam", case_dir, distro=distro, timeout=timeout, log_name="log.simpleFoam"
    )
    info = parse_simple_foam_log(proc.stdout or "")
    info["wall_time"] = time.perf_counter() - t0
    return info


@dataclass
class OpenFOAMResult:
    """Outcome of a full-case ``simpleFoam`` solve."""

    field: FlowField
    iterations: int
    converged: bool
    wall_time: float               # host-measured seconds, includes WSL overhead
    execution_time: float          # OpenFOAM's own ExecutionTime
    start: str                     # 'warm' | 'cold'
    case_dir: str
    residuals: dict = _dc_field(default_factory=dict)
    meta: dict = _dc_field(default_factory=dict)

    def iterations_to(self, threshold: float) -> int | None:
        """Iterations to drive every momentum/pressure residual below ``threshold``."""
        return iterations_to_threshold(self.residuals, threshold)

    @property
    def residual_floor(self) -> float:
        """Stagnation level of this run; thresholds must sit above it."""
        return residual_floor(self.residuals)


def solve_case(
    case: FlowCase,
    *,
    initial: FlowField | None = None,
    case_dir: str | None = None,
    n_iter: int = 3000,
    tol_p: float = 1e-5,
    tol_u: float = 1e-6,
    distro: str | None = None,
    timeout: float = 7200.0,
) -> OpenFOAMResult:
    """Solve ``case`` with ``simpleFoam``, optionally warm-started.

    This is the Paper-2 entry point. Call it twice on the same case -- once with
    ``initial=None`` (cold) and once with the NeuroForge prediction (warm) -- and
    compare ``iterations`` and ``wall_time``. Both arms use the identical mesh,
    schemes, solver settings and convergence criteria, so the difference is
    attributable to the initial guess alone.

    ``case_dir`` defaults to ``runs/openfoam/<case name>_<start>``; it is
    overwritten. The directory is kept after the solve for inspection.
    """
    require_openfoam(distro)
    start = "warm" if initial is not None else "cold"
    if case_dir is None:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", case.name or "case")
        case_dir = os.path.join("runs", "openfoam", f"{safe}_{start}")
    case_dir = os.path.abspath(case_dir)

    write_case(case, case_dir, initial=initial, n_iter=n_iter, tol_p=tol_p, tol_u=tol_u)
    mesh_case(case_dir, distro=distro, timeout=timeout)
    info = run_simple_foam(case_dir, distro=distro, timeout=timeout)

    # Map the result back onto the NeuroForge grid using the marker field.
    cell_ids = read_volfield(os.path.join(case_dir, "0", "cellId")).astype(np.int64)
    latest = _latest_time(case_dir)
    if latest is None:
        raise RuntimeError(
            f"simpleFoam wrote no time directory > 0 in {case_dir}; the run produced "
            f"no solution (see log.simpleFoam). Reported iterations: {info['iterations']}."
        )

    shape = case.domain.shape
    U = read_volfield(os.path.join(case_dir, latest, "U"))
    p = read_volfield(os.path.join(case_dir, latest, "p"))
    nut_path = os.path.join(case_dir, latest, "nut")
    nut_vals = read_volfield(nut_path) if os.path.isfile(nut_path) else np.zeros(len(cell_ids))

    mask = np.zeros(int(shape[0]) * int(shape[1]), dtype=np.float64)
    mask[cell_ids] = 1.0
    field = FlowField(
        domain=case.domain,
        u=_scatter_to_grid(U[:, 0], cell_ids, shape),
        v=_scatter_to_grid(U[:, 1], cell_ids, shape),
        p=_scatter_to_grid(p, cell_ids, shape),
        nut=_scatter_to_grid(nut_vals, cell_ids, shape),
        mask=mask.reshape(shape).astype(DTYPE),
        meta={
            "source": "openfoam",
            "start": start,
            "iterations": info["iterations"],
            "converged": info["converged"],
        },
    )
    return OpenFOAMResult(
        field=field,
        iterations=int(info["iterations"]),
        converged=bool(info["converged"]),
        wall_time=float(info["wall_time"]),
        execution_time=float(info["execution_time"]),
        start=start,
        case_dir=case_dir,
        residuals=info["residuals"],
        meta={
            "converged_at": info["converged_at"],
            "final_residual": info["final_residual"],
            "time_dir": latest,
            "fluid_cells": int(cell_ids.size),
        },
    )
