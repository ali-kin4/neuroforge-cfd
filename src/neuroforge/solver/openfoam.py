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
    "read_patches",
    "running_solvers",
    "potential_flow_seed",
    "read_force_coeffs",
    "read_force_components",
    "iterations_to_force_band",
    "read_volfield",
    "completed_run",
    "check_solid_region",
    "solve_case",
]

# Environment overrides (both optional).
ENV_DISTRO = "NEUROFORGE_WSL_DISTRO"
ENV_BASHRC = "NEUROFORGE_OPENFOAM_BASHRC"

# Steady SIMPLEC under-relaxation. `consistent yes` is usually paired with 0.9,
# and 0.9 is what this module shipped -- but on the C-grid at Re 3e6 it puts the
# solve into a limit cycle that stalls at a residual of 1.1e-5 and never gets
# lower, however long it runs (measured: 4000 iterations end at 1.13e-5, and the
# arm seeded with the converged field itself sits on the same level from
# iteration 100 to 800). At 0.7 the same case reaches 1e-5 by iteration 327 and
# carries on to 1.9e-6, still falling. See `scripts/convergence_diagnostic.py`,
# which also rules out the inner linear tolerances, the convection scheme and the
# mesh's 218,987 aspect-ratio cells as the cause.
#
# Once momentum is relaxed, nuTilda becomes the laggard -- at 0.7 it sits ten
# times above Ux and holds the solve up in turn. Relaxing it further to 0.4 is
# what gets the case past 1e-6: it is the only variant tried that reaches that
# level at all (iteration 1305), and it ends at Ux 3.5e-7, thirty times below
# the original floor. Its force coefficients settle correspondingly early --
# Cd within 0.01% of its converged value by iteration 2000, against 0.3% for
# uniform 0.7 relaxation.
#
# This matters for more than tidiness: an iteration-to-threshold metric only
# measures a convergence rate while the residual is still moving, so a false
# floor silently turns warm-start savings into noise.
RELAX_U = 0.7
RELAX_NUT = 0.4

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
    is given, output is redirected to ``<case_dir>/<log_name>`` **inside WSL** so
    the file grows while the solver runs, rather than being written in one go at
    the end. That is what lets a progress monitor read residuals from a solve in
    flight, and it means a run cut short by a power failure leaves a partial log
    instead of none. The text is read back afterwards so callers still receive it
    on ``.stdout``.
    """
    env = require_openfoam(distro)
    wsl_case = to_wsl_path(os.path.abspath(case_dir))
    redirect = f' > "{log_name}" 2>&1' if log_name else ""
    script = _shell_prefix(env) + f'cd "{wsl_case}" || exit 3; ' + command + redirect
    proc = _run_bash(env, script, timeout=timeout)

    if log_name:
        path = os.path.join(case_dir, log_name)
        text = ""
        if os.path.isfile(path):
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        if proc.stderr:  # anything the redirect did not catch (e.g. the shell itself)
            with open(path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write("\n--- stderr ---\n" + proc.stderr)
            text += "\n--- stderr ---\n" + proc.stderr
        proc = subprocess.CompletedProcess(proc.args, proc.returncode, text, proc.stderr)

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


def _control_dict(n_iter: int, write_interval: int, functions: str = "") -> str:
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
        + functions
    )


def _force_coeffs(
    u_inf: float, v_inf: float, *, span: float, patch: str = "airfoil", chord: float = 1.0
) -> str:
    """A ``forceCoeffs`` function object writing Cd and Cl every iteration.

    Iteration counts taken off the residual history are only meaningful while
    the residual is still falling; on this mesh it stagnates near 1e-5 and the
    deepest thresholds end up reading where a flat curve crosses a line. The
    force coefficients do not have that problem -- they settle onto a value and
    stay there -- so ``iterations_to_force_band`` measures convergence of the
    quantity an aerodynamicist actually wants, which is also how the warm-start
    literature reports it.

    Lift and drag directions come from the freestream, so the coefficients are
    resolved about the true flow direction rather than the chord line. ``rhoInf``
    is 1 because the solver's pressure is kinematic (p/rho), and ``Aref`` is
    ``chord * span`` for the one-cell-thick 2-D mesh.
    """
    speed = float(np.hypot(u_inf, v_inf))
    dx, dy = (u_inf / speed, v_inf / speed) if speed > 0 else (1.0, 0.0)
    return (
        "\nfunctions\n{\n    forceCoeffs\n    {\n"
        "        type            forceCoeffs;\n"
        '        libs            ("libforces.so");\n'
        "        writeControl    timeStep;\n        writeInterval   1;\n"
        "        log             no;\n"
        f"        patches         ({patch});\n"
        "        rho             rhoInf;\n        rhoInf          1;\n"
        f"        magUInf         {_num(speed)};\n"
        f"        lRef            {_num(chord)};\n"
        f"        Aref            {_num(chord * span)};\n"
        f"        dragDir         ({_num(dx)} {_num(dy)} 0);\n"
        f"        liftDir         ({_num(-dy)} {_num(dx)} 0);\n"
        "        CofR            (0.25 0 0);\n        pitchAxis       (0 0 1);\n"
        "    }\n\n    forces\n    {\n"
        "        type            forces;\n"
        '        libs            ("libforces.so");\n'
        "        writeControl    timeStep;\n        writeInterval   1;\n"
        "        log             no;\n"
        f"        patches         ({patch});\n"
        "        rho             rhoInf;\n        rhoInf          1;\n"
        "        CofR            (0.25 0 0);\n"
        "    }\n}\n"
    )


def _coeff_header(case_dir: str) -> dict:
    """Reference data OpenFOAM stamps at the top of ``coefficient.dat``.

    Reading it back beats recomputing it: the file records the exact directions
    and reference area the solver used, so a coefficient derived here cannot
    silently disagree with the one the solver reported.
    """
    base = os.path.join(case_dir, "postProcessing", "forceCoeffs")
    if not os.path.isdir(base):
        return {}
    for entry in sorted(os.listdir(base)):
        d = os.path.join(base, entry)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not (name.startswith("coefficient") and name.endswith(".dat")):
                continue
            out: dict = {}
            with open(os.path.join(d, name), encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.startswith("#"):
                        break
                    key, _, value = line.lstrip("#").strip().partition(":")
                    key, value = key.strip(), value.strip()
                    if not value:
                        continue
                    nums = [float(v) for v in
                            value.replace("(", " ").replace(")", " ").split()]
                    out[key] = nums if len(nums) > 1 else nums[0]
            if out:
                return out
    return {}


def read_force_components(case_dir: str) -> dict[str, np.ndarray]:
    """Split drag and lift into their pressure and viscous parts.

    ``forceCoeffs`` reports ``Cd(f)`` / ``Cd(r)``, which are the *front* and
    *rear* contributions about ``CofR`` -- not what is wanted here. The separate
    ``forces`` function object writes the pressure and viscous force vectors, and
    projecting those onto the drag and lift directions from the ``coefficient``
    header gives ``Cd_p, Cd_v, Cl_p, Cl_v``.

    The split matters because the two coefficients are dominated by different
    physics: at these Reynolds numbers drag is mostly wall shear and lift is
    mostly pressure. A seed that is accurate in the pressure field but corrupts
    the near-wall velocity gradient would then converge lift quickly and drag
    slowly -- a prediction this makes testable.

    Returns ``{}`` when the run predates the function object.
    """
    head = _coeff_header(case_dir)
    base = os.path.join(case_dir, "postProcessing", "forces")
    if not head or not os.path.isdir(base):
        return {}
    # Column names come from the file's own header. The layout is not stable
    # across releases -- v2606 writes Time, then *total*, then pressure, then
    # viscous, so counting from the left lands on the wrong vector.
    names: list[str] = []
    rows: list[list[float]] = []
    for entry in sorted(os.listdir(base), key=lambda e: (_as_float(e), e)):
        d = os.path.join(base, entry)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not (name.startswith("force") and name.endswith(".dat")):
                continue
            with open(os.path.join(d, name), encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    if line.startswith("#"):
                        parts = line.lstrip("#").replace("\t", " ").split()
                        if parts and parts[0] == "Time":
                            names = parts
                        continue
                    flat = line.replace("(", " ").replace(")", " ").split()
                    try:
                        rows.append([float(v) for v in flat])
                    except ValueError:
                        continue
    if not rows or not names:
        return {}
    width = min(len(names), min(len(r) for r in rows))
    index = {names[i]: i for i in range(width)}
    if not {"pressure_x", "viscous_x"} <= set(index):
        return {}
    data = np.array([r[:width] for r in rows], dtype=float)
    data = data[np.argsort(data[:, 0], kind="stable")]
    keep = np.ones(len(data), dtype=bool)
    keep[:-1] = data[1:, 0] != data[:-1, 0]
    data = data[keep]

    drag = np.asarray(head.get("dragDir", [1.0, 0.0, 0.0]), dtype=float)
    lift = np.asarray(head.get("liftDir", [0.0, 1.0, 0.0]), dtype=float)
    q = 0.5 * float(head.get("magUInf", 1.0)) ** 2 * float(head.get("Aref", 1.0))
    if q <= 0:
        return {}

    def vector(prefix):
        cols = [index[f"{prefix}_{axis}"] for axis in "xyz"]
        return data[:, cols]

    pressure, viscous = vector("pressure"), vector("viscous")
    return {
        "Time": data[:, 0],
        "Cd_p": pressure @ drag / q,
        "Cd_v": viscous @ drag / q,
        "Cl_p": pressure @ lift / q,
        "Cl_v": viscous @ lift / q,
    }


def _as_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("inf")


def read_force_coeffs(case_dir: str) -> dict[str, np.ndarray]:
    """Read ``postProcessing/forceCoeffs/*/coefficient*.dat`` into arrays.

    Returns ``{"Time": ..., "Cd": ..., "Cl": ...}`` (plus whatever other columns
    the file carries), or ``{}`` if the run predates the function object. Column
    names come from the file's own last ``#`` header line, because they differ
    across OpenFOAM releases. A restarted run leaves several time directories;
    they are concatenated in time order and duplicate times are dropped, keeping
    the later value.
    """
    base = os.path.join(case_dir, "postProcessing", "forceCoeffs")
    if not os.path.isdir(base):
        return {}
    files = []
    for entry in sorted(os.listdir(base)):
        try:
            start = float(entry)
        except ValueError:
            continue
        for name in sorted(os.listdir(os.path.join(base, entry))):
            if name.startswith("coefficient") and name.endswith(".dat"):
                files.append((start, os.path.join(base, entry, name)))
    if not files:
        return {}

    names: list[str] = []
    rows: list[list[float]] = []
    for _start, path in sorted(files):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    parts = line.lstrip("#").split()
                    if parts and parts[0] == "Time":
                        names = parts
                    continue
                try:
                    rows.append([float(v) for v in line.split()])
                except ValueError:
                    continue
    if not rows or not names:
        return {}

    width = min(len(names), min(len(r) for r in rows))
    data = np.array([r[:width] for r in rows], dtype=float)
    # A stable sort by time keeps file order within a repeated time, so dropping
    # all but the last of each equal-time group lets a restart overwrite the
    # samples the interrupted run had already written.
    data = data[np.argsort(data[:, 0], kind="stable")]
    keep = np.ones(len(data), dtype=bool)
    keep[:-1] = data[1:, 0] != data[:-1, 0]
    data = data[keep]
    return {names[i]: data[:, i] for i in range(width)}


def iterations_to_force_band(
    time: np.ndarray,
    values: np.ndarray,
    *,
    reference: float | None = None,
    tol: float = 0.005,
) -> int | None:
    """First iteration after which a force coefficient stays inside a band.

    Returns the earliest ``time[i]`` such that every later sample satisfies
    ``|values[j] - reference| <= tol * |reference|``, or ``None`` if the run
    never settles. The *stays* matters: a coefficient sweeping through its final
    value on the way past would otherwise score as converged at the crossing.

    ``reference`` defaults to the run's own final value, which is right for
    describing a single run. When comparing a cold start against a warm one,
    pass the **same** reference to both -- the converged coefficient for that
    case -- so the arms are measured against one target rather than each against
    wherever it happened to stop.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if time.size == 0 or values.size == 0:
        return None
    n = min(time.size, values.size)
    time, values = time[:n], values[:n]
    ref = float(values[-1]) if reference is None else float(reference)
    if not np.isfinite(ref) or ref == 0.0:
        return None
    inside = np.abs(values - ref) <= tol * abs(ref)
    if not inside.all() and not inside.any():
        return None
    # Walk back from the end over the maximal run of in-band samples.
    outside = np.flatnonzero(~inside)
    first = 0 if outside.size == 0 else int(outside[-1]) + 1
    if first >= n:
        return None
    return int(round(float(time[first])))


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


def _fv_solution(
    tol_p: float,
    tol_u: float,
    n_non_orth: int = 0,
    relax: float = RELAX_U,
    relax_nut: float | None = None,
) -> str:
    """``system/fvSolution`` for steady SIMPLEC.

    ``relax`` defaults to :data:`RELAX_U`; see the note there on why it is not
    the 0.9 that SIMPLEC's usual advice suggests.
    """
    relax_nut = RELAX_NUT if relax_nut is None else relax_nut
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
        + f"        U               {_num(relax)};\n"
        + f"        nuTilda         {_num(relax_nut)};\n"
        + "    }\n}\n"
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
    seconds, ``residuals``: ``{field: [initial residual per outer iteration]}``
    for ``Ux, Uy, p, nuTilda``, and ``elapsed``: cumulative solver seconds per
    iteration, so cost can be read at the point a run met a target rather than
    only at the end.

    The histories are grouped **per outer iteration**, not flattened. This
    matters: with ``nNonOrthogonalCorrectors 2`` the pressure equation is solved
    three times per SIMPLE iteration, so a flat scan yields a ``p`` history three
    times longer than ``Ux``'s and index ``i`` then means outer iteration ``i``
    for velocity but ``i/3`` for pressure. Any metric that compares fields at a
    common index -- :func:`iterations_to_threshold` does -- silently reads
    pressure from a third of the way back. Only the *first* solve of a field
    inside a ``Time`` block is the outer iteration's initial residual; the rest
    are corrector sub-solves. A field absent from a block records ``inf``, so
    every history has one entry per outer iteration and indices line up.
    """
    times = _TIME_RE.findall(text)
    iterations = int(times[-1]) if times else 0

    # split() with a capturing group -> [preamble, time, body, time, body, ...]
    blocks = _TIME_RE.split(text)
    per_iteration: list[dict[str, float]] = []
    elapsed: list[float] = []
    for i in range(1, len(blocks), 2):
        seen: dict[str, float] = {}
        for name, initial, _final, _n in _RES_RE.findall(blocks[i + 1]):
            seen.setdefault(name, float(initial))
        per_iteration.append(seen)
        # Cumulative solver time at the end of this iteration. A warm start
        # shortens the inner linear solves as well as the outer loop, so cost per
        # iteration is not a constant across arms and an iteration saving is not
        # the whole speed-up.
        stamp = _CLOCK_RE.findall(blocks[i + 1])
        elapsed.append(float(stamp[-1][0]) if stamp else float("nan"))

    names: list[str] = []
    for block in per_iteration:
        for name in block:
            if name not in names:
                names.append(name)
    inf = float("inf")
    residuals: dict[str, list[float]] = {
        name: [block.get(name, inf) for block in per_iteration] for name in names
    }

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
        "elapsed": elapsed,   # cumulative solver seconds, one entry per iteration
        # Last *solved* value: a run cut off mid-block leaves inf padding behind.
        "final_residual": {
            k: [x for x in v if x != inf][-1]
            for k, v in residuals.items()
            if any(x != inf for x in v)
        },
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


def completed_run(
    case_dir: str, *, n_iter: int | None = None, start: str | None = None
) -> dict | None:
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
    experiment never mixes budgets; when ``start`` is given ('cold' / 'warm'),
    a directory holding the *other* arm is rejected, so reusing one directory
    across arms cannot silently return the wrong result.
    """
    log = os.path.join(case_dir, "log.simpleFoam")
    if not os.path.isfile(log):
        return None
    if start is not None:
        meta = os.path.join(case_dir, "neuroforge.json")
        if not os.path.isfile(meta):
            return None
        import json as _json

        with open(meta, encoding="utf-8") as fh:
            if _json.load(fh).get("start") != start:
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
    reuse: bool = True,
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

    # Resume: a solve writes its log and fields to disk as it goes, so a run cut
    # short by a power failure leaves every finished case recoverable.
    info = completed_run(case_dir, n_iter=n_iter, start=start) if reuse else None
    if info is None:
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
        wall_time=float(info.get("wall_time", float("nan"))),
        execution_time=float(info["execution_time"]),
        start=start,
        case_dir=case_dir,
        residuals=info["residuals"],
        meta={
            "converged_at": info["converged_at"],
            "final_residual": info["final_residual"],
            "time_dir": latest,
            "fluid_cells": int(cell_ids.size),
            "reused": bool(info.get("reused", False)),
        },
    )


_BOUNDARY_RE = re.compile(r"^\s{4}(\w+)\s*$\s*^\s*\{([^{}]*)\}", re.MULTILINE)


def read_patches(case_dir: str) -> dict[str, str]:
    """``{patch name: type}`` from ``constant/polyMesh/boundary``.

    Read rather than assumed, because the uniform-Cartesian mesh and the two
    body-fitted ones do not name their patches the same way.
    """
    path = os.path.join(case_dir, "constant", "polyMesh", "boundary")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    out = {}
    for name, body in _BOUNDARY_RE.findall(text):
        m = re.search(r"type\s+(\w+)\s*;", body)
        if m:
            out[name] = m.group(1)
    return out


def potential_flow_seed(
    case_dir: str, *, distro: str | None = None, timeout: float = 1800.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve potential flow on an already-meshed case and return ``(u, v, p)``.

    This is the baseline a warm-start claim has to beat. ``potentialFoam`` ships
    with OpenFOAM, needs no model, no training data and no GPU, runs in seconds,
    and is what industry actually reaches for -- it is also the field NVIDIA's
    hybrid initialisation *blends* its surrogate with rather than replaces
    (arXiv:2503.15766). A surrogate seed that does not beat it has no practical
    claim to make, however good its iteration count looks against a cold start.

    Potential flow gets the outer field and the leading-edge pressure close and
    has no boundary layer at all, which is exactly the division of labour
    :func:`~neuroforge.solver.warmstart.masked_seed` exploits.

    **Mutates ``case_dir``'s time-zero fields** -- ``potentialFoam`` overwrites
    ``0/U`` and ``0/p``. Point it at a scratch copy, not at a case whose cold
    start you still need.
    """
    require_openfoam(distro)
    patches = read_patches(case_dir)
    if not patches:
        raise OpenFOAMUnavailable(f"no mesh found in {case_dir}; run blockMesh first")

    # Phi needs one Dirichlet anchor or the Poisson problem is singular. The
    # outlet is the natural place; failing that, the far field.
    anchor = next((p for p in patches if "out" in p.lower()),
                  next((p for p in patches if p not in ("airfoil",)
                        and patches[p] != "empty"), None))
    entries = {}
    for name, kind in patches.items():
        if kind == "empty":
            entries[name] = "        type            empty;\n"
        elif name == anchor:
            entries[name] = ("        type            fixedValue;\n"
                             "        value           uniform 0;\n")
        else:
            entries[name] = "        type            zeroGradient;\n"
    _write(
        os.path.join(case_dir, "0", "Phi"),
        _header("volScalarField", "Phi", "0")
        + "dimensions      [0 2 -1 0 0 0 0];\n\ninternalField   uniform 0;\n\n"
        + _boundary_field(entries),
    )

    path = os.path.join(case_dir, "system", "fvSolution")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    text = text.replace(
        "solvers\n{\n",
        "solvers\n{\n    Phi\n    {\n        solver          GAMG;\n"
        "        smoother        GaussSeidel;\n"
        "        nPreSweeps      0;\n        nPostSweeps     2;\n"
        "        cacheAgglomeration on;\n        agglomerator    faceAreaPair;\n"
        "        nCellsInCoarsestLevel 100;\n        mergeLevels     1;\n"
        "        tolerance       1e-07;\n        relTol          0;\n    }\n\n",
        1,
    )
    text += "\npotentialFlow\n{\n    nNonOrthogonalCorrectors 10;\n}\n"
    _write(path, text)

    # `-writep` needs the Euler-pressure divergence scheme, which a simpleFoam
    # fvSchemes has no reason to carry.
    schemes = os.path.join(case_dir, "system", "fvSchemes")
    with open(schemes, encoding="utf-8") as fh:
        text = fh.read()
    if "div(div(phi,U))" not in text:
        _write(schemes, text.replace("divSchemes\n{\n",
                                     "divSchemes\n{\n    div(div(phi,U)) Gauss linear;\n", 1))

    run_openfoam("potentialFoam -writep", case_dir, distro=distro, timeout=timeout,
                 log_name="log.potentialFoam")
    velocity = read_volfield(os.path.join(case_dir, "0", "U"))
    pressure = read_volfield(os.path.join(case_dir, "0", "p"))
    return velocity[:, 0], velocity[:, 1], pressure


def running_solvers(path_fragment: str = "", *, distro: str | None = None) -> list[str]:
    """Case directories with a live ``simpleFoam`` in them, newest process last.

    Two processes writing one case directory is silent corruption, not a crash:
    :func:`~neuroforge.solver.cgrid.write_cgrid_case` removes the directory
    before rewriting it, so a second run started while a first is still solving
    deletes the first's working directory out from under it. The solve then dies
    with ``bash: log.simpleFoam: No such file or directory``, which names neither
    the cause nor the case, and the arm is quietly missing from the results.

    Check before launching a sweep into a tree something else may still be using.
    ``path_fragment`` narrows it to one tree; an empty string matches every case.
    """
    env = detect_openfoam(distro)
    if env is None:
        return []
    try:
        proc = _run_bash(env, "ps -eo args | grep '[s]impleFoam' || true", 60)
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in (proc.stdout or "").splitlines():
        m = re.search(r'cd "([^"]+)"', line)
        if m and path_fragment in m.group(1):
            found.append(m.group(1))
    return found
