"""Live dashboard for the OpenFOAM runs: progress, residuals, meshes, results.

    python scripts/dashboard.py            # http://localhost:8009
    python scripts/dashboard.py --port 9000 --root runs/openfoam

Serves a single self-contained page that polls ``/data.json``. The collector
walks the case directories on every request (cached for a second), so nothing has
to be instrumented and a run started from any other terminal shows up on its own.

Liveness depends on ``openfoam.run_openfoam`` redirecting the solver's output
inside WSL, so ``log.simpleFoam`` grows while the solve runs. Cases written by
the older buffered path only appear once they finish; they are marked ``solving``
until then rather than being reported as stalled.
"""

from __future__ import annotations

import argparse
import json
import os
import time

# Must precede numpy/torch: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

from neuroforge.solver import openfoam as of

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(os.path.dirname(HERE), "app", "openfoam_dashboard.html")

# Residual series are long; the page never draws more points than a chart column
# has pixels, so they are thinned before transport.
MAX_POINTS = 320
# A case whose files have not moved in this long is not making progress.
STALE_AFTER = 180.0


def _thin(seq, n=MAX_POINTS):
    if len(seq) <= n:
        return [round(float(v), 8) for v in seq]
    idx = np.linspace(0, len(seq) - 1, n).astype(int)
    return [round(float(seq[i]), 8) for i in idx]


def _mtime(path):
    newest = 0.0
    for name in ("log.simpleFoam", "log.blockMesh", "neuroforge.json"):
        p = os.path.join(path, name)
        if os.path.isfile(p):
            newest = max(newest, os.path.getmtime(p))
    return newest


def collect_run(path: str, root: str = "") -> dict | None:
    meta_path = os.path.join(path, "neuroforge.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return None

    log = os.path.join(path, "log.simpleFoam")
    text = ""
    if os.path.isfile(log):
        try:
            with open(log, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            text = ""

    info = of.parse_simple_foam_log(text) if text else {
        "iterations": 0, "converged": False, "residuals": {},
        "execution_time": float("nan"), "final_residual": {},
    }
    n_iter = int(meta.get("n_iter") or 0)
    now = time.time()
    age = now - (_mtime(path) or now)
    # neuroforge.json is rewritten immediately before simpleFoam starts, so its
    # mtime is the closest thing to a start stamp without instrumenting the run.
    started = os.path.getmtime(meta_path)
    log_m = os.path.getmtime(log) if os.path.isfile(log) else started
    finished = text.rstrip().endswith("End") or info["converged"]

    # Match the *handler*, not the banner. Every OpenFOAM log opens with
    # "trapFpe: Floating point exception trapping enabled", so a substring test
    # for "Floating point" marks every run in flight as failed -- finished runs
    # escape only because `finished` is tested first.
    broken = ("FOAM FATAL" in text or "sigFpe::sigHandler" in text
              or "sigSegv::sigHandler" in text)
    if broken:
        status = "failed"
    elif finished:
        status = "converged" if info["converged"] else "done"
    elif age > STALE_AFTER:
        status = "stale"
    else:
        status = "solving"

    res = {k: _thin(v) for k, v in info["residuals"].items()}
    floor = of.residual_floor(info["residuals"]) if info["residuals"] else float("nan")
    # Group by the folder under the watch root, so 85 case directories read as a
    # handful of experiments rather than one flat list.
    rel = os.path.relpath(path, root) if root else os.path.basename(path)
    parts = rel.replace("\\", "/").split("/")
    group = parts[0] if len(parts) > 1 else "misc"

    # A coarse ETA from the solver's own clock, only once there is enough to
    # extrapolate from; a fresh run's rate is meaningless.
    eta = None
    ex = info["execution_time"]
    if (status == "solving" and n_iter and info["iterations"] > 20
            and ex == ex and ex > 0):
        eta = round((n_iter - info["iterations"]) * ex / info["iterations"], 0)

    spark = _thin(info["residuals"].get("Ux") or [], 44)
    elapsed = max(0.0, (now if status == "solving" else log_m) - started)

    return {
        "name": os.path.basename(path),
        "group": group,
        "eta": eta,
        "started": started,
        "elapsed": round(elapsed, 1),
        "spark": spark,
        "case": meta.get("case"),
        "airfoil": meta.get("airfoil"),
        "mesh": meta.get("mesh", "cartesian"),
        "start": meta.get("start", "cold"),
        "n_cells": meta.get("n_cells") or meta.get("fluid_cells"),
        "nu": meta.get("nu"),
        "u_inf": meta.get("u_inf"),
        "n_iter": n_iter,
        "iterations": int(info["iterations"]),
        "progress": (min(1.0, info["iterations"] / n_iter) if n_iter else 0.0),
        "status": status,
        "converged": bool(info["converged"]),
        "exec_time": (None if info["execution_time"] != info["execution_time"]
                      else round(float(info["execution_time"]), 1)),
        "floor": (None if floor != floor else float(floor)),
        "final_residual": {k: float(v) for k, v in (info.get("final_residual") or {}).items()},
        "residuals": res,
        "age": round(age, 1),
        "to_1e3": of.iterations_to_threshold(info["residuals"], 1e-3) if res else None,
        "to_1e4": of.iterations_to_threshold(info["residuals"], 1e-4) if res else None,
    }


def collect_geometry(airfoil: str = "naca0012") -> dict:
    """Section outline plus a thinned near-field wireframe of each mesh."""
    from neuroforge.solver import cgrid as cg

    out: dict = {"airfoil": airfoil}
    try:
        spec = cg.CGridSpec()
        inner, nw, ns = cg.inner_curve(airfoil, spec)
        off = cg.offset_open(inner, spec.offset, spec.n_smooth,
                             smooth_range=(nw - 1 - spec.smooth_pad,
                                           nw + ns - 2 + spec.smooth_pad))
        far = cg.outer_curve(spec, nw, ns)
        surf = inner[nw - 1: nw + ns - 1]
        out["section"] = [[round(float(x), 5), round(float(y), 5)] for x, y in surf]
        out["cut"] = [[round(float(x), 5), round(float(y), 5)] for x, y in inner[:nw]]
    except Exception:  # pragma: no cover - geometry is a nicety, never fatal
        pass
    return out


def _wireframe(airfoil: str = "naca0012") -> list:
    """Thinned mesh lines for the geometry panel, computed the way blockMesh does."""
    from neuroforge.solver import cgrid as cg, ogrid as og

    spec = cg.CGridSpec()
    inner, nw, ns = cg.inner_curve(airfoil, spec)
    off = cg.offset_open(inner, spec.offset, spec.n_smooth,
                         smooth_range=(nw - 1 - spec.smooth_pad, nw + ns - 2 + spec.smooth_pad))
    far = cg.outer_curve(spec, nw, ns)
    g_in = og.expansion_ratio(spec.offset, spec.first_cell, spec.n_inner)
    growth = g_in ** (1.0 / max(spec.n_inner - 1, 1))
    last = spec.first_cell * growth ** max(spec.n_inner - 1, 0)
    g_out = og.expansion_ratio(spec.far_radius - spec.offset, last, spec.n_outer)

    def fracs(n, ratio):
        r = ratio ** (1.0 / max(n - 1, 1))
        s = np.concatenate([[0.0], np.cumsum(r ** np.arange(n))])
        return s / s[-1]

    cols = [inner]
    for a, b, n, ratio in ((inner, off, spec.n_inner, g_in),
                           (off, far, spec.n_outer, g_out)):
        for t in fracs(n, ratio)[1:]:
            cols.append(a + t * (b - a))
    nodes = np.stack(cols, axis=1)

    # Sparse on purpose. At panel size a dense wireframe is illegible noise; a few
    # wall-parallel layers show the boundary-layer clustering, which is the one
    # thing worth reading here. The full mesh lives in results/mesh_structure.png.
    lines = []
    keep = (nodes[:, :, 0] > -0.5) & (nodes[:, :, 0] < 2.0) & (np.abs(nodes[:, :, 1]) < 0.75)
    for i in range(0, nodes.shape[0], 8):
        pts = nodes[i][keep[i]]
        if len(pts) > 1:
            lines.append([[round(float(x), 4), round(float(y), 4)] for x, y in pts])
    for j in range(0, nodes.shape[1], 10):
        pts = nodes[:, j][keep[:, j]]
        if len(pts) > 1:
            lines.append([[round(float(x), 4), round(float(y), 4)] for x, y in pts])
    return lines


def collect(root: str, results_dir: str) -> dict:
    runs = []
    if os.path.isdir(root):
        for dirpath, dirnames, filenames in os.walk(root):
            if "neuroforge.json" in filenames:
                r = collect_run(dirpath, root)
                if r:
                    runs.append(r)
                dirnames[:] = []          # a case directory has no case children
    runs.sort(key=lambda r: (-r["age"] if r["status"] == "solving" else 1e9, r["name"]))

    experiments = []
    if os.path.isdir(results_dir):
        for name in sorted(os.listdir(results_dir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(results_dir, name), encoding="utf-8") as fh:
                    blob = json.load(fh)
            except (OSError, ValueError):
                continue
            if isinstance(blob, dict) and "summary" in blob:
                experiments.append({
                    "name": name[:-5],
                    "summary": blob.get("summary"),
                    "rows": blob.get("rows", []),
                    "updated": os.path.getmtime(os.path.join(results_dir, name)),
                })

    active = [r for r in runs if r["status"] == "solving"]
    solver_s = sum(float(r["exec_time"] or 0) for r in runs)
    etas = [r["eta"] for r in active if r.get("eta")]
    return {
        "generated": time.time(),
        "runs": runs,
        "experiments": experiments,
        "totals": {
            "runs": len(runs),
            "solving": len(active),
            "converged": sum(1 for r in runs if r["converged"]),
            "failed": sum(1 for r in runs if r["status"] == "failed"),
            "cells": sum(int(r["n_cells"] or 0) for r in runs),
            "iterations": sum(int(r["iterations"] or 0) for r in runs),
            "planned": sum(int(r["n_iter"] or 0) for r in runs),
            "solver_seconds": round(solver_s, 1),
            "eta": (max(etas) if etas else None),
            "elapsed": round(sum(float(r["elapsed"] or 0) for r in active), 1),
        },
    }


def serve(root: str, results_dir: str, port: int, airfoil: str) -> int:
    import http.server
    import socketserver

    cache: dict = {"at": 0.0, "blob": None}
    geom = dict(collect_geometry(airfoil))
    geom["wire"] = _wireframe(airfoil)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):          # keep the console for our own output
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/data.json"):
                now = time.time()
                if cache["blob"] is None or now - cache["at"] > 1.0:
                    blob = collect(root, results_dir)
                    blob["geometry"] = geom
                    cache.update(at=now, blob=json.dumps(blob).encode("utf-8"))
                self._send(cache["blob"], "application/json")
            else:
                try:
                    with open(PAGE, "rb") as fh:
                        self._send(fh.read(), "text/html; charset=utf-8")
                except OSError:
                    self.send_error(404, f"page not found: {PAGE}")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"NeuroForge OpenFOAM dashboard: http://localhost:{port}")
        print(f"  watching {os.path.abspath(root)}")
        print("  Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.join("runs", "openfoam"))
    ap.add_argument("--results", default="results")
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--airfoil", default="naca0012")
    ap.add_argument("--once", action="store_true", help="print one JSON snapshot and exit")
    args = ap.parse_args(argv)

    if args.once:
        blob = collect(args.root, args.results)
        print(json.dumps({k: v for k, v in blob.items() if k != "runs"}, indent=2)[:2000])
        print(f"\n{len(blob['runs'])} runs")
        for r in blob["runs"][:12]:
            print(f"  {r['name']:44s} {r['status']:10s} {r['iterations']:>5}/{r['n_iter']:<5} "
                  f"cells={r['n_cells']}")
        return 0
    return serve(args.root, args.results, args.port, args.airfoil)


if __name__ == "__main__":
    raise SystemExit(main())
