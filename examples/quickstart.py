"""NeuroForge CFD — the 60-second quickstart (the README example).

Build an airfoil case, load (or auto-train) the pretrained engine, solve, and
write a self-contained HTML report. Everything runs on CPU.

Run it with::

    python examples/quickstart.py
"""

from __future__ import annotations


def main() -> None:
    # Lazy imports so the file is importable even while the engine is being
    # built in parallel; the engine is only needed when actually running.
    from neuroforge import FlowCase, NeuroForgeEngine

    # 1) Define the problem: a NACA 2412 at 5 deg, Re = 3e6, U = 30 m/s.
    case = FlowCase.from_airfoil("naca2412", aoa=5, reynolds=3e6, u_inf=30.0)

    # 2) Load the self-correcting engine (downloads/trains a tiny demo model the
    #    first time, then caches it under checkpoints/demo.pt).
    engine = NeuroForgeEngine.pretrained()

    # 3) Solve: predict -> verify physics -> self-correct low-trust regions.
    result = engine.solve(case)

    # 4) Inspect the engineering metrics and save a shareable report.
    print("metrics:", result.summary())
    path = result.save_report("report.html")
    print("report written to", path)


if __name__ == "__main__":
    main()
