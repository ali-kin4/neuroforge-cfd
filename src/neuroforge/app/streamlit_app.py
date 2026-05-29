"""Streamlit web UI for NeuroForge CFD.

Run with::

    streamlit run src/neuroforge/app/streamlit_app.py

The sidebar selects a NACA airfoil and the flow conditions (AoA, Re, freestream
speed, grid resolution); the *Solve* button builds a :class:`FlowCase`, runs the
self-correcting engine, and displays the overview figure, key metrics, and the
convergence history.

Streamlit, the engine, and the trainer are all imported lazily inside
:func:`main` so this module imports cleanly without those dependencies (and the
engine may still be under construction in parallel).
"""

from __future__ import annotations

__all__ = ["main"]

# A small, friendly bank of airfoils to choose from in the UI.
_AIRFOILS = (
    "naca0012", "naca2412", "naca4412", "naca0015",
    "naca2415", "naca6409", "naca23012", "naca23015",
)


def main() -> None:
    """Launch the Streamlit application (call inside a ``streamlit run`` process)."""
    try:
        import streamlit as st
    except Exception:
        msg = (
            "Streamlit is not installed. Install the app extras with:\n"
            "    pip install 'neuroforge-cfd[app]'\n"
            "or:\n"
            "    pip install streamlit\n"
            "then run:\n"
            "    streamlit run src/neuroforge/app/streamlit_app.py"
        )
        print(msg)
        return

    st.set_page_config(page_title="NeuroForge CFD", page_icon="🌀", layout="wide")
    st.title("NeuroForge CFD")
    st.caption(
        "Self-correcting, geometry-native AI CFD — predict a flow field from "
        "geometry + boundary conditions, verify physics residuals, and "
        "iteratively self-correct."
    )

    # ---- Sidebar: case definition ---------------------------------------- #
    with st.sidebar:
        st.header("Case")
        airfoil = st.selectbox("NACA airfoil", _AIRFOILS, index=1)
        custom = st.text_input("…or custom NACA code", value="")
        if custom.strip():
            airfoil = custom.strip()
        aoa = st.slider("Angle of attack (deg)", -10.0, 15.0, 5.0, 0.5)
        re_exp = st.slider("Reynolds number (log10)", 5.0, 7.0, 6.5, 0.1)
        reynolds = 10.0 ** re_exp
        u_inf = st.slider("Freestream speed U∞", 1.0, 60.0, 30.0, 1.0)
        resolution = st.select_slider(
            "Grid resolution", options=[48, 64, 96, 128], value=96
        )
        st.caption(f"Re = {reynolds:.3g}")
        solve_clicked = st.button("Solve", type="primary", use_container_width=True)

    if not solve_clicked:
        st.info("Set the case in the sidebar and press **Solve**.")
        return

    # ---- Build the case --------------------------------------------------- #
    try:
        from neuroforge.core.types import FlowCase

        case = FlowCase.from_airfoil(
            airfoil=airfoil, aoa=aoa, reynolds=reynolds,
            u_inf=u_inf, resolution=int(resolution),
        )
    except Exception as exc:
        st.error(f"Could not build the case: {exc}")
        return

    # ---- Run the engine (lazy import; may not exist yet) ------------------ #
    try:
        from neuroforge.solver.engine import NeuroForgeEngine
    except Exception as exc:
        st.error(
            "The solver engine is not available in this build. "
            f"({exc})"
        )
        return

    with st.spinner("Loading engine and solving…"):
        try:
            engine = NeuroForgeEngine.pretrained()
            result = engine.solve(case)
        except Exception as exc:
            st.error(f"Solve failed: {exc}")
            return

    # ---- Display results -------------------------------------------------- #
    summary = result.summary()
    cols = st.columns(4)
    for col, key in zip(cols, ("cl", "cd", "residual_norm", "n_iters")):
        if key in summary:
            col.metric(key, f"{summary[key]:.4g}")

    st.subheader("Overview")
    from neuroforge.viz.plots import overview_figure, plot_convergence

    fig = overview_figure(result)
    st.pyplot(fig)

    with st.expander("Convergence history", expanded=False):
        cfig = plot_convergence(result.history)
        st.pyplot(cfig.figure)
        st.dataframe(result.history)

    with st.expander("All metrics"):
        st.json({k: float(v) for k, v in summary.items()})

    # Offer the standalone HTML report as a download.
    try:
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "report.html")
            result.save_report(path)
            with open(path, encoding="utf-8") as fh:
                html_data = fh.read()
        st.download_button(
            "Download HTML report", data=html_data,
            file_name=f"{case.name}.html", mime="text/html",
        )
    except Exception as exc:  # pragma: no cover - report is best-effort in UI
        st.caption(f"(report download unavailable: {exc})")


if __name__ == "__main__":
    main()
