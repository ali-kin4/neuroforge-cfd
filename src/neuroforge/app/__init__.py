"""Interactive web application layer for NeuroForge CFD.

The Streamlit UI lives in :mod:`neuroforge.app.streamlit_app`. Streamlit is an
optional dependency imported lazily inside ``main()`` so this package imports
cleanly without it installed.
"""

from __future__ import annotations

__all__ = ["streamlit_app"]
