"""Geometry layer: airfoil generation, SDF/mask rasterisation, and encoding.

Public surface (re-exported here for convenience)::

    from neuroforge.geometry import naca_airfoil, signed_distance, encode_case
"""

from __future__ import annotations

from .airfoil import airfoil_from_dat, naca_airfoil
from .encode import case_geometry_fields, encode_case
from .io import load_obj, load_stl
from .sdf import signed_distance, solid_mask, surface_normals
from .shapes import cylinder_geometry

__all__ = [
    "naca_airfoil",
    "airfoil_from_dat",
    "cylinder_geometry",
    "signed_distance",
    "solid_mask",
    "surface_normals",
    "encode_case",
    "case_geometry_fields",
    "load_stl",
    "load_obj",
]
