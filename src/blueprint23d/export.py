"""Mesh cleanup and export for reconstructed parts."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import trimesh

#: Formats trimesh can write that make sense as a "3D part" deliverable.
SUPPORTED_FORMATS = ("stl", "obj", "glb", "ply")


def repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Best-effort cleanup so slicers/CAD tools accept the mesh.

    Extrusion and boolean CSG normally produce a clean watertight mesh
    already; this exists to paper over the occasional degenerate triangle
    or inconsistent winding introduced by nearly-coincident input geometry.
    """
    mesh.process(validate=True)
    if not mesh.is_watertight:
        mesh.fill_holes()
    mesh.fix_normals()
    return mesh


def export(mesh: trimesh.Trimesh, path: Union[str, Path], repair_mesh: bool = True) -> Path:
    """Write ``mesh`` to ``path``; format is inferred from the extension."""
    path = Path(path)
    fmt = path.suffix.lstrip(".").lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported export format '{fmt}' (supported: {', '.join(SUPPORTED_FORMATS)})")
    if repair_mesh:
        mesh = repair(mesh)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    return path
