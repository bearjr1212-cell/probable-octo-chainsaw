"""Extract 2D profiles from DXF blueprints (AutoCAD / most CAD exports)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import ezdxf
from ezdxf.entities import Arc, Line
from shapely.geometry import MultiPolygon

from ..geometry import (
    Chain,
    DEFAULT_TOLERANCE,
    closed_loops,
    flatten_arc,
    loops_to_polygons,
    weld_chains,
)


def _entity_chains(entity, sag: float = 0.01) -> List[Chain]:
    """Reduce a single DXF entity to zero or more 2D point chains."""
    dxftype = entity.dxftype()

    if dxftype == "LINE":
        start, end = entity.dxf.start, entity.dxf.end
        return [[(start.x, start.y), (end.x, end.y)]]

    if dxftype == "ARC":
        c = entity.dxf.center
        return [flatten_arc((c.x, c.y), entity.dxf.radius, entity.dxf.start_angle, entity.dxf.end_angle)]

    if dxftype == "CIRCLE":
        c = entity.dxf.center
        return [flatten_arc((c.x, c.y), entity.dxf.radius, 0, 360)]

    if dxftype in ("LWPOLYLINE", "POLYLINE"):
        # virtual_entities() expands bulges (arc segments encoded on a
        # polyline vertex) into concrete LINE/ARC entities, so a single
        # recursive call handles both straight and rounded polylines.
        chains: List[Chain] = []
        for virt in entity.virtual_entities():
            if isinstance(virt, (Line, Arc)):
                chains.extend(_entity_chains(virt, sag=sag))
        return _stitch(chains)

    if dxftype in ("SPLINE", "ELLIPSE"):
        return [[(p.x, p.y) for p in entity.flattening(sag)]]

    return []


def _stitch(chains: List[Chain]) -> List[Chain]:
    """Merge consecutive line/arc chains from a single polyline into one."""
    if not chains:
        return []
    merged = [chains[0]]
    for chain in chains[1:]:
        merged[-1] = merged[-1] + chain[1:]
    return merged


def extract_chains(msp, layer: Optional[str] = None, sag: float = 0.01) -> List[Chain]:
    """Collect raw point chains from every relevant entity in modelspace."""
    chains: List[Chain] = []
    for entity in msp:
        if layer is not None and entity.dxf.layer != layer:
            continue
        chains.extend(_entity_chains(entity, sag=sag))
    return chains


def load_profile(
    path: Union[str, Path],
    layer: Optional[str] = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> MultiPolygon:
    """Parse a DXF file into a :class:`~shapely.geometry.MultiPolygon` profile.

    Args:
        path: Path to the ``.dxf`` file.
        layer: If given, only entities on this layer are considered -- useful
            when a drawing packs multiple views (front/top/side) or
            dimensions/annotations onto separate layers.
        tolerance: Endpoint-matching tolerance, in drawing units, used to
            weld independent LINE/ARC entities back into closed loops.
    """
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    chains = extract_chains(msp, layer=layer)
    loops = closed_loops(weld_chains(chains, tolerance=tolerance), tolerance=tolerance)
    return loops_to_polygons(loops, tolerance=tolerance)


def list_layers(path: Union[str, Path]) -> List[str]:
    """Return the names of layers present in a DXF file's modelspace."""
    doc = ezdxf.readfile(str(path))
    return sorted({e.dxf.layer for e in doc.modelspace()})
