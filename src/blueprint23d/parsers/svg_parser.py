"""Extract 2D profiles from SVG blueprints (vector line art / exported CAD)."""

from __future__ import annotations

from pathlib import Path as FsPath
from typing import List, Union

import svgelements as se
from shapely.geometry import MultiPolygon

from ..geometry import Chain, DEFAULT_TOLERANCE, closed_loops, loops_to_polygons, weld_chains

_CURVE_TYPES = (se.QuadraticBezier, se.CubicBezier, se.Arc)


def _shape_chains(shape: "se.Shape", curve_samples: int = 24) -> List[Chain]:
    """Flatten one SVG shape's segments (already transformed to user space)."""
    chains: List[Chain] = []
    current: Chain = []
    for seg in shape.segments():
        if isinstance(seg, se.Move):
            if len(current) >= 2:
                chains.append(current)
            current = [(seg.end[0], seg.end[1])]
        elif isinstance(seg, se.Close):
            if current:
                current.append((seg.end[0], seg.end[1]))
        elif isinstance(seg, _CURVE_TYPES):
            for i in range(1, curve_samples + 1):
                pt = seg.point(i / curve_samples)
                current.append((pt[0], pt[1]))
        else:  # Line
            current.append((seg.end[0], seg.end[1]))
    if len(current) >= 2:
        chains.append(current)
    return chains


def load_profile(
    path: Union[str, FsPath],
    tolerance: float = DEFAULT_TOLERANCE,
    curve_samples: int = 24,
) -> MultiPolygon:
    """Parse an SVG file into a :class:`~shapely.geometry.MultiPolygon` profile.

    All shape elements (``path``, ``rect``, ``circle``, ``ellipse``,
    ``polygon``, ``polyline``, ``line``) are flattened to polylines in the
    document's user-space coordinates (group transforms applied), then
    welded and nested the same way DXF geometry is.
    """
    svg = se.SVG.parse(str(path))
    chains: List[Chain] = []
    for element in svg.elements():
        if isinstance(element, se.Shape):
            chains.extend(_shape_chains(element, curve_samples=curve_samples))
    loops = closed_loops(weld_chains(chains, tolerance=tolerance), tolerance=tolerance)
    return loops_to_polygons(loops, tolerance=tolerance)
