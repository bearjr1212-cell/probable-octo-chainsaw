"""Dispatch a blueprint file to the right parser based on its extension."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from shapely.geometry import MultiPolygon

from .geometry import DEFAULT_TOLERANCE
from .parsers import dxf_parser, raster_parser, svg_parser

RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_profile(
    path: Union[str, Path],
    layer: Optional[str] = None,
    scale: float = 1.0,
    invert: bool = False,
    tolerance: float = DEFAULT_TOLERANCE,
) -> MultiPolygon:
    """Load a 2D profile from a DXF, SVG, or raster image blueprint.

    ``layer`` only applies to DXF input; ``scale``/``invert`` only apply to
    raster input. The format is chosen from the file extension.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".dxf":
        return dxf_parser.load_profile(path, layer=layer, tolerance=tolerance)
    if ext == ".svg":
        return svg_parser.load_profile(path, tolerance=tolerance)
    if ext in RASTER_EXTENSIONS:
        return raster_parser.load_profile(path, scale=scale, invert=invert, tolerance=tolerance)
    raise ValueError(
        f"unrecognized blueprint format '{ext}' for {path} "
        f"(supported: .dxf, .svg, {', '.join(sorted(RASTER_EXTENSIONS))})"
    )
