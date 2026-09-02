"""Extract 2D profiles from scanned/rasterized blueprints (PNG, JPG, ...).

This is the least reliable input path: a raster has no notion of an
idealized zero-width "line" the way DXF/SVG geometry does, so results
depend on scan quality, contrast, and the tuning knobs below. Prefer the
DXF or SVG parsers when a vector source is available at all; use this one
for photographed or scanned paper drawings.

Ink strokes are given real pixel width once rasterized, so naively tracing
"foreground" contours finds *two* boundaries per stroke (its inner and
outer edge) instead of the one idealized curve a vector parser would give.
To avoid that, this flood-fills from the image border through everything
that isn't ink; whatever pixels the flood can't reach are the enclosed
pockets a closed loop of ink encircles -- exactly the regions a DXF/SVG
closed loop would describe -- and those get contoured and nested the same
way (holes-in-solids-in-holes, by geometric containment).
"""

from __future__ import annotations

from pathlib import Path as FsPath
from typing import List, Union

import cv2
import numpy as np
from shapely.geometry import MultiPolygon

from ..geometry import Chain, DEFAULT_TOLERANCE, loops_to_polygons


def scale_from_reference(pixel_distance: float, real_distance: float) -> float:
    """Compute a ``units per pixel`` scale from one known dimension on the drawing.

    Measure a known feature in pixels (e.g. a dimension line or a ruler
    placed in the photo) and pass its true length to get the ``scale``
    value expected by :func:`load_profile`.
    """
    if pixel_distance <= 0:
        raise ValueError("pixel_distance must be positive")
    return real_distance / pixel_distance


def _ink_mask(gray: np.ndarray, invert: bool) -> np.ndarray:
    """Binarize so drawn ink strokes are foreground (255), everything else 0."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # invert=False: dark ink on a light background -> low values are ink.
    # invert=True: light ink on a dark background -> high values are ink.
    flag = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV
    _, ink = cv2.threshold(blurred, 0, 255, flag | cv2.THRESH_OTSU)
    return ink


def _enclosed_regions(ink: np.ndarray) -> np.ndarray:
    """Non-ink pixels not reachable from the image border, as a 0/255 mask."""
    # Pad with a guaranteed-background 1px border so a single flood fill
    # from (0, 0) reaches every border-connected pixel regardless of which
    # edge of the image the drawing happens to touch.
    padded = cv2.copyMakeBorder(ink, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ff_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 128)
    exterior = flood == 128
    enclosed = (~exterior) & (padded == 0)
    return (enclosed[1:-1, 1:-1].astype(np.uint8)) * 255


def load_profile(
    path: Union[str, FsPath],
    scale: float = 1.0,
    invert: bool = False,
    min_area_px: float = 25.0,
    approx_epsilon_px: float = 1.5,
    tolerance: float = DEFAULT_TOLERANCE,
) -> MultiPolygon:
    """Parse a raster blueprint into a :class:`~shapely.geometry.MultiPolygon` profile.

    Args:
        path: Path to the image file.
        scale: Real-world units per pixel (see :func:`scale_from_reference`).
            Defaults to 1.0, i.e. output stays in pixel units.
        invert: Set True if the drawing is light lines on a dark background
            (e.g. a traditional blue "blueprint"); False for the common case
            of dark lines on a light/white background.
        min_area_px: Enclosed regions smaller than this (in squared pixels,
            before scaling) are dropped as scan noise/text/dimension marks.
        approx_epsilon_px: Polygon simplification tolerance passed to
            ``cv2.approxPolyDP``, in pixels. Larger values give coarser,
            more forgiving outlines.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    height = image.shape[0]

    ink = _ink_mask(image, invert=invert)
    enclosed = _enclosed_regions(ink)
    contours, _hierarchy = cv2.findContours(enclosed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    loops: List[Chain] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area_px:
            continue
        approx = cv2.approxPolyDP(contour, approx_epsilon_px, closed=True)
        if len(approx) < 3:
            continue
        # Image pixel (col, row) -> real-world (x, y): flip the row axis so
        # "up" in the drawing is +y, matching CAD/SVG/DXF convention.
        loop = [((px * scale), (height - py) * scale) for px, py in approx.reshape(-1, 2)]
        loop.append(loop[0])
        loops.append(loop)

    return loops_to_polygons(loops, tolerance=tolerance)
