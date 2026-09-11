"""One entry point for reading a blueprint, whatever format it is in.

Dispatches on the file extension and returns exact
:class:`~blueprint23d.brep.Face2D` geometry in every case -- vector formats
by preserving what the file says, raster formats by fitting primitives back
out of the pixels. Downstream code therefore never has to care which kind
of drawing it came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

from .brep import Face2D
from .curves import Curve2D
from .geometry import DEFAULT_TOLERANCE

RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VECTOR_EXTENSIONS = {".dxf", ".svg", ".dwg"}


def supported_extensions() -> List[str]:
    return sorted(VECTOR_EXTENSIONS | RASTER_EXTENSIONS)


def load_faces(
    path: Union[str, Path],
    layer: Optional[str] = None,
    scale: float = 1.0,
    invert: bool = False,
    fit_tolerance: float = 0.6,
    weld_tolerance: float = 1e-7,
) -> Tuple[List[Face2D], List[List[Curve2D]]]:
    """Read every closed profile from a drawing as exact faces.

    ``layer`` applies to DXF only; ``scale``, ``invert`` and
    ``fit_tolerance`` apply to raster input only. Returns the faces and any
    chains that would not close -- usually dimension lines or leaders,
    which are worth reporting rather than discarding silently.
    """
    path = Path(path)
    extension = path.suffix.lower()

    if extension == ".dxf":
        from .parsers import dxf_exact

        return dxf_exact.load_faces(path, layer=layer, weld_tolerance=weld_tolerance)

    if extension == ".svg":
        from .parsers import svg_exact

        return svg_exact.load_faces(path, weld_tolerance=weld_tolerance)

    if extension == ".dwg":
        # DWG needs an external converter; whatever that costs is reported
        # rather than silently absorbed.
        from .parsers import dwg as dwg_parser

        faces, open_chains, _report = dwg_parser.load_faces(
            path, layer=layer, weld_tolerance=weld_tolerance
        )
        return faces, open_chains

    if extension in RASTER_EXTENSIONS:
        faces, _reports = _load_raster_faces(
            path, scale=scale, invert=invert, fit_tolerance=fit_tolerance
        )
        return faces, []

    raise ValueError(
        f"unrecognized blueprint format '{extension}' for {path} "
        f"(supported: {', '.join(supported_extensions())})"
    )


def load_face(
    path: Union[str, Path],
    layer: Optional[str] = None,
    scale: float = 1.0,
    invert: bool = False,
    fit_tolerance: float = 0.6,
    weld_tolerance: float = 1e-7,
) -> Face2D:
    """Read the largest closed profile -- the part, on a sheet with several."""
    faces, _ = load_faces(
        path,
        layer=layer,
        scale=scale,
        invert=invert,
        fit_tolerance=fit_tolerance,
        weld_tolerance=weld_tolerance,
    )
    if not faces:
        raise ValueError(f"no closed profile found in {path}")
    return max(faces, key=lambda f: f.area())


def load_raster_faces(
    path: Union[str, Path],
    scale: float = 1.0,
    invert: bool = False,
    fit_tolerance: float = 0.6,
):
    """Raster input together with the fit quality of every recovered primitive.

    ``load_faces`` drops the fit reports because most callers only want the
    geometry, but a number recovered from pixels is not the same kind of
    number as one read out of a DXF, and anything that intends to *quote*
    against it needs to know which it is holding. See
    :mod:`blueprint23d.certificates`.
    """
    return _load_raster_faces(
        Path(path), scale=scale, invert=invert, fit_tolerance=fit_tolerance
    )


def _load_raster_faces(
    path: Path,
    scale: float,
    invert: bool,
    fit_tolerance: float,
) -> Tuple[List[Face2D], List]:
    """Trace a raster drawing and fit exact primitives to each contour.

    Contour topology comes from OpenCV, which is reliable at finding what
    encloses what; the geometry comes from sub-pixel refinement and
    least-squares fitting, which is what beats the pixel grid. Nesting then
    goes through the same parity rule as every other input.

    Returns the nested faces and the flat list of
    :class:`~blueprint23d.fitting.FitReport` for every primitive fitted.
    """
    import cv2
    import numpy as np

    from .assembly import nest_loops
    from .fitting import loop_from_points, refine_subpixel

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    flag = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV
    _, ink = cv2.threshold(blurred, 0, 255, flag | cv2.THRESH_OTSU)

    # Enclosed regions, exactly as in the older raster path: flood from the
    # border through everything that is not ink; whatever the flood cannot
    # reach is bounded by a closed loop of ink.
    padded = cv2.copyMakeBorder(ink, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 128)
    enclosed = ((~(flood == 128)) & (padded == 0))[1:-1, 1:-1].astype(np.uint8) * 255

    contours, _ = cv2.findContours(enclosed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    height = image.shape[0]

    loops = []
    reports = []
    for contour in contours:
        if cv2.contourArea(contour) < 25.0:
            continue
        points = [(float(p[0][0]), float(p[0][1])) for p in contour]
        if len(points) < 8:
            continue
        refined = refine_subpixel(image, points)
        # Pixel space to drawing space: flip the row axis so "up" is +y.
        world = [(x * scale, (height - y) * scale) for x, y in refined]
        try:
            loop, fits = loop_from_points(
                world, tolerance=fit_tolerance * scale, min_arc_points=8
            )
        except (ValueError, RuntimeError):
            continue
        loops.append(loop)
        reports.extend(fits)

    return nest_loops(loops), reports


# Backwards-compatible alias for the older polygon-based pipeline.
def load_profile(
    path: Union[str, Path],
    layer: Optional[str] = None,
    scale: float = 1.0,
    invert: bool = False,
    tolerance: float = DEFAULT_TOLERANCE,
):
    """Legacy shapely-polygon loader, kept for the original mesh pipeline."""
    from .parsers import dxf_parser, raster_parser, svg_parser

    path = Path(path)
    extension = path.suffix.lower()
    if extension == ".dxf":
        return dxf_parser.load_profile(path, layer=layer, tolerance=tolerance)
    if extension == ".svg":
        return svg_parser.load_profile(path, tolerance=tolerance)
    if extension in RASTER_EXTENSIONS:
        return raster_parser.load_profile(path, scale=scale, invert=invert, tolerance=tolerance)
    raise ValueError(f"unrecognized blueprint format '{extension}' for {path}")
