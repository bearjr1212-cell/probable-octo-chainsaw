"""Reading DXF into exact analytic curves.

The older :mod:`dxf_parser` flattens everything to point chains on the way
in. That is fine for a mesh, and fatal for anything else: once a ``CIRCLE``
becomes a 64-gon, the drawing's nominal diameter is gone and no amount of
care downstream brings it back.

This reader preserves what the file actually says:

===================  ==========================================
DXF entity           Result
===================  ==========================================
``LINE``             :class:`~blueprint23d.curves.Line2D`
``ARC``              :class:`~blueprint23d.curves.Arc2D`
``CIRCLE``           full :class:`~blueprint23d.curves.Arc2D`
``ELLIPSE``          :class:`~blueprint23d.curves.EllipseArc2D`
``SPLINE``           :class:`~blueprint23d.curves.NurbsCurve2D`
``LWPOLYLINE``       lines, and exact arcs from bulge values
``POLYLINE``         same
===================  ==========================================

The bulge case is worth calling out. A rounded corner in a polyline is
stored as a single number on the preceding vertex -- the tangent of a
quarter of the arc's included angle -- and a flattening reader turns it
into a handful of short segments. Converted properly it is an exact arc,
which is why a filleted part read through here exports with true
cylindrical faces.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import ezdxf
from ezdxf.math import bulge_to_arc

from ..assembly import faces_from_curves
from ..brep import Face2D
from ..curves import Arc2D, Curve2D, EllipseArc2D, Line2D, NurbsCurve2D

Point2 = Tuple[float, float]


def _as_point(value) -> Point2:
    return (float(value[0]), float(value[1]))


def _line(entity) -> List[Curve2D]:
    start, end = _as_point(entity.dxf.start), _as_point(entity.dxf.end)
    if start == end:
        return []
    return [Line2D(start, end)]


def _arc(entity) -> List[Curve2D]:
    center = _as_point(entity.dxf.center)
    # DXF stores arc angles in degrees, always counter-clockwise.
    return [
        Arc2D(
            center,
            float(entity.dxf.radius),
            math.radians(float(entity.dxf.start_angle)),
            math.radians(float(entity.dxf.end_angle)),
            ccw=True,
        )
    ]


def _circle(entity) -> List[Curve2D]:
    return [Arc2D.full_circle(_as_point(entity.dxf.center), float(entity.dxf.radius))]


def _ellipse(entity) -> List[Curve2D]:
    major = _as_point(entity.dxf.major_axis)
    return [
        EllipseArc2D(
            _as_point(entity.dxf.center),
            major,
            float(entity.dxf.ratio),
            float(entity.dxf.start_param),
            float(entity.dxf.end_param),
            ccw=True,
        )
    ]


def _spline(entity) -> List[Curve2D]:
    """A SPLINE is already a NURBS; copy it across rather than sampling it.

    The DXF may define the curve by fit points instead of control points,
    in which case ``control_points`` is empty on the entity itself and the
    B-spline has to be asked for. Reading the raw attributes would silently
    miss that case and fall back to flattening a curve that was perfectly
    exact all along.
    """
    try:
        spline = entity.construction_tool()
        control = [_as_point(p) for p in spline.control_points]
        knots = [float(k) for k in spline.knots()]
        degree = int(spline.degree)
        raw_weights = list(spline.weights()) if spline.is_rational else []
    except Exception:
        control = [_as_point(p) for p in entity.control_points]
        knots = [float(k) for k in entity.knots]
        degree = int(entity.dxf.degree)
        raw_weights = list(entity.weights) if entity.weights is not None else []

    weights = None
    if raw_weights:
        raw = [float(w) for w in raw_weights]
        if any(abs(w - raw[0]) > 1e-12 for w in raw):
            weights = tuple(raw)

    if len(control) < degree + 1 or len(knots) != len(control) + degree + 1:
        # Some writers emit an unclamped or otherwise irregular knot vector.
        # Rather than guess, fall back to ezdxf's own flattening, which at
        # least keeps the geometry; the curve is no longer exact and the
        # caller can tell because the result is a chain of lines.
        points = [_as_point(p) for p in entity.flattening(0.001)]
        return [Line2D(a, b) for a, b in zip(points, points[1:]) if a != b]

    return [NurbsCurve2D(tuple(control), tuple(knots), degree, weights)]


def _polyline(entity) -> List[Curve2D]:
    """Expand a polyline, converting bulge values into exact arcs."""
    try:
        points = list(entity.get_points("xyb"))
    except (AttributeError, TypeError):
        points = [(v.dxf.location.x, v.dxf.location.y, float(getattr(v.dxf, "bulge", 0.0))) for v in entity.vertices]

    closed = bool(getattr(entity, "closed", False)) or bool(int(getattr(entity.dxf, "flags", 0)) & 1)
    if len(points) < 2:
        return []

    pairs = list(zip(points, points[1:]))
    if closed:
        pairs.append((points[-1], points[0]))

    curves: List[Curve2D] = []
    for (x0, y0, bulge), (x1, y1, _) in pairs:
        start, end = (float(x0), float(y0)), (float(x1), float(y1))
        if start == end:
            continue
        if abs(float(bulge)) < 1e-12:
            curves.append(Line2D(start, end))
            continue
        # bulge = tan(included_angle / 4); ezdxf turns that into an exact
        # centre/radius/angle triple, which is what an arc really is.
        center, start_angle, end_angle, radius = bulge_to_arc(start, end, float(bulge))
        curves.append(Arc2D(_as_point(center), float(radius), float(start_angle), float(end_angle), ccw=True))
    return curves


_HANDLERS = {
    "LINE": _line,
    "ARC": _arc,
    "CIRCLE": _circle,
    "ELLIPSE": _ellipse,
    "SPLINE": _spline,
    "LWPOLYLINE": _polyline,
    "POLYLINE": _polyline,
}

#: Entity types that carry annotation rather than part geometry. Reading a
#: dimension's leader lines as part of the outline is a classic way to end
#: up with a nonsense profile, so they are skipped by default.
ANNOTATION_TYPES = frozenset(
    {"DIMENSION", "TEXT", "MTEXT", "LEADER", "MULTILEADER", "HATCH", "INSERT", "POINT", "ATTDEF"}
)


def extract_curves(
    msp,
    layer: Optional[str] = None,
    skip_annotations: bool = True,
) -> List[Curve2D]:
    """Collect exact curves from a modelspace."""
    curves: List[Curve2D] = []
    for entity in msp:
        dxftype = entity.dxftype()
        if skip_annotations and dxftype in ANNOTATION_TYPES:
            continue
        if layer is not None and entity.dxf.layer != layer:
            continue
        handler = _HANDLERS.get(dxftype)
        if handler is None:
            continue
        try:
            curves.extend(handler(entity))
        except Exception:
            # One malformed entity must not lose the rest of the drawing.
            continue
    return curves


def load_curves(
    path: Union[str, Path],
    layer: Optional[str] = None,
    skip_annotations: bool = True,
) -> List[Curve2D]:
    doc = ezdxf.readfile(str(path))
    return extract_curves(doc.modelspace(), layer=layer, skip_annotations=skip_annotations)


def load_faces(
    path: Union[str, Path],
    layer: Optional[str] = None,
    skip_annotations: bool = True,
    weld_tolerance: float = 1e-7,
) -> Tuple[List[Face2D], List[List[Curve2D]]]:
    """Read a DXF into exact nested faces, plus any chains that did not close."""
    curves = load_curves(path, layer=layer, skip_annotations=skip_annotations)
    if not curves:
        return [], []
    return faces_from_curves(curves, weld_tolerance=weld_tolerance)


def load_face(
    path: Union[str, Path],
    layer: Optional[str] = None,
    skip_annotations: bool = True,
    weld_tolerance: float = 1e-7,
) -> Face2D:
    """Read the largest closed face from a DXF -- the part, on a busy sheet."""
    faces, _ = load_faces(path, layer=layer, skip_annotations=skip_annotations, weld_tolerance=weld_tolerance)
    if not faces:
        raise ValueError(f"no closed profile found in {path}")
    return max(faces, key=lambda f: f.area())
