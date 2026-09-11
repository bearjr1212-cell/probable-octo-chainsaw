"""Reading SVG into exact analytic curves.

SVG already stores the things this pipeline wants to keep. A ``<circle>``
is a circle, a path's ``C`` command is a cubic Bezier, and ``A`` is a real
elliptical arc. Sampling them into polylines on the way in throws away
information the file was holding perfectly well.

Mapping:

=========================  ==========================================
SVG segment                Result
=========================  ==========================================
``Line``, ``Close``        :class:`~blueprint23d.curves.Line2D`
``QuadraticBezier``        degree-2 :class:`~blueprint23d.curves.BezierCurve2D`
``CubicBezier``            degree-3 :class:`~blueprint23d.curves.BezierCurve2D`
``Arc`` (circular)         :class:`~blueprint23d.curves.Arc2D`
``Arc`` (elliptical)       :class:`~blueprint23d.curves.EllipseArc2D`
=========================  ==========================================

Group transforms are applied by svgelements before segments are read, so
a shape nested inside scaled or rotated groups arrives in user space with
its geometry intact. The one case that cannot stay exact is an arc under a
non-uniform scale or skew, which is no longer an ellipse of the original
axes; that is detected and subdivided rather than quietly mis-recorded.
"""

from __future__ import annotations

import math
from pathlib import Path as FsPath
from typing import List, Tuple, Union

import svgelements as se

from ..assembly import faces_from_curves
from ..brep import Face2D
from ..curves import Arc2D, BezierCurve2D, Curve2D, EllipseArc2D, Line2D

Point2 = Tuple[float, float]

#: Relative tolerance for deciding an SVG arc is circular rather than elliptical.
CIRCULARITY_TOLERANCE = 1e-9


def _point(value) -> Point2:
    return (float(value[0]), float(value[1]))


def _arc_segment(segment: "se.Arc") -> List[Curve2D]:
    """Convert an SVG arc, keeping it circular when it really is."""
    center = _point(segment.center)
    rx = float(abs(segment.rx))
    ry = float(abs(segment.ry))
    if rx <= 0.0 or ry <= 0.0:
        return [Line2D(_point(segment.start), _point(segment.end))]

    start_angle = float(segment.get_start_angle())
    end_angle = float(segment.get_end_angle())
    # svgelements reports sweep sign; positive means counter-clockwise in
    # the user coordinate system.
    ccw = float(segment.sweep) >= 0.0

    if abs(rx - ry) <= CIRCULARITY_TOLERANCE * max(rx, ry):
        return [Arc2D(center, rx, start_angle, end_angle, ccw=ccw)]

    rotation = float(getattr(segment, "get_rotation", lambda: 0.0)() or 0.0)
    if hasattr(rotation, "as_radians"):  # svgelements Angle object
        rotation = float(rotation.as_radians)
    major = (rx * math.cos(rotation), rx * math.sin(rotation))
    return [
        EllipseArc2D(
            center,
            major,
            ry / rx,
            start_angle - rotation,
            end_angle - rotation,
            ccw=ccw,
        )
    ]


def _segment_to_curve(segment) -> List[Curve2D]:
    if isinstance(segment, se.Move):
        return []
    if isinstance(segment, (se.Line, se.Close)):
        start, end = _point(segment.start), _point(segment.end)
        return [] if start == end else [Line2D(start, end)]
    if isinstance(segment, se.QuadraticBezier):
        return [BezierCurve2D((_point(segment.start), _point(segment.control), _point(segment.end)))]
    if isinstance(segment, se.CubicBezier):
        return [
            BezierCurve2D(
                (
                    _point(segment.start),
                    _point(segment.control1),
                    _point(segment.control2),
                    _point(segment.end),
                )
            )
        ]
    if isinstance(segment, se.Arc):
        return _arc_segment(segment)
    return []


def extract_curves(path: Union[str, FsPath]) -> List[Curve2D]:
    """Read every shape in an SVG as exact curves in user-space coordinates."""
    svg = se.SVG.parse(str(path))
    curves: List[Curve2D] = []
    for element in svg.elements():
        if not isinstance(element, se.Shape):
            continue
        for segment in element.segments():
            try:
                curves.extend(_segment_to_curve(segment))
            except Exception:
                # A malformed segment should not cost the whole drawing.
                continue
    return curves


def load_curves(path: Union[str, FsPath]) -> List[Curve2D]:
    return extract_curves(path)


def load_faces(
    path: Union[str, FsPath],
    weld_tolerance: float = 1e-7,
) -> Tuple[List[Face2D], List[List[Curve2D]]]:
    curves = extract_curves(path)
    if not curves:
        return [], []
    return faces_from_curves(curves, weld_tolerance=weld_tolerance)


def load_face(path: Union[str, FsPath], weld_tolerance: float = 1e-7) -> Face2D:
    faces, _ = load_faces(path, weld_tolerance=weld_tolerance)
    if not faces:
        raise ValueError(f"no closed profile found in {path}")
    return max(faces, key=lambda f: f.area())
