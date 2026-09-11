"""Exact clipping of curved faces against axis-aligned lines.

Multiview reconstruction needs to ask, over and over, "what is this profile
restricted to x between a and b?". Doing that on a tessellation throws away
everything the exact layer was built to preserve, so it is done here on the
curves themselves: an arc clipped by a line is still an arc, with the same
centre and radius, over a shorter angular range.

The intersections are closed form, which is the whole reason this stays
exact:

* a **line** meets ``x = c`` at a single parameter, by one division;
* an **arc** meets it where :math:`\\cos\\theta = (c - c_x)/r`, giving two
  candidate angles from one ``acos``;
* an **ellipse** reduces to :math:`a\\cos\\theta + b\\sin\\theta = d`, solved
  by the standard amplitude-phase substitution;
* a **Bezier** has no closed form in general, so its x-component is
  bracketed by sign changes on its control polygon and refined by bisection
  to full double precision -- still a root of the true curve, not of an
  approximation to it.

Clipping a loop then walks its curves in order, trims each at the crossing
parameters, keeps the pieces on the wanted side, and rejoins consecutive
kept runs along the clip line. For a half-plane -- a convex window -- that
loop-order pairing is the correct boundary, which is what makes this
usable on the non-convex profiles real parts have.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .brep import Face2D, Loop2D
from .curves import (
    Arc2D,
    BezierCurve2D,
    Curve2D,
    DEFAULT_CHORD_TOLERANCE,
    EllipseArc2D,
    Line2D,
    NurbsCurve2D,
)

Point2 = Tuple[float, float]

X, Y = 0, 1

#: Parameters closer than this to 0 or 1 are treated as endpoint crossings.
_PARAM_EPS = 1e-12


def _coord(point: Point2, axis: int) -> float:
    return point[axis]


# --------------------------------------------------------------------------
# Curve / axis-line intersection, in closed form where one exists
# --------------------------------------------------------------------------


def _line_crossings(curve: Line2D, axis: int, value: float) -> List[float]:
    a = _coord(curve.p0, axis)
    b = _coord(curve.p1, axis)
    if a == b:
        return []
    t = (value - a) / (b - a)
    return [t] if 0.0 < t < 1.0 else []


def _arc_crossings(curve: Arc2D, axis: int, value: float) -> List[float]:
    """Where a circular arc meets an axis-aligned line.

    On ``x = c`` the condition is ``cos(theta) = (c - cx)/r``; on ``y = c``
    it is ``sin(theta) = (c - cy)/r``. Both give two candidate angles,
    which are then tested against the arc's actual swept range.
    """
    center = curve.center
    ratio = (value - center[axis]) / curve.radius
    if ratio < -1.0 or ratio > 1.0:
        return []

    if axis == X:
        base = math.acos(max(-1.0, min(1.0, ratio)))
        angles = (base, -base)
    else:
        base = math.asin(max(-1.0, min(1.0, ratio)))
        angles = (base, math.pi - base)

    sweep = curve.sweep
    if sweep == 0.0:
        return []

    params: List[float] = []
    for angle in angles:
        # Offset from the arc's start, measured in the sweep's direction.
        if sweep > 0.0:
            offset = (angle - curve.start_angle) % (2.0 * math.pi)
        else:
            offset = -((curve.start_angle - angle) % (2.0 * math.pi))
        t = offset / sweep
        if _PARAM_EPS < t < 1.0 - _PARAM_EPS:
            params.append(t)
    return params


def _ellipse_crossings(curve: EllipseArc2D, axis: int, value: float) -> List[float]:
    """Solve ``a cos(theta) + b sin(theta) = d`` by amplitude and phase."""
    mx, my = curve.major
    nx, ny = -my * curve.ratio, mx * curve.ratio
    a = mx if axis == X else my
    b = nx if axis == X else ny
    d = value - curve.center[axis]

    amplitude = math.hypot(a, b)
    if amplitude == 0.0 or abs(d) > amplitude:
        return []
    phase = math.atan2(b, a)
    base = math.acos(max(-1.0, min(1.0, d / amplitude)))

    sweep = curve.sweep
    if sweep == 0.0:
        return []

    params: List[float] = []
    for angle in (phase + base, phase - base):
        if sweep > 0.0:
            offset = (angle - curve.start_param) % (2.0 * math.pi)
        else:
            offset = -((curve.start_param - angle) % (2.0 * math.pi))
        t = offset / sweep
        if _PARAM_EPS < t < 1.0 - _PARAM_EPS:
            params.append(t)
    return params


def _sampled_crossings(curve: Curve2D, axis: int, value: float, samples: int = 64) -> List[float]:
    """Bracket sign changes, then bisect to full precision.

    Used for free-form curves, which have no closed-form intersection. The
    root found is a root of the real curve: only the *search* is numerical,
    and bisection on a bracketed sign change cannot converge anywhere else.
    """
    params: List[float] = []
    previous_t = 0.0
    previous = _coord(curve.point(0.0), axis) - value
    for i in range(1, samples + 1):
        t = i / samples
        current = _coord(curve.point(t), axis) - value
        if previous == 0.0 and _PARAM_EPS < previous_t < 1.0 - _PARAM_EPS:
            params.append(previous_t)
        elif previous * current < 0.0:
            lo, hi = previous_t, t
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                mid_value = _coord(curve.point(mid), axis) - value
                if (mid_value < 0.0) == (previous < 0.0):
                    lo = mid
                else:
                    hi = mid
            root = 0.5 * (lo + hi)
            if _PARAM_EPS < root < 1.0 - _PARAM_EPS:
                params.append(root)
        previous_t, previous = t, current
    return params


def curve_axis_crossings(curve: Curve2D, axis: int, value: float) -> List[float]:
    """Sorted parameters where ``curve`` crosses the line ``axis == value``."""
    if isinstance(curve, Line2D):
        params = _line_crossings(curve, axis, value)
    elif isinstance(curve, Arc2D):
        params = _arc_crossings(curve, axis, value)
    elif isinstance(curve, EllipseArc2D):
        params = _ellipse_crossings(curve, axis, value)
    elif isinstance(curve, (BezierCurve2D, NurbsCurve2D)):
        params = _sampled_crossings(curve, axis, value)
    else:
        params = _sampled_crossings(curve, axis, value)
    return sorted(set(params))


# --------------------------------------------------------------------------
# Half-plane clipping
# --------------------------------------------------------------------------


def _split_at(curve: Curve2D, params: Sequence[float]) -> List[Curve2D]:
    """Cut a curve at the given parameters, keeping every piece exact."""
    cuts = [0.0] + [p for p in params if 0.0 < p < 1.0] + [1.0]
    pieces: List[Curve2D] = []
    for lo, hi in zip(cuts, cuts[1:]):
        if hi - lo <= _PARAM_EPS:
            continue
        try:
            pieces.append(curve.subcurve(lo, hi))
        except NotImplementedError:
            # A curve type that cannot be trimmed exactly keeps its whole
            # self; the caller's inside/outside test then applies to all of
            # it, which is conservative rather than wrong.
            return [curve]
    return pieces or [curve]


def clip_loop_to_halfplane(
    loop: Loop2D,
    axis: int,
    value: float,
    keep_below: bool,
    join_tolerance: float = 1e-9,
) -> Optional[Loop2D]:
    """Clip a closed loop to one side of an axis-aligned line.

    Returns the clipped loop, ``None`` if nothing survives, or the original
    loop when it lies entirely on the kept side.
    """
    pieces: List[Curve2D] = []
    for curve in loop.curves:
        params = curve_axis_crossings(curve, axis, value)
        pieces.extend(_split_at(curve, params))

    kept: List[Curve2D] = []
    for piece in pieces:
        midpoint = _coord(piece.point(0.5), axis)
        inside = midpoint <= value if keep_below else midpoint >= value
        if inside:
            kept.append(piece)

    if not kept:
        return None
    if len(kept) == len(pieces):
        return loop

    # Rejoin consecutive kept runs along the clip line. Both endpoints of
    # each gap lie on the line by construction, so the connecting edge is a
    # straight segment of it.
    bridged: List[Curve2D] = []
    for index, piece in enumerate(kept):
        bridged.append(piece)
        nxt = kept[(index + 1) % len(kept)]
        gap = math.hypot(piece.end[0] - nxt.start[0], piece.end[1] - nxt.start[1])
        if gap > join_tolerance:
            bridged.append(Line2D(piece.end, nxt.start))

    try:
        return Loop2D(bridged, tolerance=max(join_tolerance, 1e-9))
    except ValueError:
        return None


def clip_face_to_halfplane(
    face: Face2D,
    axis: int,
    value: float,
    keep_below: bool,
) -> Optional[Face2D]:
    """Clip a face (and its holes) to one side of an axis-aligned line."""
    outer = clip_loop_to_halfplane(face.outer, axis, value, keep_below)
    if outer is None:
        return None

    holes: List[Loop2D] = []
    for hole in face.inners:
        clipped = clip_loop_to_halfplane(hole, axis, value, keep_below)
        if clipped is not None:
            holes.append(clipped)
    return Face2D.create(outer, holes)


def clip_face_to_slab(face: Face2D, axis: int, low: float, high: float) -> Optional[Face2D]:
    """Restrict a face to ``low <= axis <= high``, exactly."""
    if high < low:
        low, high = high, low
    clipped = clip_face_to_halfplane(face, axis, high, keep_below=True)
    if clipped is None:
        return None
    return clip_face_to_halfplane(clipped, axis, low, keep_below=False)


def face_extent(face: Face2D, axis: int) -> Tuple[float, float]:
    """Exact ``(min, max)`` of a face along one axis."""
    minx, miny, maxx, maxy = face.bounds()
    return (minx, maxx) if axis == X else (miny, maxy)


def loop_spans_at(loop: Loop2D, axis: int, value: float, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> List[Tuple[float, float]]:
    """Intervals of the other axis where the line ``axis == value`` is inside.

    Used by the sweep: for a front view, this answers "at this height, over
    what range of x is there material?".
    """
    other = 1 - axis
    crossings: List[float] = []
    for curve in loop.curves:
        for t in curve_axis_crossings(curve, axis, value):
            crossings.append(_coord(curve.point(t), other))
        # Endpoints exactly on the line count too.
        for endpoint in (curve.start, curve.end):
            if abs(_coord(endpoint, axis) - value) <= 1e-12:
                crossings.append(_coord(endpoint, other))

    crossings = sorted(set(round(c, 12) for c in crossings))
    if len(crossings) < 2:
        return []
    return [(a, b) for a, b in zip(crossings[0::2], crossings[1::2])]
