"""Exact analytic curves, and tessellation with proven deviation bounds.

The precision ceiling of a CAD pipeline is set at parse time. If a DXF
``CIRCLE`` becomes a 64-gon the moment it is read, no amount of care
downstream recovers the lost geometry: the hole is permanently 0.1%
undersized, its "cylindrical" wall is permanently faceted, and the STEP
file can only ever describe a prism. So nothing here is flattened on
input. A circle stays a circle -- center, radius, angles -- all the way to
the exporter, and is sampled only when something genuinely needs points,
at a tolerance the caller chooses.

When sampling does happen, the deviation is *bounded by a theorem*, not by
a segment count someone guessed:

* **Circular arcs** use the exact sagitta identity. A chord subtending
  angle :math:`\\Delta` on radius :math:`r` deviates from the arc by
  exactly :math:`r(1 - \\cos(\\Delta/2))` at its midpoint, and that is the
  maximum. Inverting it gives the largest angular step meeting a
  tolerance, :math:`\\Delta_{max} = 2\\arccos(1 - \\varepsilon/r)`, so the
  segment count is optimal -- no sample is wasted and none is missing.

* **Ellipses** use the linear-interpolation error bound
  :math:`\\|f - L\\|_\\infty \\le (b-a)^2 \\|f''\\|_\\infty / 8` applied
  componentwise, with :math:`\\|C''\\|` bounded analytically by the
  semi-axes.

* **Bezier and NURBS** use the convex-hull flatness bound. A Bezier curve
  lies in the convex hull of its control points; distance to a segment is
  a convex function, so its maximum over the hull is attained at a control
  point. Hence the curve's deviation from its chord is at most
  :math:`\\max_i \\operatorname{dist}(P_i, [P_0, P_n])`, which is
  computable directly and holds for rational curves too, since a rational
  Bezier is still contained in that hull. de Casteljau subdivision shrinks
  it quadratically, so refinement terminates quickly.

Every :meth:`Curve2D.tessellate` returns points that provably lie within
the requested tolerance of the true curve, and
:meth:`Curve2D.deviation_certificate` reports the bound actually achieved.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Point2 = Tuple[float, float]

#: Default chordal tolerance, in drawing units. At 1 micron a tessellated
#: circle is well inside the tolerance of any additive or subtractive
#: process this tool feeds.
DEFAULT_CHORD_TOLERANCE = 1e-3

_TAU = 2.0 * math.pi


def _normalize_angle(angle: float) -> float:
    """Fold an angle into ``[0, 2*pi)``."""
    a = math.fmod(angle, _TAU)
    return a + _TAU if a < 0.0 else a


def _distance_point_to_segment(p: Point2, a: Point2, b: Point2) -> float:
    """Euclidean distance from ``p`` to the closed segment ``ab``."""
    abx, aby = b[0] - a[0], b[1] - a[1]
    denom = abx * abx + aby * aby
    if denom == 0.0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / denom
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(p[0] - (a[0] + t * abx), p[1] - (a[1] + t * aby))


@dataclass(frozen=True)
class DeviationCertificate:
    """What a tessellation actually guarantees.

    ``bound`` is an upper bound on the Hausdorff distance between the true
    curve and the returned polyline. ``exact`` marks bounds that come from
    a closed-form identity (arcs) rather than from a subdivision criterion.
    """

    bound: float
    tolerance: float
    segments: int
    exact: bool

    @property
    def satisfied(self) -> bool:
        return self.bound <= self.tolerance

    def __str__(self) -> str:
        kind = "exact" if self.exact else "bounded"
        return f"{self.segments} segments, deviation <= {self.bound:.3e} ({kind}, tol {self.tolerance:.3e})"


class Curve2D(ABC):
    """A planar curve parameterised over ``t`` in ``[0, 1]``."""

    @abstractmethod
    def point(self, t: float) -> Point2:
        """Position at parameter ``t``."""

    @abstractmethod
    def derivative(self, t: float) -> Point2:
        """First derivative with respect to ``t``."""

    @abstractmethod
    def bounds(self) -> Tuple[float, float, float, float]:
        """Exact ``(min_x, min_y, max_x, max_y)`` of the true curve.

        Exact means the real extremes of the curve, not of a sampling of
        it -- an arc's bounding box includes the axis-crossing points it
        passes through even when no sample lands on them.
        """

    @abstractmethod
    def length(self) -> float:
        """Arc length."""

    @abstractmethod
    def reverse(self) -> "Curve2D":
        """The same geometry traversed in the opposite direction."""

    @abstractmethod
    def tessellate(self, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> List[Point2]:
        """Polyline within ``tolerance`` of the curve, endpoints included."""

    @abstractmethod
    def deviation_certificate(self, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> DeviationCertificate:
        """The proven deviation bound for :meth:`tessellate` at this tolerance."""

    @property
    def start(self) -> Point2:
        return self.point(0.0)

    @property
    def end(self) -> Point2:
        return self.point(1.0)

    def tangent(self, t: float) -> Point2:
        dx, dy = self.derivative(t)
        norm = math.hypot(dx, dy)
        if norm == 0.0:
            return (0.0, 0.0)
        return (dx / norm, dy / norm)

    def is_closed(self, tolerance: float = 1e-9) -> bool:
        return math.hypot(self.start[0] - self.end[0], self.start[1] - self.end[1]) <= tolerance


# --------------------------------------------------------------------------
# Line
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Line2D(Curve2D):
    """A straight segment. Tessellation is exact: deviation is identically zero."""

    p0: Point2
    p1: Point2

    def point(self, t: float) -> Point2:
        return (self.p0[0] + t * (self.p1[0] - self.p0[0]), self.p0[1] + t * (self.p1[1] - self.p0[1]))

    def derivative(self, t: float) -> Point2:
        return (self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])

    def bounds(self) -> Tuple[float, float, float, float]:
        return (
            min(self.p0[0], self.p1[0]),
            min(self.p0[1], self.p1[1]),
            max(self.p0[0], self.p1[0]),
            max(self.p0[1], self.p1[1]),
        )

    def length(self) -> float:
        return math.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])

    def reverse(self) -> "Line2D":
        return Line2D(self.p1, self.p0)

    def tessellate(self, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> List[Point2]:
        return [self.p0, self.p1]

    def deviation_certificate(self, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> DeviationCertificate:
        return DeviationCertificate(bound=0.0, tolerance=tolerance, segments=1, exact=True)


# --------------------------------------------------------------------------
# Circular arc
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Arc2D(Curve2D):
    """A circular arc, held exactly as center/radius/angles.

    This is the representation a hole, fillet, or bore keeps from the DXF
    entity all the way into the STEP file. Extruding one produces a true
    ``CYLINDRICAL_SURFACE``, and a CNC or slicer reading that file sees a
    real cylinder of the exact nominal diameter, not a faceted
    approximation of one.
    """

    center: Point2
    radius: float
    start_angle: float  # radians
    end_angle: float  # radians
    ccw: bool = True

    def __post_init__(self):
        if self.radius <= 0.0:
            raise ValueError(f"arc radius must be positive, got {self.radius}")

    @classmethod
    def full_circle(cls, center: Point2, radius: float, ccw: bool = True) -> "Arc2D":
        return cls(center, radius, 0.0, _TAU, ccw)

    @property
    def sweep(self) -> float:
        """Signed swept angle; magnitude is the total turn."""
        if self.ccw:
            delta = self.end_angle - self.start_angle
            if delta <= 0.0:
                delta += _TAU
        else:
            delta = self.end_angle - self.start_angle
            if delta >= 0.0:
                delta -= _TAU
        # A start/end pair that is exactly equal means a full circle, not a
        # zero-length arc; DXF CIRCLE entities arrive that way.
        if abs(delta) < 1e-15 and abs(self.end_angle - self.start_angle) >= _TAU - 1e-12:
            delta = _TAU if self.ccw else -_TAU
        return delta

    def angle_at(self, t: float) -> float:
        return self.start_angle + t * self.sweep

    def point(self, t: float) -> Point2:
        theta = self.angle_at(t)
        return (self.center[0] + self.radius * math.cos(theta), self.center[1] + self.radius * math.sin(theta))

    def derivative(self, t: float) -> Point2:
        theta = self.angle_at(t)
        s = self.sweep
        return (-self.radius * s * math.sin(theta), self.radius * s * math.cos(theta))

    def curvature(self, t: float = 0.0) -> float:
        return 1.0 / self.radius

    #: Unit vectors at the four axis angles, written down exactly. Going
    #: through cos/sin instead would introduce error precisely where the
    #: answer is meant to be exact: cos(pi/2) evaluates to 6.1e-17, not 0,
    #: because the float nearest pi/2 is not pi/2.
    _AXIS_DIRECTIONS = ((0.0, (1.0, 0.0)), (math.pi / 2.0, (0.0, 1.0)),
                        (math.pi, (-1.0, 0.0)), (3.0 * math.pi / 2.0, (0.0, -1.0)))

    def bounds(self) -> Tuple[float, float, float, float]:
        """Exact box: includes any axis extreme the arc actually sweeps through."""
        xs = [self.start[0], self.end[0]]
        ys = [self.start[1], self.end[1]]
        for axis_angle, (ux, uy) in self._AXIS_DIRECTIONS:
            if self._sweeps_through(axis_angle):
                xs.append(self.center[0] + self.radius * ux)
                ys.append(self.center[1] + self.radius * uy)
        return (min(xs), min(ys), max(xs), max(ys))

    def _sweeps_through(self, angle: float) -> bool:
        """Does the arc pass through ``angle`` (mod 2*pi)?"""
        sweep = self.sweep
        if sweep >= 0.0:
            offset = _normalize_angle(angle - self.start_angle)
            return offset <= sweep + 1e-12
        offset = _normalize_angle(self.start_angle - angle)
        return offset <= -sweep + 1e-12

    def length(self) -> float:
        return abs(self.sweep) * self.radius

    def reverse(self) -> "Arc2D":
        return Arc2D(self.center, self.radius, self.end_angle, self.start_angle, not self.ccw)

    def max_angular_step(self, tolerance: float) -> float:
        """Largest angular step whose chord stays within ``tolerance``.

        From the sagitta identity :math:`s = r(1 - \\cos(\\Delta/2))`,
        solved for :math:`\\Delta`. If the tolerance is at least the
        diameter the whole arc is within tolerance of any chord, so the
        step is unbounded and gets clamped to a half turn (keeping the
        chord well defined).
        """
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        ratio = 1.0 - tolerance / self.radius
        if ratio <= -1.0:
            return math.pi
        return 2.0 * math.acos(ratio)

    def segment_count(self, tolerance: float) -> int:
        return max(1, math.ceil(abs(self.sweep) / self.max_angular_step(tolerance)))

    def tessellate(self, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> List[Point2]:
        n = self.segment_count(tolerance)
        sweep = self.sweep
        # Evaluate each sample from the exact center/radius rather than by
        # accumulating rotations, so error does not compound along the arc
        # and the closing point of a full circle lands exactly on the first.
        pts = []
        for i in range(n + 1):
            theta = self.start_angle + sweep * (i / n)
            pts.append((self.center[0] + self.radius * math.cos(theta), self.center[1] + self.radius * math.sin(theta)))
        return pts

    def deviation_certificate(self, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> DeviationCertificate:
        n = self.segment_count(tolerance)
        step = abs(self.sweep) / n
        bound = self.radius * (1.0 - math.cos(step / 2.0))
        return DeviationCertificate(bound=bound, tolerance=tolerance, segments=n, exact=True)


# --------------------------------------------------------------------------
# Elliptical arc
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EllipseArc2D(Curve2D):
    """An elliptical arc, exact in center/axes/angles form.

    DXF ``ELLIPSE`` and SVG elliptical arc segments land here rather than
    being approximated by splines, which keeps them exportable as STEP
    ``ELLIPSE`` entities.
    """

    center: Point2
    major: Point2  # major semi-axis vector (direction and length)
    ratio: float  # minor/major axis length ratio, in (0, 1]
    start_param: float = 0.0
    end_param: float = _TAU
    ccw: bool = True

    def __post_init__(self):
        if not (0.0 < self.ratio <= 1.0):
            raise ValueError(f"axis ratio must be in (0, 1], got {self.ratio}")

    @property
    def major_length(self) -> float:
        return math.hypot(self.major[0], self.major[1])

    @property
    def minor_length(self) -> float:
        return self.major_length * self.ratio

    @property
    def sweep(self) -> float:
        if self.ccw:
            delta = self.end_param - self.start_param
            if delta <= 0.0:
                delta += _TAU
        else:
            delta = self.end_param - self.start_param
            if delta >= 0.0:
                delta -= _TAU
        if abs(delta) < 1e-15 and abs(self.end_param - self.start_param) >= _TAU - 1e-12:
            delta = _TAU if self.ccw else -_TAU
        return delta

    def point(self, t: float) -> Point2:
        theta = self.start_param + t * self.sweep
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        mx, my = self.major
        # Minor axis is the major rotated a quarter turn, scaled by ratio.
        nx, ny = -my * self.ratio, mx * self.ratio
        return (self.center[0] + mx * cos_t + nx * sin_t, self.center[1] + my * cos_t + ny * sin_t)

    def derivative(self, t: float) -> Point2:
        theta = self.start_param + t * self.sweep
        s = self.sweep
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        mx, my = self.major
        nx, ny = -my * self.ratio, mx * self.ratio
        return (s * (-mx * sin_t + nx * cos_t), s * (-my * sin_t + ny * cos_t))

    def bounds(self) -> Tuple[float, float, float, float]:
        """Exact box via the parametric extremes of a rotated ellipse.

        d/dt of the x-component vanishes at ``tan(theta) = n_x / m_x``,
        giving two candidate parameters a half turn apart; likewise for y.
        Only those actually inside the swept range count.
        """
        mx, my = self.major
        nx, ny = -my * self.ratio, mx * self.ratio
        xs = [self.start[0], self.end[0]]
        ys = [self.start[1], self.end[1]]

        for axis_m, axis_n, coords, index in ((mx, nx, xs, 0), (my, ny, ys, 1)):
            theta = math.atan2(axis_n, axis_m) if (axis_m, axis_n) != (0.0, 0.0) else 0.0
            for candidate in (theta, theta + math.pi):
                if self._sweeps_through(candidate):
                    coords.append(self.point(self._param_to_t(candidate))[index])

        return (min(xs), min(ys), max(xs), max(ys))

    def _param_to_t(self, param: float) -> float:
        sweep = self.sweep
        if sweep == 0.0:
            return 0.0
        if sweep > 0.0:
            return _normalize_angle(param - self.start_param) / sweep
        return -_normalize_angle(self.start_param - param) / -sweep

    def _sweeps_through(self, param: float) -> bool:
        sweep = self.sweep
        if sweep >= 0.0:
            return _normalize_angle(param - self.start_param) <= sweep + 1e-12
        return _normalize_angle(self.start_param - param) <= -sweep + 1e-12

    def length(self) -> float:
        """Arc length by adaptive Gauss-Legendre quadrature.

        No closed form exists (this is the elliptic integral of the second
        kind), so a 20-node rule is used per subinterval; for a smooth
        integrand like this its error falls far below the tolerances the
        rest of the pipeline works at.
        """
        return _gauss_legendre_length(self, subdivisions=16)

    def reverse(self) -> "EllipseArc2D":
        return EllipseArc2D(self.center, self.major, self.ratio, self.end_param, self.start_param, not self.ccw)

    def segment_count(self, tolerance: float) -> int:
        """From the linear-interpolation error bound.

        For an interval of width ``h`` in the parameter,
        ``|C - L| <= h^2 |C''| / 8`` componentwise. Here
        ``|C''| <= (a, b)`` with ``a``, ``b`` the semi-axis lengths, so the
        norm of the deviation is at most ``h^2 sqrt(a^2 + b^2) / 8``.
        """
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        a, b = self.major_length, self.minor_length
        second_derivative_norm = math.hypot(a, b)
        total = abs(self.sweep)
        if second_derivative_norm == 0.0:
            return 1
        h_max = math.sqrt(8.0 * tolerance / second_derivative_norm)
        return max(1, math.ceil(total / h_max))

    def tessellate(self, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> List[Point2]:
        n = self.segment_count(tolerance)
        return [self.point(i / n) for i in range(n + 1)]

    def deviation_certificate(self, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> DeviationCertificate:
        n = self.segment_count(tolerance)
        h = abs(self.sweep) / n
        bound = h * h * math.hypot(self.major_length, self.minor_length) / 8.0
        return DeviationCertificate(bound=bound, tolerance=tolerance, segments=n, exact=False)


# --------------------------------------------------------------------------
# Bezier / NURBS
# --------------------------------------------------------------------------


def _flatness(control: Sequence[Point2]) -> float:
    """Max distance from any control point to the chord of the first and last.

    This is a rigorous bound on the curve's deviation from that chord. A
    Bezier curve -- rational or not -- lies inside the convex hull of its
    control points, and distance-to-a-segment is convex, so its maximum
    over the hull is attained at a vertex of the hull, i.e. at one of the
    control points.
    """
    if len(control) < 3:
        return 0.0
    a, b = control[0], control[-1]
    return max(_distance_point_to_segment(p, a, b) for p in control[1:-1])


@dataclass(frozen=True)
class BezierCurve2D(Curve2D):
    """A rational or polynomial Bezier curve of any degree."""

    control: Tuple[Point2, ...]
    weights: Optional[Tuple[float, ...]] = None

    def __post_init__(self):
        if len(self.control) < 2:
            raise ValueError("a Bezier curve needs at least two control points")
        if self.weights is not None:
            if len(self.weights) != len(self.control):
                raise ValueError("weights and control points must have the same length")
            if any(w <= 0.0 for w in self.weights):
                raise ValueError("weights must be strictly positive")

    @property
    def degree(self) -> int:
        return len(self.control) - 1

    @property
    def is_rational(self) -> bool:
        return self.weights is not None and any(w != self.weights[0] for w in self.weights)

    def _homogeneous(self) -> List[Tuple[float, float, float]]:
        if self.weights is None:
            return [(p[0], p[1], 1.0) for p in self.control]
        return [(p[0] * w, p[1] * w, w) for p, w in zip(self.control, self.weights)]

    def point(self, t: float) -> Point2:
        pts = self._homogeneous()
        n = len(pts)
        for r in range(1, n):
            for i in range(n - r):
                a, b = pts[i], pts[i + 1]
                pts[i] = (
                    a[0] + t * (b[0] - a[0]),
                    a[1] + t * (b[1] - a[1]),
                    a[2] + t * (b[2] - a[2]),
                )
        x, y, w = pts[0]
        return (x / w, y / w)

    def derivative(self, t: float) -> Point2:
        """Derivative via the quotient rule on the homogeneous form."""
        h = self._homogeneous()
        n = self.degree
        # Hodograph of the homogeneous curve.
        dh = [(n * (h[i + 1][0] - h[i][0]), n * (h[i + 1][1] - h[i][1]), n * (h[i + 1][2] - h[i][2])) for i in range(n)]
        num = _de_casteljau3(h, t)
        dnum = _de_casteljau3(dh, t) if dh else (0.0, 0.0, 0.0)
        w = num[2]
        dw = dnum[2]
        return ((dnum[0] * w - num[0] * dw) / (w * w), (dnum[1] * w - num[1] * dw) / (w * w))

    def subdivide(self, t: float = 0.5) -> Tuple["BezierCurve2D", "BezierCurve2D"]:
        """Split at ``t`` with de Casteljau. Exact: both halves are the same curve."""
        h = self._homogeneous()
        n = len(h)
        left: List[Tuple[float, float, float]] = [h[0]]
        right: List[Tuple[float, float, float]] = [h[-1]]
        work = list(h)
        for r in range(1, n):
            for i in range(n - r):
                a, b = work[i], work[i + 1]
                work[i] = (
                    a[0] + t * (b[0] - a[0]),
                    a[1] + t * (b[1] - a[1]),
                    a[2] + t * (b[2] - a[2]),
                )
            left.append(work[0])
            right.append(work[n - r - 1])
        right.reverse()
        return (_from_homogeneous(left), _from_homogeneous(right))

    def bounds(self) -> Tuple[float, float, float, float]:
        """Convex-hull box.

        Contains the curve rigorously (hull property). It is not the
        tightest possible box, but it never excludes part of the curve,
        which is the property callers depend on.
        """
        xs = [p[0] for p in self.control]
        ys = [p[1] for p in self.control]
        return (min(xs), min(ys), max(xs), max(ys))

    def length(self) -> float:
        return _gauss_legendre_length(self, subdivisions=16)

    def reverse(self) -> "BezierCurve2D":
        return BezierCurve2D(
            tuple(reversed(self.control)),
            tuple(reversed(self.weights)) if self.weights is not None else None,
        )

    def _flatten(self, tolerance: float, depth: int = 0, max_depth: int = 32) -> List[Point2]:
        if depth >= max_depth or _flatness(self.control) <= tolerance:
            return [self.control[0], self.control[-1]]
        left, right = self.subdivide(0.5)
        head = left._flatten(tolerance, depth + 1, max_depth)
        tail = right._flatten(tolerance, depth + 1, max_depth)
        return head[:-1] + tail

    def tessellate(self, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> List[Point2]:
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        return self._flatten(tolerance)

    def achieved_flatness(self, tolerance: float) -> float:
        """Largest per-piece flatness bound left after refinement."""
        worst = 0.0
        stack = [(self, 0)]
        while stack:
            curve, depth = stack.pop()
            flat = _flatness(curve.control)
            if depth >= 32 or flat <= tolerance:
                worst = max(worst, flat)
                continue
            left, right = curve.subdivide(0.5)
            stack.append((left, depth + 1))
            stack.append((right, depth + 1))
        return worst

    def deviation_certificate(self, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> DeviationCertificate:
        points = self.tessellate(tolerance)
        return DeviationCertificate(
            bound=self.achieved_flatness(tolerance),
            tolerance=tolerance,
            segments=len(points) - 1,
            exact=False,
        )


def _de_casteljau3(points: Sequence[Tuple[float, float, float]], t: float) -> Tuple[float, float, float]:
    work = list(points)
    n = len(work)
    for r in range(1, n):
        for i in range(n - r):
            a, b = work[i], work[i + 1]
            work[i] = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]))
    return work[0]


def _from_homogeneous(points: Sequence[Tuple[float, float, float]]) -> BezierCurve2D:
    control = tuple((p[0] / p[2], p[1] / p[2]) for p in points)
    weights = tuple(p[2] for p in points)
    if all(abs(w - weights[0]) < 1e-15 for w in weights):
        return BezierCurve2D(control)
    return BezierCurve2D(control, weights)


@dataclass(frozen=True)
class NurbsCurve2D(Curve2D):
    """A NURBS curve: control points, knot vector, degree, optional weights.

    Evaluated with de Boor's algorithm in homogeneous coordinates, so
    rational curves are handled exactly rather than by pretending the
    weights are all one. For tessellation the curve is decomposed into
    Bezier segments by knot insertion -- an exact operation that changes
    the representation, not the geometry -- and each segment is flattened
    under the convex-hull bound.
    """

    control: Tuple[Point2, ...]
    knots: Tuple[float, ...]
    degree: int
    weights: Optional[Tuple[float, ...]] = None

    def __post_init__(self):
        expected = len(self.control) + self.degree + 1
        if len(self.knots) != expected:
            raise ValueError(
                f"knot vector must have len(control) + degree + 1 = {expected} entries, got {len(self.knots)}"
            )
        if self.degree < 1:
            raise ValueError("degree must be at least 1")
        if self.weights is not None and len(self.weights) != len(self.control):
            raise ValueError("weights and control points must have the same length")

    @property
    def domain(self) -> Tuple[float, float]:
        return (self.knots[self.degree], self.knots[len(self.control)])

    def _to_parameter(self, t: float) -> float:
        lo, hi = self.domain
        return lo + t * (hi - lo)

    def _find_span(self, u: float) -> int:
        n = len(self.control) - 1
        p = self.degree
        if u >= self.knots[n + 1]:
            return n
        if u <= self.knots[p]:
            return p
        low, high = p, n + 1
        mid = (low + high) // 2
        while u < self.knots[mid] or u >= self.knots[mid + 1]:
            if u < self.knots[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2
        return mid

    def point(self, t: float) -> Point2:
        u = self._to_parameter(t)
        span = self._find_span(u)
        p = self.degree
        weights = self.weights or (1.0,) * len(self.control)
        work = [
            (
                self.control[span - p + i][0] * weights[span - p + i],
                self.control[span - p + i][1] * weights[span - p + i],
                weights[span - p + i],
            )
            for i in range(p + 1)
        ]
        for r in range(1, p + 1):
            for i in range(p, r - 1, -1):
                left = self.knots[span + i - p]
                right = self.knots[span + i - r + 1]
                denom = right - left
                alpha = 0.0 if denom == 0.0 else (u - left) / denom
                a, b = work[i - 1], work[i]
                work[i] = (
                    a[0] + alpha * (b[0] - a[0]),
                    a[1] + alpha * (b[1] - a[1]),
                    a[2] + alpha * (b[2] - a[2]),
                )
        x, y, w = work[p]
        return (x / w, y / w)

    def derivative(self, t: float, h: Optional[float] = None) -> Point2:
        """Derivative by a central difference on the exact evaluator.

        The Bezier segments produced by :meth:`bezier_segments` carry
        analytic derivatives; this is the convenience path for callers
        that just want a tangent direction, and the step is scaled to
        balance truncation against round-off.
        """
        if h is None:
            h = 1e-6
        t0 = min(max(t - h, 0.0), 1.0)
        t1 = min(max(t + h, 0.0), 1.0)
        p0, p1 = self.point(t0), self.point(t1)
        span = t1 - t0
        if span == 0.0:
            return (0.0, 0.0)
        return ((p1[0] - p0[0]) / span, (p1[1] - p0[1]) / span)

    def bezier_segments(self) -> List[BezierCurve2D]:
        """Decompose into Bezier segments by knot insertion (exact).

        Raising every interior knot to full multiplicity ``p`` splits the
        spline at each break without moving the curve at all -- same
        geometry, different basis -- after which each span is an ordinary
        Bezier the flatness bound applies to.
        """
        p = self.degree
        weights = self.weights or (1.0,) * len(self.control)
        homogeneous = [(c[0] * w, c[1] * w, w) for c, w in zip(self.control, weights)]
        knots = list(self.knots)

        lo, hi = self.domain
        breakpoints = sorted({k for k in knots if lo < k < hi})
        for knot in breakpoints:
            multiplicity = knots.count(knot)
            for _ in range(p - multiplicity):
                homogeneous, knots = _insert_knot(homogeneous, knots, p, knot)

        # With every interior knot at multiplicity p, the control points
        # split into consecutive overlapping runs of p+1, each a Bezier
        # segment sharing its end point with the next.
        segments: List[BezierCurve2D] = []
        index = 0
        while index + p < len(homogeneous) and knots[index + p] < lo:
            index += 1  # skip any leading span outside the domain
        while index + p < len(homogeneous):
            segments.append(_from_homogeneous(homogeneous[index : index + p + 1]))
            index += p
        return segments

    def bounds(self) -> Tuple[float, float, float, float]:
        xs = [c[0] for c in self.control]
        ys = [c[1] for c in self.control]
        return (min(xs), min(ys), max(xs), max(ys))

    def length(self) -> float:
        return sum(seg.length() for seg in self.bezier_segments()) or _gauss_legendre_length(self, 16)

    def reverse(self) -> "NurbsCurve2D":
        lo, hi = self.knots[0], self.knots[-1]
        return NurbsCurve2D(
            tuple(reversed(self.control)),
            tuple(lo + hi - k for k in reversed(self.knots)),
            self.degree,
            tuple(reversed(self.weights)) if self.weights is not None else None,
        )

    def tessellate(self, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> List[Point2]:
        segments = self.bezier_segments()
        if not segments:
            return [self.point(0.0), self.point(1.0)]
        points = segments[0].tessellate(tolerance)
        for segment in segments[1:]:
            points.extend(segment.tessellate(tolerance)[1:])
        return points

    def deviation_certificate(self, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> DeviationCertificate:
        segments = self.bezier_segments()
        bound = max((s.achieved_flatness(tolerance) for s in segments), default=0.0)
        return DeviationCertificate(
            bound=bound,
            tolerance=tolerance,
            segments=len(self.tessellate(tolerance)) - 1,
            exact=False,
        )


def _insert_knot(
    homogeneous: Sequence[Tuple[float, float, float]], knots: Sequence[float], degree: int, u: float
) -> Tuple[List[Tuple[float, float, float]], List[float]]:
    """Boehm knot insertion: adds ``u`` once, leaving the curve unchanged."""
    knots = list(knots)
    pts = list(homogeneous)
    # Index of the span containing u.
    k = max(i for i in range(len(knots) - 1) if knots[i] <= u < knots[i + 1])

    new_pts: List[Tuple[float, float, float]] = pts[: k - degree + 1]
    for i in range(k - degree + 1, k + 1):
        denom = knots[i + degree] - knots[i]
        alpha = 0.0 if denom == 0.0 else (u - knots[i]) / denom
        a, b = pts[i - 1], pts[i]
        new_pts.append(
            (
                (1.0 - alpha) * a[0] + alpha * b[0],
                (1.0 - alpha) * a[1] + alpha * b[1],
                (1.0 - alpha) * a[2] + alpha * b[2],
            )
        )
    new_pts.extend(pts[k:])
    new_knots = knots[: k + 1] + [u] + knots[k + 1 :]
    return new_pts, new_knots


# 20-node Gauss-Legendre nodes/weights on [-1, 1], used for arc length.
_GL20_NODES = (
    -0.9931285991850949, -0.9639719272779138, -0.9122344282513259, -0.8391169718222188,
    -0.7463319064601508, -0.6360536807265150, -0.5108670019508271, -0.3737060887154196,
    -0.2277858511416451, -0.0765265211334973, 0.0765265211334973, 0.2277858511416451,
    0.3737060887154196, 0.5108670019508271, 0.6360536807265150, 0.7463319064601508,
    0.8391169718222188, 0.9122344282513259, 0.9639719272779138, 0.9931285991850949,
)
_GL20_WEIGHTS = (
    0.0176140071391521, 0.0406014298003869, 0.0626720483341091, 0.0832767415767048,
    0.1019301198172404, 0.1181945319615184, 0.1316886384491766, 0.1420961093183820,
    0.1491729864726037, 0.1527533871307258, 0.1527533871307258, 0.1491729864726037,
    0.1420961093183820, 0.1316886384491766, 0.1181945319615184, 0.1019301198172404,
    0.0832767415767048, 0.0626720483341091, 0.0406014298003869, 0.0176140071391521,
)


def _gauss_legendre_length(curve: Curve2D, subdivisions: int = 16) -> float:
    """Arc length as the integral of the speed, by composite Gauss-Legendre.

    A 20-node rule is exact for polynomials up to degree 39, so on each of
    the (smooth) subintervals here the quadrature error is far below the
    geometric tolerances the rest of the pipeline works to.
    """
    total = 0.0
    step = 1.0 / subdivisions
    for s in range(subdivisions):
        a = s * step
        b = a + step
        half = 0.5 * (b - a)
        mid = 0.5 * (a + b)
        acc = 0.0
        for node, weight in zip(_GL20_NODES, _GL20_WEIGHTS):
            dx, dy = curve.derivative(mid + half * node)
            acc += weight * math.hypot(dx, dy)
        total += acc * half
    return total


def tessellate_all(curves: Sequence[Curve2D], tolerance: float = DEFAULT_CHORD_TOLERANCE) -> List[Point2]:
    """Tessellate a chain of curves into one polyline, without duplicate joints."""
    points: List[Point2] = []
    for curve in curves:
        segment = curve.tessellate(tolerance)
        if points:
            segment = segment[1:]
        points.extend(segment)
    return points
