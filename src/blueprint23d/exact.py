"""Filtered exact predicates and certified interval arithmetic.

Every topological decision in a geometry kernel -- which side of a line a
point falls on, whether a point is inside a circumcircle, whether a loop
winds clockwise -- is a *sign* question. Answer one of those signs wrongly
and the failure is not a small numerical error: the topology becomes
inconsistent, and downstream code produces a self-intersecting solid, an
inverted face, or an infinite loop. Floating point answers these questions
wrongly whenever the true value is smaller than the accumulated rounding
error, which is exactly what happens on the near-degenerate inputs real
blueprints are full of (tangent arcs, collinear edges, coincident vertices).

This module answers them *provably* correctly, using the standard
two-stage arrangement from CGAL and Shewchuk:

1. **Static filter.** Evaluate in floating point and compare the result
   against a rigorous a-priori bound on the accumulated rounding error. If
   the magnitude exceeds the bound, the sign is certified and we are done
   -- this is the fast path taken by the overwhelming majority of calls.
2. **Exact fallback.** Otherwise, re-evaluate in exact rational arithmetic
   (:class:`fractions.Fraction`). Every finite ``float`` is exactly a
   dyadic rational, so the rational evaluation carries *no* error at all
   and its sign is the true sign, by construction.

The correctness of the result therefore does not depend on the filter
constant being tight -- only its speed does. The constants below are
Shewchuk's, multiplied by a safety factor: an over-large bound merely
sends more calls down the exact path, while an over-small one would be a
correctness bug. Erring toward the exact path is the safe direction.

References:
    J. R. Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast
    Robust Geometric Predicates", Discrete & Computational Geometry 18(3),
    1997.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Optional, Sequence, Tuple

try:  # pragma: no cover - exercised by whichever build the tests run on
    from ._native import predicates as _native
except ImportError:  # pragma: no cover
    _native = None

#: Which implementation is answering predicate queries: ``"c"`` when the
#: compiled kernel is available, ``"python"`` otherwise. The two are
#: required to agree on every input; tests/test_native.py enforces that
#: against adversarial cases, using the Python path as the oracle.
BACKEND = "c" if _native is not None else "python"

Point2 = Tuple[float, float]

#: Unit roundoff for IEEE-754 binary64: the maximum relative error of a
#: single correctly-rounded operation is bounded by this.
UNIT_ROUNDOFF = 2.0**-53

#: Safety factor applied to every static filter bound. Larger => more
#: calls fall through to the exact path => slower but never less correct.
_FILTER_SAFETY = 8.0

# Shewchuk's error bounds for the floating-point evaluations below.
_ORIENT2D_BOUND = (3.0 + 16.0 * UNIT_ROUNDOFF) * UNIT_ROUNDOFF * _FILTER_SAFETY
_INCIRCLE_BOUND = (10.0 + 96.0 * UNIT_ROUNDOFF) * UNIT_ROUNDOFF * _FILTER_SAFETY


# --------------------------------------------------------------------------
# Predicates
# --------------------------------------------------------------------------


def orient2d(a: Point2, b: Point2, c: Point2) -> int:
    """Sign of the orientation determinant of the triangle ``a, b, c``.

    Returns ``+1`` if ``a -> b -> c`` turns counter-clockwise, ``-1`` if
    clockwise, and ``0`` if the three points are *exactly* collinear. The
    returned sign is always the true sign of

    .. math:: \\det \\begin{pmatrix} a_x - c_x & a_y - c_y \\\\
                                     b_x - c_x & b_y - c_y \\end{pmatrix}

    evaluated on the exact rational values of the inputs -- never an
    artifact of rounding.
    """
    if _native is not None:
        return _native.orient2d(a, b, c)
    return orient2d_python(a, b, c)


def incircle(a: Point2, b: Point2, c: Point2, d: Point2) -> int:
    """Sign of the in-circle determinant for ``d`` against circle ``abc``.

    Returns ``+1`` if ``d`` lies strictly inside the circle through ``a``,
    ``b``, ``c`` (which must be given counter-clockwise), ``-1`` if
    strictly outside, and ``0`` if the four points are exactly cocircular.
    This is the predicate that makes Delaunay triangulation well defined;
    a wrong sign here produces a non-Delaunay, potentially non-planar
    triangulation.

    The compiled kernel resolves the easy majority with a filtered
    floating-point evaluation and falls back to exact expansion arithmetic
    in C; the pure-Python path falls back to rational arithmetic. Both are
    exact, and tests/test_native.py holds them to identical answers.
    """
    if _native is not None:
        return _native.incircle(a, b, c, d)
    return incircle_python(a, b, c, d)


def predicate_stats() -> dict:
    """Call counts and exact-fallback counts, or ``{}`` on the Python backend.

    Exactness is not free, and this makes the price visible: a healthy
    workload shows the adaptive/deferred counts as a small fraction of
    total calls. A high fallback ratio means the input is genuinely
    degenerate (many collinear or cocircular configurations), which is
    itself worth knowing about a blueprint.
    """
    return _native.stats() if _native is not None else {}


def reset_predicate_stats() -> None:
    if _native is not None:
        _native.reset_stats()


def orient2d_python(a: Point2, b: Point2, c: Point2) -> int:
    """Reference implementation of :func:`orient2d`, in pure Python.

    Kept as an independent oracle for the compiled kernel: it reaches the
    same answers by a different route (rational fallback rather than
    Shewchuk expansions), so agreement between the two is meaningful
    evidence rather than a tautology.
    """
    detleft = (a[0] - c[0]) * (b[1] - c[1])
    detright = (a[1] - c[1]) * (b[0] - c[0])
    det = detleft - detright

    # When the two products have strictly opposite signs (or one is zero)
    # the subtraction cannot cancel, so the floating-point sign is already
    # the true sign and no bound is needed.
    if detleft > 0.0:
        if detright <= 0.0:
            return _sign(det)
        detsum = detleft + detright
    elif detleft < 0.0:
        if detright >= 0.0:
            return _sign(det)
        detsum = -detleft - detright
    else:
        return _sign(det)

    if abs(det) >= _ORIENT2D_BOUND * detsum:
        return _sign(det)

    return _orient2d_exact(a, b, c)


def _orient2d_exact(a: Point2, b: Point2, c: Point2) -> int:
    ax, ay = Fraction(a[0]), Fraction(a[1])
    bx, by = Fraction(b[0]), Fraction(b[1])
    cx, cy = Fraction(c[0]), Fraction(c[1])
    det = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
    return (det > 0) - (det < 0)


def incircle_python(a: Point2, b: Point2, c: Point2, d: Point2) -> int:
    """Reference implementation of :func:`incircle`, in pure Python."""
    adx, ady = a[0] - d[0], a[1] - d[1]
    bdx, bdy = b[0] - d[0], b[1] - d[1]
    cdx, cdy = c[0] - d[0], c[1] - d[1]

    bdxcdy, cdxbdy = bdx * cdy, cdx * bdy
    cdxady, adxcdy = cdx * ady, adx * cdy
    adxbdy, bdxady = adx * bdy, bdx * ady

    alift = adx * adx + ady * ady
    blift = bdx * bdx + bdy * bdy
    clift = cdx * cdx + cdy * cdy

    det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady)

    permanent = (
        (abs(bdxcdy) + abs(cdxbdy)) * alift
        + (abs(cdxady) + abs(adxcdy)) * blift
        + (abs(adxbdy) + abs(bdxady)) * clift
    )
    if abs(det) >= _INCIRCLE_BOUND * permanent:
        return _sign(det)

    return _incircle_exact(a, b, c, d)


def _incircle_exact(a: Point2, b: Point2, c: Point2, d: Point2) -> int:
    ax, ay = Fraction(a[0]) - Fraction(d[0]), Fraction(a[1]) - Fraction(d[1])
    bx, by = Fraction(b[0]) - Fraction(d[0]), Fraction(b[1]) - Fraction(d[1])
    cx, cy = Fraction(c[0]) - Fraction(d[0]), Fraction(c[1]) - Fraction(d[1])
    det = (
        (ax * ax + ay * ay) * (bx * cy - cx * by)
        + (bx * bx + by * by) * (cx * ay - ax * cy)
        + (cx * cx + cy * cy) * (ax * by - bx * ay)
    )
    return (det > 0) - (det < 0)


def _sign(x: float) -> int:
    return (x > 0.0) - (x < 0.0)


# --------------------------------------------------------------------------
# Predicate-derived geometry
# --------------------------------------------------------------------------


def collinear(a: Point2, b: Point2, c: Point2) -> bool:
    """True iff the three points are exactly collinear."""
    return orient2d(a, b, c) == 0


def segments_properly_intersect(p1: Point2, p2: Point2, q1: Point2, q2: Point2) -> bool:
    """True iff segments ``p1p2`` and ``q1q2`` cross at a single interior point.

    Shared endpoints and collinear overlaps return False -- those are
    topological adjacency, not a crossing, and callers need to tell the
    two apart.
    """
    d1 = orient2d(q1, q2, p1)
    d2 = orient2d(q1, q2, p2)
    d3 = orient2d(p1, p2, q1)
    d4 = orient2d(p1, p2, q2)
    return d1 * d2 < 0 and d3 * d4 < 0


def signed_area2(points: Sequence[Point2]) -> Fraction:
    """Exact twice-signed-area (shoelace) of a closed polygon.

    Computed in rational arithmetic, so the sign is exact even for
    slivers whose floating-point area underflows to zero. Positive means
    counter-clockwise.
    """
    n = len(points)
    if n < 3:
        return Fraction(0)
    total = Fraction(0)
    for i in range(n):
        x1, y1 = Fraction(points[i][0]), Fraction(points[i][1])
        x2, y2 = Fraction(points[(i + 1) % n][0]), Fraction(points[(i + 1) % n][1])
        total += x1 * y2 - x2 * y1
    return total


def polygon_orientation(points: Sequence[Point2]) -> int:
    """``+1`` counter-clockwise, ``-1`` clockwise, ``0`` degenerate. Exact."""
    area = signed_area2(points)
    return (area > 0) - (area < 0)


def point_in_polygon(point: Point2, polygon: Sequence[Point2]) -> int:
    """Locate ``point`` against a closed polygon. Exact.

    Returns ``+1`` strictly inside, ``-1`` strictly outside, ``0`` exactly
    on the boundary. Uses a crossing-number test whose every branch is
    decided by :func:`orient2d`, so vertices lying exactly on the ray --
    the case that breaks naive implementations -- are handled by the
    half-open edge rule rather than by perturbation.
    """
    n = len(polygon)
    if n < 3:
        return -1

    px, py = point
    inside = False
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]

        # On-boundary test: collinear and within the segment's bounding box.
        if orient2d(a, b, point) == 0:
            if min(a[0], b[0]) <= px <= max(a[0], b[0]) and min(a[1], b[1]) <= py <= max(a[1], b[1]):
                return 0

        # Half-open rule in y: an edge counts if it strictly straddles the
        # horizontal ray, counting the lower endpoint and not the upper, so
        # a vertex exactly on the ray is counted exactly once.
        if (a[1] > py) != (b[1] > py):
            # Which side of the edge is the point on? Orient so the test is
            # a single exact sign rather than a division.
            side = orient2d(a, b, point)
            if (b[1] > a[1]) == (side > 0):
                inside = not inside

    return 1 if inside else -1


# --------------------------------------------------------------------------
# Certified interval arithmetic
# --------------------------------------------------------------------------


def _down(x: float) -> float:
    """Next float toward -inf: a rigorous lower bound on a rounded result."""
    return math.nextafter(x, -math.inf)


def _up(x: float) -> float:
    """Next float toward +inf: a rigorous upper bound on a rounded result."""
    return math.nextafter(x, math.inf)


class Interval:
    """A machine interval that provably encloses a real value.

    Python gives no access to the IEEE rounding mode, so every operation
    here computes in round-to-nearest and then widens the result outward
    by one ULP with :func:`math.nextafter`. A correctly-rounded result is
    within half an ULP of the exact value, so widening by a full ULP in
    each direction is a rigorous enclosure -- conservative by roughly a
    factor of two, and never wrong.

    The point of carrying these around is that a bound computed with
    intervals is a *theorem*, not an estimate: if
    ``curve_deviation_bound.hi < tol`` then the deviation really is below
    ``tol``, rounding included.
    """

    __slots__ = ("lo", "hi")

    def __init__(self, lo: float, hi: Optional[float] = None):
        if hi is None:
            hi = lo
        if lo > hi:
            raise ValueError(f"empty interval [{lo}, {hi}]")
        self.lo = lo
        self.hi = hi

    def __repr__(self) -> str:
        return f"Interval({self.lo!r}, {self.hi!r})"

    @property
    def width(self) -> float:
        return _up(self.hi - self.lo)

    @property
    def midpoint(self) -> float:
        return self.lo + (self.hi - self.lo) / 2.0

    def __add__(self, other: "Interval | float") -> "Interval":
        o = _as_interval(other)
        return Interval(_down(self.lo + o.lo), _up(self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other: "Interval | float") -> "Interval":
        o = _as_interval(other)
        return Interval(_down(self.lo - o.hi), _up(self.hi - o.lo))

    def __rsub__(self, other: "Interval | float") -> "Interval":
        return _as_interval(other) - self

    def __mul__(self, other: "Interval | float") -> "Interval":
        o = _as_interval(other)
        products = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return Interval(_down(min(products)), _up(max(products)))

    __rmul__ = __mul__

    def __truediv__(self, other: "Interval | float") -> "Interval":
        o = _as_interval(other)
        if o.lo <= 0.0 <= o.hi:
            raise ZeroDivisionError("interval divisor straddles zero")
        quotients = (self.lo / o.lo, self.lo / o.hi, self.hi / o.lo, self.hi / o.hi)
        return Interval(_down(min(quotients)), _up(max(quotients)))

    def sqrt(self) -> "Interval":
        if self.hi < 0.0:
            raise ValueError("sqrt of a strictly negative interval")
        lo = math.sqrt(self.lo) if self.lo > 0.0 else 0.0
        return Interval(_down(lo) if lo > 0.0 else 0.0, _up(math.sqrt(self.hi)))

    def abs(self) -> "Interval":
        if self.lo >= 0.0:
            return Interval(self.lo, self.hi)
        if self.hi <= 0.0:
            return Interval(-self.hi, -self.lo)
        return Interval(0.0, max(-self.lo, self.hi))

    def contains_zero(self) -> bool:
        return self.lo <= 0.0 <= self.hi

    def sign(self) -> Optional[int]:
        """``+1``/``-1`` if certified, ``None`` if the enclosure straddles zero."""
        if self.lo > 0.0:
            return 1
        if self.hi < 0.0:
            return -1
        return None

    def hull(self, other: "Interval") -> "Interval":
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))


def _as_interval(value: "Interval | float") -> Interval:
    return value if isinstance(value, Interval) else Interval(float(value))
