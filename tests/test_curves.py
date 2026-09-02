"""Verification of the exact curve layer and its deviation bounds.

The claim under test is not "the tessellation looks fine" but "the
tessellation is provably within the stated tolerance of the true curve".
So each test measures the *actual* deviation by densely sampling the
analytic curve and comparing against the polyline, then checks it against
the bound the certificate promised.
"""

import math

import pytest

from blueprint23d.curves import (
    Arc2D,
    BezierCurve2D,
    EllipseArc2D,
    Line2D,
    NurbsCurve2D,
    tessellate_all,
)

TAU = 2.0 * math.pi


def distance_to_segment(p, a, b):
    abx, aby = b[0] - a[0], b[1] - a[1]
    denom = abx * abx + aby * aby
    if denom == 0.0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / denom))
    return math.hypot(p[0] - (a[0] + t * abx), p[1] - (a[1] + t * aby))


def measured_deviation(curve, polyline, samples=4000):
    """True max distance from the analytic curve to its polyline approximation."""
    worst = 0.0
    for i in range(samples + 1):
        p = curve.point(i / samples)
        worst = max(worst, min(distance_to_segment(p, polyline[j], polyline[j + 1]) for j in range(len(polyline) - 1)))
    return worst


# --------------------------------------------------------------------------
# Lines
# --------------------------------------------------------------------------


def test_line_tessellation_is_exact():
    line = Line2D((1.0, 2.0), (5.0, 7.0))
    cert = line.deviation_certificate(1e-9)
    assert cert.bound == 0.0
    assert cert.segments == 1
    assert cert.exact
    assert line.tessellate(1e-9) == [(1.0, 2.0), (5.0, 7.0)]
    assert math.isclose(line.length(), math.hypot(4.0, 5.0))


# --------------------------------------------------------------------------
# Arcs: the sagitta bound should be exactly attained, not merely respected
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "radius,tolerance",
    [(5.0, 1e-3), (5.0, 1e-5), (0.5, 1e-4), (100.0, 1e-2), (1e-3, 1e-6)],
)
def test_arc_deviation_bound_holds_and_is_tight(radius, tolerance):
    arc = Arc2D.full_circle((0.0, 0.0), radius)
    cert = arc.deviation_certificate(tolerance)
    points = arc.tessellate(tolerance)
    measured = measured_deviation(arc, points, samples=6000)

    assert cert.bound <= tolerance, "certificate must meet the requested tolerance"
    assert measured <= cert.bound * (1.0 + 1e-6), "measured deviation must respect the bound"
    # The sagitta identity is exact, so the bound should be attained, not
    # merely satisfied. A loose bound here would mean wasted segments.
    assert measured >= cert.bound * 0.99, f"bound not tight: {measured} vs {cert.bound}"


def test_arc_segment_count_follows_inverse_square_root_law():
    """Chordal error is quadratic in step, so segments scale as tol^(-1/2)."""
    counts = {}
    for tolerance in (1e-3, 1e-4, 1e-5, 1e-6):
        counts[tolerance] = Arc2D.full_circle((0.0, 0.0), 10.0).segment_count(tolerance)

    ratios = [
        counts[1e-4] / counts[1e-3],
        counts[1e-5] / counts[1e-4],
        counts[1e-6] / counts[1e-5],
    ]
    for ratio in ratios:
        assert math.isclose(ratio, math.sqrt(10.0), rel_tol=0.02)


def test_arc_geometry_is_exact_not_sampled():
    circle = Arc2D.full_circle((3.0, -2.0), 5.0)
    # Circumference from the analytic formula: no quadrature error at all.
    assert circle.length() == 2.0 * math.pi * 5.0
    assert circle.bounds() == (-2.0, -7.0, 8.0, 3.0)

    quarter = Arc2D((0.0, 0.0), 1.0, 0.0, math.pi / 2)
    assert quarter.bounds() == (0.0, 0.0, 1.0, 1.0)
    assert math.isclose(quarter.length(), math.pi / 2)


def test_arc_bounds_capture_axis_extremes_between_samples():
    """The extreme point is on the curve even when no vertex lands there."""
    arc = Arc2D((0.0, 0.0), 1.0, -0.3, 0.3)  # sweeps through angle 0
    minx, miny, maxx, maxy = arc.bounds()
    assert maxx == 1.0  # exact tangent point, not max(cos(-0.3), cos(0.3))
    assert maxx > max(arc.start[0], arc.end[0])


def test_arc_tessellation_endpoints_land_on_the_curve():
    arc = Arc2D((1.0, 1.0), 2.0, 0.4, 2.9)
    points = arc.tessellate(1e-4)
    assert points[0] == pytest.approx(arc.start, abs=1e-15)
    assert points[-1] == pytest.approx(arc.end, abs=1e-15)
    # Every sample sits on the true circle to machine precision.
    for x, y in points:
        assert math.isclose(math.hypot(x - 1.0, y - 1.0), 2.0, rel_tol=1e-14)


def test_full_circle_closes_exactly():
    circle = Arc2D.full_circle((0.0, 0.0), 7.0)
    points = circle.tessellate(1e-4)
    assert points[0] == pytest.approx(points[-1], abs=1e-14)


def test_arc_reverse_traverses_same_geometry():
    arc = Arc2D((0.0, 0.0), 3.0, 0.2, 1.9)
    backward = arc.reverse()
    assert backward.start == pytest.approx(arc.end)
    assert backward.end == pytest.approx(arc.start)
    assert math.isclose(backward.length(), arc.length())


def test_arc_rejects_nonsense():
    with pytest.raises(ValueError):
        Arc2D((0.0, 0.0), 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        Arc2D.full_circle((0.0, 0.0), 1.0).segment_count(0.0)


# --------------------------------------------------------------------------
# Ellipses
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tolerance", [1e-2, 1e-3, 1e-4])
def test_ellipse_deviation_bound_holds(tolerance):
    ellipse = EllipseArc2D((0.0, 0.0), (4.0, 0.0), 0.5)
    cert = ellipse.deviation_certificate(tolerance)
    measured = measured_deviation(ellipse, ellipse.tessellate(tolerance), samples=6000)
    assert cert.bound <= tolerance
    assert measured <= cert.bound


def test_ellipse_bounds_are_exact():
    ellipse = EllipseArc2D((0.0, 0.0), (4.0, 0.0), 0.5)
    assert ellipse.bounds() == pytest.approx((-4.0, -2.0, 4.0, 2.0), abs=1e-12)


def test_ellipse_with_unit_ratio_is_a_circle():
    ellipse = EllipseArc2D((0.0, 0.0), (3.0, 0.0), 1.0)
    for i in range(50):
        x, y = ellipse.point(i / 49)
        assert math.isclose(math.hypot(x, y), 3.0, rel_tol=1e-14)


def test_ellipse_rejects_bad_ratio():
    with pytest.raises(ValueError):
        EllipseArc2D((0.0, 0.0), (1.0, 0.0), 0.0)
    with pytest.raises(ValueError):
        EllipseArc2D((0.0, 0.0), (1.0, 0.0), 1.5)


# --------------------------------------------------------------------------
# Bezier and NURBS
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tolerance", [1e-2, 1e-4, 1e-6])
def test_bezier_flatness_bound_holds(tolerance):
    curve = BezierCurve2D(((0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)))
    cert = curve.deviation_certificate(tolerance)
    measured = measured_deviation(curve, curve.tessellate(tolerance), samples=6000)
    assert cert.bound <= tolerance
    assert measured <= cert.bound


def test_rational_bezier_represents_a_circle_exactly():
    """The standard weighted quarter-circle must be a circle to machine precision.

    This is the property that makes rational curves worth supporting: a
    conic is represented exactly, not fitted.
    """
    w = 1.0 / math.sqrt(2.0)
    quarter = BezierCurve2D(((1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), (1.0, w, 1.0))
    for i in range(201):
        x, y = quarter.point(i / 200)
        assert math.isclose(math.hypot(x, y), 1.0, rel_tol=1e-14)


def test_bezier_subdivision_is_geometry_preserving():
    curve = BezierCurve2D(((0.0, 0.0), (1.0, 4.0), (5.0, 4.0), (6.0, 0.0)))
    left, right = curve.subdivide(0.37)
    for i in range(101):
        s = i / 100
        assert left.point(s) == pytest.approx(curve.point(0.37 * s), abs=1e-12)
        assert right.point(s) == pytest.approx(curve.point(0.37 + 0.63 * s), abs=1e-12)


def test_bezier_rejects_bad_weights():
    with pytest.raises(ValueError):
        BezierCurve2D(((0.0, 0.0), (1.0, 1.0)), (1.0, -1.0))
    with pytest.raises(ValueError):
        BezierCurve2D(((0.0, 0.0),))


def test_nurbs_bezier_decomposition_is_exact():
    """Knot insertion changes the representation, never the geometry."""
    control = ((0.0, 0.0), (1.0, 3.0), (3.0, 3.0), (4.0, 0.0), (6.0, -2.0), (8.0, 1.0))
    knots = (0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0)
    curve = NurbsCurve2D(control, knots, 3)

    segments = curve.bezier_segments()
    assert len(segments) == 3

    for i in range(301):
        t = i / 300
        u = t * len(segments)
        index = min(int(u), len(segments) - 1)
        assert segments[index].point(u - index) == pytest.approx(curve.point(t), abs=1e-12)


def test_nurbs_single_span_yields_one_bezier():
    curve = NurbsCurve2D(
        ((0.0, 0.0), (1.0, 2.0), (3.0, 2.0), (4.0, 0.0)),
        (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
        3,
    )
    segments = curve.bezier_segments()
    assert len(segments) == 1
    assert segments[0].control == curve.control


def test_nurbs_endpoints_interpolate_clamped_control_points():
    control = ((0.0, 0.0), (1.0, 3.0), (3.0, 3.0), (4.0, 0.0))
    curve = NurbsCurve2D(control, (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0), 3)
    assert curve.point(0.0) == pytest.approx(control[0])
    assert curve.point(1.0) == pytest.approx(control[-1])


def test_nurbs_rejects_inconsistent_knot_vector():
    with pytest.raises(ValueError):
        NurbsCurve2D(((0.0, 0.0), (1.0, 1.0)), (0.0, 0.0, 1.0), 1)


def test_nurbs_tessellation_respects_tolerance():
    control = ((0.0, 0.0), (1.0, 3.0), (3.0, 3.0), (4.0, 0.0), (6.0, -2.0), (8.0, 1.0))
    knots = (0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0)
    curve = NurbsCurve2D(control, knots, 3)
    for tolerance in (1e-2, 1e-4):
        cert = curve.deviation_certificate(tolerance)
        assert cert.bound <= tolerance
        measured = measured_deviation(curve, curve.tessellate(tolerance), samples=3000)
        assert measured <= max(cert.bound, 1e-12)


def test_tessellate_all_joins_without_duplicating_vertices():
    chain = [Line2D((0.0, 0.0), (1.0, 0.0)), Line2D((1.0, 0.0), (1.0, 1.0))]
    assert tessellate_all(chain, 1e-6) == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
