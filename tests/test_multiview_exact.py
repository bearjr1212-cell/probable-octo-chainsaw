"""Exact clipping, and multiview reconstruction with stated error."""

import math

import pytest

from blueprint23d.booleans2d import (
    X,
    Y,
    clip_face_to_halfplane,
    clip_face_to_slab,
    curve_axis_crossings,
)
from blueprint23d.brep import Face2D, Loop2D
from blueprint23d.curves import Arc2D, BezierCurve2D, EllipseArc2D, Line2D
from blueprint23d.multiview import (
    ViewSpec,
    as_axis_aligned_rectangle,
    reconstruct,
    thickness_view,
)


def rect_loop(w, h, x0=0.0, y0=0.0):
    return Loop2D(
        [
            Line2D((x0, y0), (x0 + w, y0)),
            Line2D((x0 + w, y0), (x0 + w, y0 + h)),
            Line2D((x0 + w, y0 + h), (x0, y0 + h)),
            Line2D((x0, y0 + h), (x0, y0)),
        ]
    )


def half_disc_area(radius, cut_x):
    """Exact area of the part of a disc with x <= cut_x."""
    theta = math.acos(max(-1.0, min(1.0, cut_x / radius)))
    segment = radius * radius * (theta - math.sin(theta) * math.cos(theta))
    return math.pi * radius * radius - segment


# --------------------------------------------------------------------------
# Curve / line intersection
# --------------------------------------------------------------------------


def test_arc_line_intersection_is_closed_form():
    circle = Arc2D.full_circle((0.0, 0.0), 5.0)
    params = curve_axis_crossings(circle, X, 3.0)
    assert len(params) == 2
    ys = sorted(circle.point(t)[1] for t in params)
    # 3-4-5: the crossings are exactly at y = +/-4.
    assert ys == pytest.approx([-4.0, 4.0], abs=1e-12)
    for t in params:
        assert circle.point(t)[0] == pytest.approx(3.0, abs=1e-14)


def test_arc_line_intersection_misses_cleanly():
    circle = Arc2D.full_circle((0.0, 0.0), 5.0)
    assert curve_axis_crossings(circle, X, 6.0) == []
    assert curve_axis_crossings(circle, X, 5.0) == []  # tangent, not a crossing


def test_line_intersection():
    line = Line2D((0.0, 0.0), (10.0, 10.0))
    params = curve_axis_crossings(line, X, 3.0)
    assert params == [pytest.approx(0.3)]
    assert line.point(params[0]) == pytest.approx((3.0, 3.0))


def test_ellipse_line_intersection():
    ellipse = EllipseArc2D((0.0, 0.0), (8.0, 0.0), 0.5)
    params = curve_axis_crossings(ellipse, X, 4.0)
    ys = sorted(ellipse.point(t)[1] for t in params)
    expected = 4.0 * math.sqrt(1.0 - 0.25)  # b * sqrt(1 - (x/a)^2), b = 4
    assert ys == pytest.approx([-expected, expected], abs=1e-9)


def test_bezier_intersection_lands_on_the_curve():
    curve = BezierCurve2D(((0.0, 0.0), (3.0, 10.0), (7.0, -10.0), (10.0, 0.0)))
    for t in curve_axis_crossings(curve, Y, 0.0):
        assert abs(curve.point(t)[1]) < 1e-12


# --------------------------------------------------------------------------
# Clipping
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cut", [2.5, 5.0, 7.5])
def test_clipping_a_rectangle_conserves_area(cut):
    face = Face2D.create(rect_loop(10.0, 6.0))
    clipped = clip_face_to_halfplane(face, X, cut, keep_below=True)
    assert clipped.area() == pytest.approx(cut * 6.0, abs=1e-12)


@pytest.mark.parametrize("cut", [0.0, 2.0, -3.0])
def test_clipping_a_disc_matches_the_analytic_segment(cut):
    disc = Face2D.create(Loop2D([Arc2D.full_circle((0.0, 0.0), 5.0)]))
    clipped = clip_face_to_halfplane(disc, X, cut, keep_below=True)
    assert clipped.area() == pytest.approx(half_disc_area(5.0, cut), abs=1e-12)


def test_a_clipped_arc_is_still_an_arc():
    """The property that makes clipping worth doing on curves at all.

    If clipping degraded arcs to polylines, the exact pipeline would end
    here and a clipped bore would stop being a cylinder.
    """
    disc = Face2D.create(Loop2D([Arc2D.full_circle((0.0, 0.0), 5.0)]))
    clipped = clip_face_to_halfplane(disc, X, 2.0, keep_below=True)

    arcs = [c for c in clipped.outer.curves if isinstance(c, Arc2D)]
    assert arcs, "the curved part of the boundary was lost"
    assert all(a.radius == 5.0 for a in arcs)  # bit-for-bit
    assert all(a.center == (0.0, 0.0) for a in arcs)
    assert clipped.is_area_exact()


def test_slab_clipping():
    face = Face2D.create(rect_loop(10.0, 6.0))
    assert clip_face_to_slab(face, X, 2.0, 7.0).area() == pytest.approx(30.0, abs=1e-12)

    disc = Face2D.create(Loop2D([Arc2D.full_circle((0.0, 0.0), 5.0)]))
    band = clip_face_to_slab(disc, X, -2.0, 2.0)
    theta = math.acos(2.0 / 5.0)
    segment = 25.0 * (theta - math.sin(theta) * math.cos(theta))
    assert band.area() == pytest.approx(math.pi * 25.0 - 2 * segment, abs=1e-12)


def test_clipping_keeps_holes():
    plate = Face2D.create(rect_loop(10.0, 6.0), [Loop2D([Arc2D.full_circle((5.0, 3.0), 1.5)])])
    clipped = clip_face_to_halfplane(plate, X, 8.0, keep_below=True)
    assert len(clipped.inners) == 1
    assert clipped.area() == pytest.approx(8.0 * 6.0 - math.pi * 1.5**2, abs=1e-12)


def test_clip_that_removes_everything_returns_none():
    face = Face2D.create(rect_loop(10.0, 6.0))
    assert clip_face_to_halfplane(face, X, -5.0, keep_below=True) is None


def test_clip_that_removes_nothing_returns_the_original():
    face = Face2D.create(rect_loop(10.0, 6.0))
    assert clip_face_to_halfplane(face, X, 99.0, keep_below=True).area() == pytest.approx(60.0)


# --------------------------------------------------------------------------
# Rectangle detection gates the exact path
# --------------------------------------------------------------------------


def test_rectangle_detection_is_strict():
    """A "nearly rectangular" view is not a rectangle.

    Taking the exact path on something that only looks like a box would
    silently produce a different part, so every disqualifying feature must
    actually disqualify.
    """
    assert as_axis_aligned_rectangle(Face2D.create(rect_loop(10.0, 5.0))) == (0.0, 10.0, 0.0, 5.0)

    trapezoid = Loop2D(
        [
            Line2D((0.0, 0.0), (10.0, 0.0)),
            Line2D((10.0, 0.0), (8.0, 5.0)),
            Line2D((8.0, 5.0), (0.0, 5.0)),
            Line2D((0.0, 5.0), (0.0, 0.0)),
        ]
    )
    assert as_axis_aligned_rectangle(Face2D.create(trapezoid)) is None

    holed = Face2D.create(rect_loop(10.0, 5.0), [Loop2D([Arc2D.full_circle((5.0, 2.5), 1.0)])])
    assert as_axis_aligned_rectangle(holed) is None

    disc = Face2D.create(Loop2D([Arc2D.full_circle((0.0, 0.0), 3.0)]))
    assert as_axis_aligned_rectangle(disc) is None


# --------------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------------


def test_constant_thickness_part_reconstructs_exactly():
    """The common case: a plate seen edge-on has a rectangular front view.

    The reconstruction is then a clip and an extrusion, both exact, so the
    result is a real B-rep -- holes included -- rather than a mesh.
    """
    top = ViewSpec(
        Face2D.create(
            rect_loop(100.0, 60.0),
            [
                Loop2D([Arc2D.full_circle((25.0, 30.0), 6.35)]),
                Loop2D([Arc2D.full_circle((75.0, 30.0), 6.35)]),
            ],
        ),
        "top",
    )
    result = reconstruct(top, thickness_view(0.0, 100.0, 8.0), name="plate")

    assert result.certificate.exact
    assert result.certificate.method == "exact-prismatic"
    assert result.certificate.deviation == 0.0

    expected = (100 * 60 - 2 * math.pi * 6.35**2) * 8
    assert result.volume() == pytest.approx(expected, abs=1e-9)

    assert result.solid is not None
    assert result.solid.validate() == []
    assert result.solid.genus == 2  # two through-holes


def test_exact_reconstruction_exports_holes_as_cylinders():
    import re

    from blueprint23d.step_writer import StepWriter, summarize_step, validate_step

    top = ViewSpec(
        Face2D.create(rect_loop(100.0, 60.0), [Loop2D([Arc2D.full_circle((50.0, 30.0), 6.35)])]),
        "top",
    )
    result = reconstruct(top, thickness_view(0.0, 100.0, 8.0), name="plate")
    text = StepWriter(name="plate").render(result.solid)

    assert validate_step(text) == []
    assert summarize_step(text)["CYLINDRICAL_SURFACE"] == 1
    assert "6.35" in set(re.findall(r"CYLINDRICAL_SURFACE\('',#\d+,([0-9.E+-]+)\)", text))


def test_front_view_narrower_than_the_part_clips_it():
    top = ViewSpec(Face2D.create(rect_loop(100.0, 60.0)), "top")
    result = reconstruct(top, thickness_view(20.0, 80.0, 8.0))
    assert result.certificate.exact
    assert result.volume() == pytest.approx(60.0 * 60.0 * 8.0, abs=1e-9)


def test_stepped_front_view_needs_only_two_slabs():
    """A shouldered part is prismatic within each step, so stepping is exact.

    The slab boundary lands on the discontinuity by construction, and the
    interior of each slab does not vary -- so this must report zero
    deviation, not subdivide chasing the step itself.
    """
    step = Loop2D(
        [
            Line2D((0.0, 0.0), (100.0, 0.0)),
            Line2D((100.0, 0.0), (100.0, 5.0)),
            Line2D((100.0, 5.0), (70.0, 5.0)),
            Line2D((70.0, 5.0), (70.0, 12.0)),
            Line2D((70.0, 12.0), (30.0, 12.0)),
            Line2D((30.0, 12.0), (30.0, 5.0)),
            Line2D((30.0, 5.0), (0.0, 5.0)),
            Line2D((0.0, 5.0), (0.0, 0.0)),
        ]
    )
    result = reconstruct(
        ViewSpec(Face2D.create(rect_loop(100.0, 60.0)), "top"),
        ViewSpec(Face2D.create(step), "front"),
        tolerance=1e-3,
    )
    assert result.certificate.slabs == 2
    assert result.certificate.deviation == 0.0
    assert result.volume() == pytest.approx(100 * 60 * 5 + 40 * 60 * 7, abs=1e-9)


def test_curved_front_view_converges_and_respects_its_bound():
    """A dome has a genuinely varying cross-section, so it is stepped.

    The point is that the stepping is *certified*: the reported deviation
    must honour the requested tolerance, and the volume must converge.
    """
    dome = Loop2D([Line2D((-20.0, 0.0), (20.0, 0.0)), Arc2D((0.0, 0.0), 20.0, 0.0, math.pi)])
    top = ViewSpec(Face2D.create(rect_loop(40.0, 40.0, -20.0, -20.0)), "top")
    true_volume = 40.0 * (math.pi * 400.0 / 2.0)

    errors = {}
    for tolerance in (1.0, 0.1, 0.01):
        result = reconstruct(
            top, ViewSpec(Face2D.create(dome), "front"), tolerance=tolerance, max_slabs=50000
        )
        assert not result.certificate.exact
        assert result.certificate.deviation <= tolerance, "reported bound exceeds the request"
        errors[tolerance] = abs(result.volume() - true_volume) / true_volume

    assert errors[0.01] < errors[1.0]
    assert errors[0.01] < 1e-4


def test_non_overlapping_views_are_reported_not_silently_empty():
    top = ViewSpec(Face2D.create(rect_loop(10.0, 10.0)), "top")
    with pytest.raises(ValueError, match="do not overlap"):
        reconstruct(top, thickness_view(500.0, 600.0, 5.0))
