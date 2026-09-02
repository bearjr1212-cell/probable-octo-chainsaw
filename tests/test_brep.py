"""B-rep topology and closed-form area verification."""

import math

import pytest

from blueprint23d.brep import (
    CylindricalSurface,
    Face2D,
    LinearExtrusionSurface,
    Loop2D,
    PlaneSurface,
    extrude_face,
    solid_volume,
)
from blueprint23d.curves import Arc2D, BezierCurve2D, EllipseArc2D, Line2D


def rectangle_loop(w=4.0, h=3.0):
    return Loop2D(
        [
            Line2D((0.0, 0.0), (w, 0.0)),
            Line2D((w, 0.0), (w, h)),
            Line2D((w, h), (0.0, h)),
            Line2D((0.0, h), (0.0, 0.0)),
        ]
    )


def rounded_rectangle_loop(w=10.0, h=6.0, r=1.5):
    """Lines and tangent arcs -- the outline of an ordinary machined part."""
    return Loop2D(
        [
            Line2D((r, 0.0), (w - r, 0.0)),
            Arc2D((w - r, r), r, -math.pi / 2, 0.0),
            Line2D((w, r), (w, h - r)),
            Arc2D((w - r, h - r), r, 0.0, math.pi / 2),
            Line2D((w - r, h), (r, h)),
            Arc2D((r, h - r), r, math.pi / 2, math.pi),
            Line2D((0.0, h - r), (0.0, r)),
            Arc2D((r, r), r, math.pi, 3 * math.pi / 2),
        ]
    )


# --------------------------------------------------------------------------
# Areas: closed form, not tessellated
# --------------------------------------------------------------------------


def test_disc_area_is_exactly_pi_r_squared():
    """The headline precision property: no polygon ever enters the calculation.

    A tessellating pipeline reports the area of an inscribed n-gon, which
    is short of pi*r^2 by O(1/n^2). Green's theorem on the exact arc gives
    the real value, to the last bit.
    """
    for radius in (0.5, 1.0, 7.0, 123.456):
        disc = Loop2D([Arc2D.full_circle((3.0, -1.0), radius)])
        assert disc.area() == math.pi * radius * radius
        assert disc.is_area_exact()


def test_ellipse_area_is_exactly_pi_a_b():
    ellipse = Loop2D([EllipseArc2D((1.0, 2.0), (5.0, 0.0), 0.4)])
    assert ellipse.area() == math.pi * 5.0 * 2.0
    assert ellipse.is_area_exact()


def test_polygon_area_is_exact():
    assert rectangle_loop().area() == 12.0


def test_mixed_line_and_arc_area_matches_analytic_value():
    loop = rounded_rectangle_loop(10.0, 6.0, 1.5)
    # A rounded rectangle loses (4 - pi)r^2 relative to the sharp one.
    expected = 10.0 * 6.0 - (4.0 - math.pi) * 1.5**2
    assert loop.area() == pytest.approx(expected, abs=1e-12)
    assert loop.is_area_exact()


def test_face_area_subtracts_holes_exactly():
    plate = Face2D.create(rectangle_loop(), [Loop2D([Arc2D.full_circle((2.0, 1.5), 0.5)])])
    assert plate.area() == pytest.approx(12.0 - math.pi * 0.25, abs=1e-14)
    assert solid_volume(plate, 5.0) == pytest.approx((12.0 - math.pi * 0.25) * 5.0, abs=1e-13)


def test_spline_area_falls_back_to_quadrature_and_says_so():
    loop = Loop2D(
        [
            BezierCurve2D(((0.0, 0.0), (2.0, 4.0), (6.0, 4.0), (8.0, 0.0))),
            Line2D((8.0, 0.0), (0.0, 0.0)),
        ]
    )
    assert not loop.is_area_exact()  # honest about which path was taken
    assert loop.area() > 0.0


def test_loop_orientation_and_reversal():
    loop = rectangle_loop()
    assert loop.orientation() == 1
    assert loop.reverse().orientation() == -1
    assert loop.oriented(False).orientation() == -1
    assert math.isclose(loop.reverse().area(), loop.area())


def test_loop_rejects_a_gap():
    with pytest.raises(ValueError, match="not continuous"):
        Loop2D([Line2D((0.0, 0.0), (1.0, 0.0)), Line2D((5.0, 5.0), (0.0, 0.0))])


def test_loop_bounds_use_exact_curve_bounds():
    loop = Loop2D([Arc2D.full_circle((0.0, 0.0), 2.0)])
    assert loop.bounds() == (-2.0, -2.0, 2.0, 2.0)


def test_loop_length_is_exact_for_arcs():
    assert Loop2D([Arc2D.full_circle((0.0, 0.0), 3.0)]).length() == 2.0 * math.pi * 3.0


def test_face_point_classification():
    plate = Face2D.create(rectangle_loop(), [Loop2D([Arc2D.full_circle((2.0, 1.5), 0.5)])])
    assert plate.contains_point((0.5, 0.5)) == 1  # in material
    assert plate.contains_point((2.0, 1.5)) == -1  # inside the hole
    assert plate.contains_point((9.0, 9.0)) == -1  # outside entirely


# --------------------------------------------------------------------------
# Extrusion topology
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,face,genus",
    [
        ("plate", Face2D.create(rectangle_loop()), 0),
        ("plate with hole", Face2D.create(rectangle_loop(), [Loop2D([Arc2D.full_circle((2.0, 1.5), 0.5)])]), 1),
        ("disc", Face2D.create(Loop2D([Arc2D.full_circle((0.0, 0.0), 2.0)])), 0),
        ("rounded rectangle", Face2D.create(rounded_rectangle_loop()), 0),
        (
            "plate with two holes",
            Face2D.create(
                rectangle_loop(),
                [
                    Loop2D([Arc2D.full_circle((1.0, 1.5), 0.3)]),
                    Loop2D([Arc2D.full_circle((3.0, 1.5), 0.3)]),
                ],
            ),
            2,
        ),
    ],
)
def test_extrusion_is_topologically_valid(name, face, genus):
    """Generalised Euler-Poincare must balance exactly.

    V - E + 2F - L - 2S + 2G = 0. The naive V - E + F = 2 would fail on
    every holed case here, because the cap faces are annuli rather than
    discs -- which is precisely why the general form is the one checked.
    """
    solid = extrude_face(face, 5.0)
    assert solid.genus == genus
    assert solid.every_edge_used_twice(), f"{name}: shell is not closed"
    assert solid.euler_poincare_defect() == 0, f"{name}: {solid.validate()}"
    assert solid.validate() == []


def test_extruded_hole_becomes_a_true_cylinder():
    """The property the whole B-rep layer exists for."""
    plate = Face2D.create(rectangle_loop(), [Loop2D([Arc2D.full_circle((2.0, 1.5), 0.5)])])
    solid = extrude_face(plate, 5.0)

    cylinders = [f.surface for f in solid.faces() if isinstance(f.surface, CylindricalSurface)]
    assert len(cylinders) == 1
    # The nominal radius survives to the last bit -- this is what a CAM
    # system reads to pick a boring bar, so "close" is not good enough.
    assert cylinders[0].radius == 0.5
    assert cylinders[0].axis == (0.0, 0.0, 1.0)
    assert cylinders[0].origin == (2.0, 1.5, 0.0)


def test_fillets_become_cylinders_not_facets():
    solid = extrude_face(Face2D.create(rounded_rectangle_loop()), 5.0)
    cylinders = [f.surface for f in solid.faces() if isinstance(f.surface, CylindricalSurface)]
    assert len(cylinders) == 4
    assert all(c.radius == 1.5 for c in cylinders)


def test_freeform_edge_becomes_surface_of_linear_extrusion():
    loop = Loop2D(
        [
            BezierCurve2D(((0.0, 0.0), (2.0, 4.0), (6.0, 4.0), (8.0, 0.0))),
            Line2D((8.0, 0.0), (0.0, 0.0)),
        ]
    )
    solid = extrude_face(Face2D.create(loop), 2.0)
    swept = [f.surface for f in solid.faces() if isinstance(f.surface, LinearExtrusionSurface)]
    assert len(swept) == 1  # the spline edge, kept exact rather than approximated


def test_extruded_face_counts_are_as_expected():
    solid = extrude_face(Face2D.create(rectangle_loop()), 5.0)
    # four walls plus two caps
    assert len(solid.faces()) == 6
    assert sum(isinstance(f.surface, PlaneSurface) for f in solid.faces()) == 6


def test_extrusion_rejects_nonpositive_depth():
    with pytest.raises(ValueError):
        extrude_face(Face2D.create(rectangle_loop()), 0.0)
