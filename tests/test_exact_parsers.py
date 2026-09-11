"""Exact readers: geometry must survive the file, not be resampled from it.

Every test here asserts on the *identity* of what came back, not just its
shape. A circle must arrive as an ``Arc2D`` carrying the drawing's radius,
because the difference between that and a polyline with the right outline
is the whole difference between a CAD part and a picture of one.
"""

import math
import re
from collections import Counter

import ezdxf
import pytest

from blueprint23d.assembly import faces_from_curves, nest_loops, weld_curves
from blueprint23d.brep import Loop2D, extrude_face
from blueprint23d.curves import Arc2D, BezierCurve2D, EllipseArc2D, Line2D, NurbsCurve2D
from blueprint23d.parsers import dxf_exact, svg_exact
from blueprint23d.step_writer import StepWriter, validate_step


# --------------------------------------------------------------------------
# DXF
# --------------------------------------------------------------------------


def write_plate(tmp_path, hole_radius=6.35):
    """A plate with bulge-filleted corners, two drilled holes, and annotations."""
    doc = ezdxf.new()
    msp = doc.modelspace()
    # bulge 0.4142... = tan(90deg / 4): a quarter-turn fillet
    msp.add_lwpolyline(
        [(0, 0, 0), (100, 0, math.tan(math.pi / 8)), (100, 60, 0), (0, 60, math.tan(math.pi / 8))],
        format="xyb",
        close=True,
        dxfattribs={"layer": "OUTLINE"},
    )
    msp.add_circle((25, 30), hole_radius, dxfattribs={"layer": "OUTLINE"})
    msp.add_circle((75, 30), hole_radius, dxfattribs={"layer": "OUTLINE"})
    msp.add_text("100.00", dxfattribs={"layer": "DIMS"})
    path = tmp_path / "plate.dxf"
    doc.saveas(path)
    return path


def test_dxf_circle_stays_a_circle(tmp_path):
    doc = ezdxf.new()
    doc.modelspace().add_circle((10.0, 20.0), 6.35)
    path = tmp_path / "hole.dxf"
    doc.saveas(path)

    curves = dxf_exact.load_curves(path)
    assert len(curves) == 1
    arc = curves[0]
    assert isinstance(arc, Arc2D)
    # Bit-for-bit: this is the number a machinist reads off the drawing.
    assert arc.radius == 6.35
    assert arc.center == (10.0, 20.0)


def test_dxf_arc_angles_convert_from_degrees(tmp_path):
    doc = ezdxf.new()
    doc.modelspace().add_arc((0.0, 0.0), 5.0, start_angle=0.0, end_angle=90.0)
    path = tmp_path / "arc.dxf"
    doc.saveas(path)

    arc = dxf_exact.load_curves(path)[0]
    assert isinstance(arc, Arc2D)
    assert arc.radius == 5.0
    assert math.isclose(arc.sweep, math.pi / 2, rel_tol=1e-12)
    assert math.isclose(arc.length(), 5.0 * math.pi / 2, rel_tol=1e-12)


def test_dxf_polyline_bulge_becomes_an_exact_arc(tmp_path):
    """A rounded corner is one number in the file, not a run of segments.

    A flattening reader turns a bulge into short chords and the fillet
    radius is gone. Converted properly it is an arc, which is what makes
    the exported face cylindrical.
    """
    doc = ezdxf.new()
    doc.modelspace().add_lwpolyline(
        [(0, 0, 0), (10, 0, math.tan(math.pi / 8)), (10, 10, 0), (0, 10, 0)],
        format="xyb",
        close=True,
    )
    path = tmp_path / "bulge.dxf"
    doc.saveas(path)

    curves = dxf_exact.load_curves(path)
    arcs = [c for c in curves if isinstance(c, Arc2D)]
    assert len(arcs) == 1
    # tan(theta/4) = tan(pi/8) means a quarter turn.
    assert math.isclose(abs(arcs[0].sweep), math.pi / 2, rel_tol=1e-9)


def test_dxf_spline_defined_by_fit_points_stays_a_nurbs(tmp_path):
    """Fit-point splines store no control points; they must still stay exact.

    Reading the raw entity attributes finds an empty control-point list and
    would fall back to flattening a curve that was perfectly exact.
    """
    doc = ezdxf.new()
    doc.modelspace().add_spline([(0, 0), (10, 20), (30, 20), (40, 0)])
    path = tmp_path / "spline.dxf"
    doc.saveas(path)

    curves = dxf_exact.load_curves(path)
    assert len(curves) == 1
    spline = curves[0]
    assert isinstance(spline, NurbsCurve2D)
    assert spline.degree == 3
    assert spline.point(0.0) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert spline.point(1.0) == pytest.approx((40.0, 0.0), abs=1e-9)


def test_dxf_control_point_spline_stays_a_nurbs(tmp_path):
    doc = ezdxf.new()
    doc.modelspace().add_open_spline([(0, 0), (10, 20), (30, 20), (40, 0)], degree=3)
    path = tmp_path / "spline2.dxf"
    doc.saveas(path)
    curves = dxf_exact.load_curves(path)
    assert isinstance(curves[0], NurbsCurve2D)
    assert len(curves[0].control) == 4


def test_dxf_ellipse_stays_an_ellipse(tmp_path):
    doc = ezdxf.new()
    doc.modelspace().add_ellipse((0.0, 0.0), major_axis=(20.0, 0.0), ratio=0.5)
    path = tmp_path / "ellipse.dxf"
    doc.saveas(path)

    curve = dxf_exact.load_curves(path)[0]
    assert isinstance(curve, EllipseArc2D)
    assert curve.major_length == 20.0
    assert curve.minor_length == 10.0
    assert Loop2D([curve]).area() == pytest.approx(math.pi * 20.0 * 10.0, rel=1e-12)


def test_dxf_annotations_are_not_read_as_geometry(tmp_path):
    """Reading a dimension's text as part of the outline ruins the profile."""
    path = write_plate(tmp_path)
    curves = dxf_exact.load_curves(path)
    assert not any(isinstance(c, Line2D) and c.length() == 0 for c in curves)
    faces, _ = dxf_exact.load_faces(path)
    assert len(faces) == 1  # the plate, not the annotation layer


def test_dxf_plate_assembles_with_holes(tmp_path):
    """Area must match the analytic value, bulge arcs included.

    A bulge of tan(pi/8) is a quarter-turn arc spanning the whole edge, so
    each of the two 60-long ends becomes a circular segment bulging
    outward on a radius of 30/sin(45 deg). Checking against the closed-form
    total tests the bulge conversion numerically, not just structurally.
    """
    path = write_plate(tmp_path)
    face = dxf_exact.load_face(path, layer="OUTLINE")
    assert len(face.inners) == 2
    assert face.is_area_exact()

    radius = 30.0 / math.sin(math.pi / 4)
    segment = 0.5 * radius**2 * (math.pi / 2 - math.sin(math.pi / 2))
    expected = 100 * 60 + 2 * segment - 2 * math.pi * 6.35**2
    assert face.area() == pytest.approx(expected, rel=1e-12)


def test_dxf_hole_diameter_reaches_the_step_file(tmp_path):
    """End to end: the drilled size in the drawing is the size in the CAD file."""
    path = write_plate(tmp_path, hole_radius=6.35)
    solid = extrude_face(dxf_exact.load_face(path, layer="OUTLINE"), 12.0, name="plate")
    text = StepWriter(name="plate").render(solid)

    assert validate_step(text) == []
    radii = set(re.findall(r"CYLINDRICAL_SURFACE\('',#\d+,([0-9.E+-]+)\)", text))
    assert "6.35" in radii


def test_dxf_layer_filter(tmp_path):
    path = write_plate(tmp_path)
    outline = dxf_exact.load_curves(path, layer="OUTLINE")
    dims = dxf_exact.load_curves(path, layer="DIMS")
    assert len(outline) > 0
    assert len(dims) == 0  # only a TEXT entity there, which is annotation


def test_dxf_empty_file_reports_clearly(tmp_path):
    doc = ezdxf.new()
    path = tmp_path / "empty.dxf"
    doc.saveas(path)
    assert dxf_exact.load_curves(path) == []
    with pytest.raises(ValueError, match="no closed profile"):
        dxf_exact.load_face(path)


# --------------------------------------------------------------------------
# SVG
# --------------------------------------------------------------------------


SAMPLE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
  <rect x="10" y="10" width="100" height="60"/>
  <circle cx="60" cy="120" r="25"/>
  <ellipse cx="150" cy="120" rx="30" ry="15"/>
  <path d="M 10,160 C 30,190 70,190 90,160 L 10,160 Z"/>
  <g transform="translate(120,10) scale(2)"><circle cx="10" cy="10" r="5"/></g>
</svg>"""


def test_svg_circle_stays_circular(tmp_path):
    path = tmp_path / "s.svg"
    path.write_text(SAMPLE_SVG)
    curves = svg_exact.load_curves(path)
    kinds = Counter(type(c).__name__ for c in curves)
    assert kinds["Arc2D"] > 0
    assert kinds["EllipseArc2D"] > 0
    assert kinds["BezierCurve2D"] > 0

    radii = {c.radius for c in curves if isinstance(c, Arc2D)}
    assert 25.0 in radii


def test_svg_group_transform_scales_the_radius(tmp_path):
    """A circle inside scale(2) is a circle of twice the radius, still a circle."""
    path = tmp_path / "s.svg"
    path.write_text(SAMPLE_SVG)
    radii = {c.radius for c in svg_exact.load_curves(path) if isinstance(c, Arc2D)}
    assert 10.0 in radii  # r=5 under scale(2)


def test_svg_areas_are_closed_form(tmp_path):
    path = tmp_path / "s.svg"
    path.write_text(SAMPLE_SVG)
    faces, _ = svg_exact.load_faces(path)
    areas = sorted((f.area() for f in faces), reverse=True)

    assert any(math.isclose(a, 6000.0, rel_tol=1e-12) for a in areas)
    assert any(math.isclose(a, math.pi * 25 * 25, rel_tol=1e-12) for a in areas)
    assert any(math.isclose(a, math.pi * 30 * 15, rel_tol=1e-12) for a in areas)
    assert any(math.isclose(a, math.pi * 100, rel_tol=1e-12) for a in areas)


def test_svg_produces_no_degenerate_faces(tmp_path):
    path = tmp_path / "s.svg"
    path.write_text(SAMPLE_SVG)
    faces, _ = svg_exact.load_faces(path)
    assert all(f.area() > 1e-9 for f in faces)


def test_svg_bezier_survives_as_a_bezier(tmp_path):
    path = tmp_path / "s.svg"
    path.write_text(SAMPLE_SVG)
    beziers = [c for c in svg_exact.load_curves(path) if isinstance(c, BezierCurve2D)]
    assert beziers and beziers[0].degree == 3


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def test_weld_joins_shuffled_reversed_curves():
    """CAD exports give no ordering guarantee; welding must not care."""
    curves = [
        Line2D((10.0, 0.0), (10.0, 10.0)),
        Line2D((0.0, 0.0), (10.0, 0.0)),
        Line2D((0.0, 10.0), (0.0, 0.0)),
        Line2D((0.0, 10.0), (10.0, 10.0)),  # reversed relative to the walk
    ]
    loops, open_chains = weld_curves(curves)
    assert len(loops) == 1
    assert open_chains == []
    assert loops[0].area() == pytest.approx(100.0)


def test_weld_reports_open_chains_instead_of_forcing_them_closed():
    curves = [Line2D((0.0, 0.0), (1.0, 0.0)), Line2D((5.0, 5.0), (6.0, 5.0))]
    loops, open_chains = weld_curves(curves)
    assert loops == []
    assert len(open_chains) == 2


def test_nesting_assigns_holes_and_islands_by_parity():
    outer = Loop2D([Arc2D.full_circle((0.0, 0.0), 10.0)])
    hole = Loop2D([Arc2D.full_circle((0.0, 0.0), 6.0)])
    island = Loop2D([Arc2D.full_circle((0.0, 0.0), 2.0)])

    faces = nest_loops([outer, hole, island])
    assert len(faces) == 2  # the ring, and the island as its own face

    ring = max(faces, key=lambda f: f.area())
    assert len(ring.inners) == 1
    assert ring.area() == pytest.approx(math.pi * (100.0 - 36.0), rel=1e-12)

    island_face = min(faces, key=lambda f: f.area())
    assert island_face.area() == pytest.approx(math.pi * 4.0, rel=1e-12)


def test_faces_from_curves_end_to_end():
    curves = [
        Line2D((0.0, 0.0), (10.0, 0.0)),
        Line2D((10.0, 0.0), (10.0, 10.0)),
        Line2D((10.0, 10.0), (0.0, 10.0)),
        Line2D((0.0, 10.0), (0.0, 0.0)),
        Arc2D.full_circle((5.0, 5.0), 2.0),
    ]
    faces, open_chains = faces_from_curves(curves)
    assert open_chains == []
    assert len(faces) == 1
    assert faces[0].area() == pytest.approx(100.0 - math.pi * 4.0, rel=1e-12)
