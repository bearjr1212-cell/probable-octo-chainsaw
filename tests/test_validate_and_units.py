"""Intake QA: units, and the three faults that stop a cutting job."""

import math

import ezdxf
import pytest

from blueprint23d import units, validate
from blueprint23d.annotations import Dimension
from blueprint23d.brep import Face2D, Loop2D
from blueprint23d.curves import Arc2D, Line2D
from blueprint23d.diagnostics import Code, Severity
from blueprint23d.units import Unit


def write_dxf(tmp_path, name, build, insunits=None):
    doc = ezdxf.new(setup=True)
    if insunits is not None:
        doc.header["$INSUNITS"] = insunits
    build(doc.modelspace())
    path = tmp_path / name
    doc.saveas(path)
    return path


def square(msp, size=100.0, height=60.0):
    msp.add_lwpolyline([(0, 0), (size, 0), (size, height), (0, height)], close=True)


# ==========================================================================
# Units
# ==========================================================================


def test_declared_header_units_are_believed(tmp_path):
    path = write_dxf(tmp_path, "mm.dxf", square, insunits=4)
    inference = units.infer_from_dxf(path)
    assert inference.unit is Unit.MILLIMETRE
    assert inference.decided
    assert units.report_units(inference).cuttable


def test_undeclared_units_fall_back_on_plausible_size(tmp_path):
    """A 250-unit part is a sensible plate in mm and a 6-metre one in inches."""
    path = write_dxf(tmp_path, "big.dxf", lambda m: square(m, 250.0, 180.0), insunits=0)
    inference = units.infer_from_dxf(path)
    assert inference.unit is Unit.MILLIMETRE

    small = write_dxf(tmp_path, "small.dxf", lambda m: square(m, 10.0, 6.0), insunits=0)
    assert units.infer_from_dxf(small).unit is Unit.INCH


def test_a_lone_size_guess_is_not_reported_as_certainty(tmp_path):
    """Share-of-evidence is not confidence when there is barely any evidence.

    A single plausibility heuristic agreeing with itself would otherwise
    read as 100%, which is exactly the case where a human should be asked.
    """
    path = write_dxf(tmp_path, "guess.dxf", lambda m: square(m, 250.0, 180.0), insunits=0)
    inference = units.infer_from_dxf(path)
    assert not inference.decided
    assert inference.confidence < 0.6

    report = units.report_units(inference)
    assert report.of(Code.UNITS_AMBIGUOUS)


def test_drawn_in_mm_but_dimensioned_in_inches_is_caught(tmp_path):
    """The mixed-unit drawing: geometry spans 101.6, the text claims 4.00.

    A ratio of 25.4 between what a dimension says and what it spans is not
    noise -- it is a drawing carrying two different unit systems, and it is
    how a part gets cut 25x wrong while every step looks fine.
    """

    def build(msp):
        msp.add_lwpolyline([(0, 0), (101.6, 0), (101.6, 50.8), (0, 50.8)], close=True)
        msp.add_linear_dim(base=(0, -10), p1=(0, 0), p2=(101.6, 0), text="4.00").render()

    path = write_dxf(tmp_path, "mixed.dxf", build, insunits=0)
    inference = units.infer_from_dxf(path)

    assert inference.unit is Unit.MILLIMETRE
    assert inference.contradicted
    assert inference.annotation_unit is Unit.INCH

    report = units.report_units(inference)
    contradiction = report.of(Code.UNITS_CONTRADICTED)
    assert contradiction and contradiction[0].severity is Severity.ERROR
    assert not report.cuttable


def test_agreeing_dimensions_raise_no_contradiction(tmp_path):
    def build(msp):
        square(msp)
        msp.add_linear_dim(base=(0, -10), p1=(0, 0), p2=(100, 0)).render()

    path = write_dxf(tmp_path, "honest.dxf", build, insunits=4)
    inference = units.infer_from_dxf(path)
    assert not inference.contradicted
    assert units.report_units(inference).cuttable


def test_inference_always_shows_its_working(tmp_path):
    path = write_dxf(tmp_path, "why.dxf", square, insunits=4)
    inference = units.infer_from_dxf(path)
    assert inference.evidence
    assert any("$INSUNITS" in e.detail for e in inference.evidence)
    assert "mm" in inference.describe()


def test_conversion_factors():
    assert units.convert_length(1.0, Unit.INCH) == 25.4
    assert units.convert_length(0.25, Unit.INCH) == pytest.approx(6.35)
    assert units.convert_length(1.0, Unit.METRE) == 1000.0
    assert units.convert_length(25.4, Unit.MILLIMETRE, Unit.INCH) == pytest.approx(1.0)


def test_dimension_ratio_evidence_needs_a_real_ratio():
    """Dimensions that simply agree settle nothing about which unit it is."""
    agreeing = [Dimension(kind=0, measurement=100.0, raw_text="<>", stated_value=100.0)]
    inference = units.infer(insunits=0, extent=100.0, dimensions=agreeing)
    assert inference.annotation_unit is None


# ==========================================================================
# Open contours
# ==========================================================================


def test_open_contour_is_located_and_measured(tmp_path):
    """The gap is tiny by definition -- that is why nobody sees it."""

    def build(msp):
        msp.add_line((0, 0), (100, 0))
        msp.add_line((100, 0), (100, 60))
        msp.add_line((100, 60), (0, 60))
        msp.add_line((0, 60), (0, 0.13))  # 0.13 short of closing

    path = write_dxf(tmp_path, "open.dxf", build)
    _, report = validate.validate_drawing(path)

    open_defects = report.of(Code.OPEN_CONTOUR)
    assert open_defects
    defect = open_defects[0]
    assert defect.measurements["gap"] == pytest.approx(0.13, abs=1e-9)
    assert defect.location.point is not None
    assert not report.cuttable


def test_clean_part_passes(tmp_path):
    def build(msp):
        square(msp)
        msp.add_circle((50, 30), 6.35)

    path = write_dxf(tmp_path, "clean.dxf", build)
    faces, report = validate.validate_drawing(path)
    assert faces
    assert report.cuttable, report.describe()


# ==========================================================================
# Duplicate and overlapping edges
# ==========================================================================


def test_exact_duplicate_edge_is_an_error():
    """Two entities on the same line means the machine cuts it twice."""
    curves = [
        Line2D((0.0, 0.0), (100.0, 0.0)),
        Line2D((0.0, 0.0), (100.0, 0.0)),
    ]
    report = validate.validate_curves(curves)
    duplicates = report.of(Code.DUPLICATE_EDGE)
    assert duplicates
    assert duplicates[0].severity is Severity.ERROR
    assert duplicates[0].measurements["overlap_length"] == pytest.approx(100.0)


def test_reversed_duplicate_is_still_a_duplicate():
    curves = [Line2D((0.0, 0.0), (100.0, 0.0)), Line2D((100.0, 0.0), (0.0, 0.0))]
    assert validate.validate_curves(curves).of(Code.DUPLICATE_EDGE)


def test_partial_overlap_is_found_and_measured():
    """One entity lying along part of another: invisible on screen."""
    curves = [Line2D((0.0, 0.0), (100.0, 0.0)), Line2D((20.0, 0.0), (70.0, 0.0))]
    report = validate.validate_curves(curves)
    overlaps = report.of(Code.OVERLAPPING_EDGE)
    assert overlaps
    assert overlaps[0].measurements["overlap_length"] == pytest.approx(50.0)


def test_parallel_but_separated_lines_are_not_overlaps():
    curves = [Line2D((0.0, 0.0), (100.0, 0.0)), Line2D((0.0, 5.0), (100.0, 5.0))]
    report = validate.validate_curves(curves)
    assert not report.of(Code.OVERLAPPING_EDGE, Code.DUPLICATE_EDGE)


def test_touching_end_to_end_lines_are_not_overlaps():
    """Collinear and touching at a point is how an outline joins, not a fault."""
    curves = [Line2D((0.0, 0.0), (50.0, 0.0)), Line2D((50.0, 0.0), (100.0, 0.0))]
    report = validate.validate_curves(curves)
    assert not report.of(Code.OVERLAPPING_EDGE, Code.DUPLICATE_EDGE)


def test_overlapping_arcs_are_found_with_exact_arc_length():
    """Concentric, same radius, overlapping sweep -- length is exact."""
    curves = [Arc2D((0.0, 0.0), 10.0, 0.0, math.pi), Arc2D((0.0, 0.0), 10.0, math.pi / 2, 1.5 * math.pi)]
    report = validate.validate_curves(curves)
    overlaps = report.of(Code.OVERLAPPING_EDGE, Code.DUPLICATE_EDGE)
    assert overlaps
    # The shared sweep is a quarter turn on r=10.
    assert overlaps[0].measurements["overlap_length"] == pytest.approx(10.0 * math.pi / 2)


def test_concentric_arcs_of_different_radius_are_not_overlaps():
    curves = [Arc2D((0.0, 0.0), 10.0, 0.0, math.pi), Arc2D((0.0, 0.0), 12.0, 0.0, math.pi)]
    assert not validate.validate_curves(curves).of(Code.OVERLAPPING_EDGE, Code.DUPLICATE_EDGE)


def test_zero_length_entity_is_reported():
    report = validate.validate_curves([Line2D((5.0, 5.0), (5.0, 5.0))])
    assert report.of(Code.ZERO_LENGTH_EDGE)


# ==========================================================================
# Self-intersection
# ==========================================================================


def test_bowtie_self_intersection_is_found_at_the_crossing():
    bowtie = Loop2D(
        [
            Line2D((0.0, 0.0), (100.0, 60.0)),
            Line2D((100.0, 60.0), (100.0, 0.0)),
            Line2D((100.0, 0.0), (0.0, 60.0)),
            Line2D((0.0, 60.0), (0.0, 0.0)),
        ]
    )
    report = validate.validate_face(Face2D.create(bowtie))
    hits = report.of(Code.SELF_INTERSECTION)
    assert hits
    point = hits[0].location.point
    assert point == pytest.approx((50.0, 30.0), abs=1e-9)


def test_a_simple_rectangle_does_not_self_intersect():
    rectangle = Loop2D(
        [
            Line2D((0.0, 0.0), (100.0, 0.0)),
            Line2D((100.0, 0.0), (100.0, 60.0)),
            Line2D((100.0, 60.0), (0.0, 60.0)),
            Line2D((0.0, 60.0), (0.0, 0.0)),
        ]
    )
    assert not validate.validate_face(Face2D.create(rectangle)).of(Code.SELF_INTERSECTION)


def test_a_circle_does_not_self_intersect():
    """Every sample on a tessellated circle is cocircular -- the degenerate case."""
    disc = Face2D.create(Loop2D([Arc2D.full_circle((0.0, 0.0), 25.0)]))
    assert not validate.validate_face(disc).of(Code.SELF_INTERSECTION)


def test_a_non_convex_part_does_not_self_intersect():
    l_shape = Loop2D(
        [
            Line2D((0.0, 0.0), (40.0, 0.0)),
            Line2D((40.0, 0.0), (40.0, 20.0)),
            Line2D((40.0, 20.0), (20.0, 20.0)),
            Line2D((20.0, 20.0), (20.0, 40.0)),
            Line2D((20.0, 40.0), (0.0, 40.0)),
            Line2D((0.0, 40.0), (0.0, 0.0)),
        ]
    )
    assert not validate.validate_face(Face2D.create(l_shape)).of(Code.SELF_INTERSECTION)


# ==========================================================================
# The whole intake check
# ==========================================================================


def test_report_carries_metrics_for_quoting(tmp_path):
    def build(msp):
        square(msp)
        msp.add_circle((50, 30), 6.35)

    path = write_dxf(tmp_path, "metrics.dxf", build)
    _, report = validate.validate_drawing(path)
    assert report.metrics["faces"] == 1
    assert report.metrics["holes"] == 1
    assert report.metrics["area"] == pytest.approx(100 * 60 - math.pi * 6.35**2, rel=1e-9)


def test_unreadable_file_reports_rather_than_raising(tmp_path):
    _, report = validate.validate_drawing(tmp_path / "missing.dxf")
    assert not report.cuttable
    assert report.at_least(Severity.CRITICAL)
