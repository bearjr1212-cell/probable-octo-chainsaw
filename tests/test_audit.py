"""The consistency audit: does the drawing agree with itself?

The error class here is the expensive one precisely because nothing looks
wrong. The geometry is valid, the dimensions are legible, every step of the
job is correct, and the part comes out the wrong size. These tests build
drawings that lie in each of the ways a real one does.
"""

import math

import ezdxf
import pytest

from blueprint23d import audit
from blueprint23d.annotations import Dimension
from blueprint23d.brep import Face2D, Loop2D
from blueprint23d.curves import Arc2D, Line2D
from blueprint23d.diagnostics import Code, Severity


def drawing(tmp_path, name, polygon, *, circles=(), linear=(), diameter=()):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_lwpolyline(polygon, close=True)
    for centre, radius in circles:
        msp.add_circle(centre, radius)
    for kwargs in linear:
        msp.add_linear_dim(**kwargs).render()
    for kwargs in diameter:
        msp.add_diameter_dim(**kwargs).render()
    path = tmp_path / name
    doc.saveas(path)
    return path


PLATE = [(0, 0), (100, 0), (100, 60), (0, 60)]


# --------------------------------------------------------------------------
# The drawing agrees with itself
# --------------------------------------------------------------------------


def test_a_consistent_drawing_raises_nothing(tmp_path):
    path = drawing(
        tmp_path,
        "honest.dxf",
        PLATE,
        circles=[((50, 30), 6.35)],
        linear=[dict(base=(0, -15), p1=(0, 0), p2=(100, 0))],
        diameter=[dict(center=(50, 30), radius=6.35, angle=45)],
    )
    findings, report = audit.audit_drawing(path)
    assert findings
    assert report.cuttable, report.describe()
    assert not report.of(Code.DIMENSION_MISMATCH)


def test_dimensions_are_measured_against_real_geometry(tmp_path):
    path = drawing(
        tmp_path, "measured.dxf", PLATE, linear=[dict(base=(0, -15), p1=(0, 0), p2=(100, 0))]
    )
    findings, report = audit.audit_drawing(path)
    assert findings[0].matched
    assert findings[0].basis == "geometry"
    assert findings[0].measured == pytest.approx(100.0, abs=1e-9)
    assert report.metrics["dimensions_matched_to_geometry"] == 1


# --------------------------------------------------------------------------
# The drawing contradicts itself
# --------------------------------------------------------------------------


def test_undersize_hole_is_caught(tmp_path):
    """Called a quarter inch, drawn 0.05 under. The drill will not fit.

    This is the flagship check: the geometry is a perfectly valid circle
    and the dimension is a perfectly legible number, but they disagree.
    """
    path = drawing(
        tmp_path,
        "undersize.dxf",
        PLATE,
        circles=[((50, 30), 6.30)],
        diameter=[dict(center=(50, 30), radius=6.30, angle=45, text="12.70")],
    )
    findings, report = audit.audit_drawing(path)

    mismatches = report.of(Code.DIMENSION_MISMATCH)
    assert mismatches, report.describe()
    defect = mismatches[0]
    assert defect.severity is Severity.ERROR
    assert defect.measurements["stated"] == pytest.approx(12.70)
    assert defect.measurements["measured"] == pytest.approx(12.60)
    assert defect.measurements["difference"] == pytest.approx(-0.10)
    assert not report.cuttable


def test_overridden_text_that_contradicts_its_own_extension_lines(tmp_path):
    """Someone typed over the dimension. The geometry never changed."""
    path = drawing(
        tmp_path,
        "override.dxf",
        PLATE,
        linear=[dict(base=(0, -15), p1=(0, 0), p2=(100, 0), text="99.50")],
    )
    _, report = audit.audit_drawing(path)

    mismatches = report.of(Code.DIMENSION_MISMATCH)
    assert mismatches
    assert mismatches[0].measurements["stated"] == pytest.approx(99.5)
    assert mismatches[0].measurements["measured"] == pytest.approx(100.0)


def test_a_dimension_left_behind_by_an_edit_is_reported(tmp_path):
    """The part was narrowed to 97 and the dimension still spans 100.

    Falling back to the dimension's own extension lines would call this
    consistent -- the annotation agrees with itself perfectly. What is
    wrong is that it no longer touches the part, and the distance is the
    evidence.
    """
    path = drawing(
        tmp_path,
        "moved.dxf",
        [(0, 0), (97, 0), (97, 60), (0, 60)],
        linear=[dict(base=(0, -15), p1=(0, 0), p2=(100, 0))],
    )
    _, report = audit.audit_drawing(path)

    unmatched = report.of(Code.DIMENSION_UNMATCHED)
    assert unmatched, report.describe()
    assert unmatched[0].severity is Severity.WARNING
    assert unmatched[0].measurements["detachment"] == pytest.approx(3.0, abs=1e-6)
    assert "left behind" in unmatched[0].message


def test_the_location_of_a_mismatch_is_reported(tmp_path):
    path = drawing(
        tmp_path,
        "located.dxf",
        PLATE,
        circles=[((50, 30), 6.30)],
        diameter=[dict(center=(50, 30), radius=6.30, angle=45, text="12.70")],
    )
    _, report = audit.audit_drawing(path)
    defect = report.of(Code.DIMENSION_MISMATCH)[0]
    assert defect.location.point is not None
    assert defect.location.entity_handles  # findable in the source file


# --------------------------------------------------------------------------
# Tolerances and edge cases
# --------------------------------------------------------------------------


def test_drafting_roundoff_is_not_a_contradiction():
    """A dimension rounded for display is not a conflict."""
    finding = audit.Finding(
        dimension=Dimension(kind=0, measurement=100.0, raw_text="<>", stated_value=100.0),
        stated=100.0,
        measured=100.000001,
        basis="geometry",
        matched=True,
    )
    assert not finding.disagrees()


def test_a_real_disagreement_survives_the_tolerance():
    finding = audit.Finding(
        dimension=Dimension(kind=0, measurement=100.0, raw_text="99.5", stated_value=99.5),
        stated=99.5,
        measured=100.0,
        basis="geometry",
        matched=True,
    )
    assert finding.disagrees()
    assert finding.difference == pytest.approx(0.5)
    assert finding.relative == pytest.approx(0.5 / 99.5)


def test_text_with_no_number_states_nothing_checkable():
    finding = audit.Finding(
        dimension=Dimension(kind=0, measurement=8.0, raw_text="TYP", stated_value=None),
        stated=None,
        measured=8.0,
        basis="geometry",
        matched=True,
    )
    assert not finding.disagrees()
    assert "nothing checkable" in finding.describe()


def test_radial_dimension_matches_by_position_not_by_radius():
    """Matching on radius would assume the answer and never find a fault."""
    curves = [Arc2D.full_circle((50.0, 30.0), 6.30), Arc2D.full_circle((10.0, 10.0), 20.0)]
    dimension = Dimension(
        kind=4,
        measurement=6.30,
        raw_text="R6.35",
        stated_value=6.35,
        reference_points=((50.0, 36.30),),
    )
    arc = audit.match_radial(dimension, curves)
    assert arc is not None
    assert arc.radius == 6.30  # the one it points at, not the one it names


def test_a_drawing_with_no_dimensions_says_so(tmp_path):
    path = drawing(tmp_path, "bare.dxf", PLATE)
    findings, report = audit.audit_drawing(path)
    assert findings == []
    assert report.of(Code.DIMENSION_UNMATCHED)
    assert "no dimensions" in report.of(Code.DIMENSION_UNMATCHED)[0].message


def test_snap_finds_the_nearest_point_on_an_arc():
    curves = [Arc2D.full_circle((0.0, 0.0), 10.0)]
    snapped = audit.snap_to_geometry((12.0, 0.0), curves, tolerance=5.0)
    assert snapped == pytest.approx((10.0, 0.0), abs=1e-9)


def test_snap_refuses_when_nothing_is_close():
    curves = [Arc2D.full_circle((0.0, 0.0), 10.0)]
    assert audit.snap_to_geometry((500.0, 500.0), curves, tolerance=1.0) is None


def test_unreadable_file_reports_rather_than_raising(tmp_path):
    findings, report = audit.audit_drawing(tmp_path / "nope.dxf")
    assert findings == []
    assert not report.cuttable
