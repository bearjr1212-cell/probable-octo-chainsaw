"""DWG input and the structured diagnostics everything reports through."""

import json
import shutil

import ezdxf
import pytest

from blueprint23d.annotations import Dimension, extract_dimensions, parse_stated_value
from blueprint23d.diagnostics import Code, Location, Report, Severity, merge
from blueprint23d.parsers import dwg

HAS_DWG = shutil.which("dwg2dxf") is not None or shutil.which("ODAFileConverter") is not None
needs_dwg = pytest.mark.skipif(not HAS_DWG, reason="no DWG converter installed")


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def test_severity_orders_correctly():
    assert Severity.INFO < Severity.WARNING < Severity.ERROR < Severity.CRITICAL


def test_report_verdict_turns_on_errors_not_warnings():
    """Warnings are the operator's judgement call; errors stop the job."""
    report = Report(source="part.dxf")
    report.add(Code.OPEN_CONTOUR, Severity.WARNING, "small gap")
    assert report.cuttable

    report.add(Code.SELF_INTERSECTION, Severity.ERROR, "outline crosses itself")
    assert not report.cuttable
    assert report.worst is Severity.ERROR


def test_defects_carry_location_and_measurements():
    report = Report()
    report.add(
        Code.OPEN_CONTOUR,
        Severity.ERROR,
        "contour does not close",
        Location(point=(45.2, 12.8), entity_handles=("2841", "2903"), layer="OUTLINE"),
        gap=0.13,
    )
    defect = report.defects[0]
    assert defect.measurements["gap"] == 0.13
    assert "45.200" in defect.location.describe()
    assert "OUTLINE" in defect.location.describe()
    assert "#2841" in defect.location.describe()


def test_every_defect_offers_a_remedy_where_one_is_known():
    report = Report()
    report.add(Code.OPEN_CONTOUR, Severity.ERROR, "gap")
    assert report.defects[0].remedy
    assert "close the gap" in report.defects[0].remedy.lower()


def test_report_serialises_to_stable_json():
    """The codes are API: an integration branches on these strings."""
    report = Report(source="part.dxf")
    report.add(Code.SELF_INTERSECTION, Severity.ERROR, "crosses", Location(point=(1.0, 2.0)))
    data = json.loads(report.to_json())

    assert data["cuttable"] is False
    assert data["worst_severity"] == "error"
    assert data["defects"][0]["code"] == "SELF_INTERSECTION"
    assert data["defects"][0]["location"]["point"] == [1.0, 2.0]


def test_defects_sort_worst_first():
    report = Report()
    report.add(Code.ZERO_LENGTH_EDGE, Severity.INFO, "tiny")
    report.add(Code.SELF_INTERSECTION, Severity.CRITICAL, "bad")
    report.add(Code.OPEN_CONTOUR, Severity.WARNING, "gap")
    assert [d.severity for d in report.sorted_defects()][0] is Severity.CRITICAL


def test_reports_merge():
    a = Report(source="x.dxf")
    a.add(Code.OPEN_CONTOUR, Severity.WARNING, "gap")
    a.metrics["area"] = 12.0
    b = Report()
    b.add(Code.SELF_INTERSECTION, Severity.ERROR, "crosses")
    b.metrics["perimeter"] = 40.0

    combined = merge(a, b)
    assert len(combined.defects) == 2
    assert combined.metrics == {"area": 12.0, "perimeter": 40.0}
    assert not combined.cuttable


def test_query_helpers():
    report = Report()
    report.add(Code.OPEN_CONTOUR, Severity.WARNING, "a")
    report.add(Code.OPEN_CONTOUR, Severity.ERROR, "b")
    report.add(Code.DUPLICATE_EDGE, Severity.INFO, "c")
    assert len(report.of(Code.OPEN_CONTOUR)) == 2
    assert len(report.at_least(Severity.WARNING)) == 2


# --------------------------------------------------------------------------
# Dimension text parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,measurement,expected",
    [
        ("<>", 25.4, 25.4),          # automatic: shows the measurement
        ("", 12.0, 12.0),
        ("99.50", 100.0, 99.5),      # override that contradicts the geometry
        ("Ø6.35", 6.35, 6.35),       # diameter symbol
        ("R3.0", 3.0, 3.0),
        ("25.4 REF", 25.4, 25.4),    # reference suffix
        ("100±0.1", 100.0, 100.0),   # tolerance
        ("2X Ø5", 5.0, 5.0),         # a count prefix is not the dimension
        ("TYP", 8.0, None),          # states nothing checkable
        ("SEE NOTE 4", 8.0, 4.0),    # a known limitation: picks up the note number
    ],
)
def test_stated_value_parsing(text, measurement, expected):
    assert parse_stated_value(text, measurement) == expected


def test_dimension_flags_an_override():
    auto = Dimension(kind=0, measurement=100.0, raw_text="<>", stated_value=100.0)
    override = Dimension(kind=0, measurement=100.0, raw_text="99.50", stated_value=99.5)
    assert not auto.is_overridden
    assert override.is_overridden


def test_radial_dimension_converts_diameter_to_radius():
    diameter = Dimension(kind=3, measurement=12.7, raw_text="<>", stated_value=12.7)
    radius = Dimension(kind=4, measurement=6.35, raw_text="<>", stated_value=6.35)
    assert diameter.nominal_radius == pytest.approx(6.35)
    assert radius.nominal_radius == pytest.approx(6.35)


def test_extract_dimensions_reads_measurement_and_override(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 60), (0, 60)], close=True)
    msp.add_linear_dim(base=(0, -10), p1=(0, 0), p2=(100, 0)).render()
    # An override that claims something the geometry does not support.
    msp.add_linear_dim(base=(0, 70), p1=(0, 60), p2=(100, 60), text="99.50").render()
    path = tmp_path / "dims.dxf"
    doc.saveas(path)

    dimensions = extract_dimensions(ezdxf.readfile(str(path)).modelspace())
    assert len(dimensions) == 2

    honest = [d for d in dimensions if not d.is_overridden]
    lying = [d for d in dimensions if d.is_overridden]
    assert honest and honest[0].measurement == pytest.approx(100.0)
    assert lying and lying[0].stated_value == pytest.approx(99.5)
    assert lying[0].measurement == pytest.approx(100.0)
    assert lying[0].reference_points  # we know where on the part it points


# --------------------------------------------------------------------------
# DWG
# --------------------------------------------------------------------------


@pytest.fixture
def dwg_file(tmp_path):
    """A DWG written by round-tripping a known DXF through LibreDWG."""
    import subprocess

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 60), (0, 60)], close=True, dxfattribs={"layer": "OUTLINE"})
    msp.add_circle((25, 30), 6.35, dxfattribs={"layer": "OUTLINE"})
    msp.add_circle((75, 30), 6.35, dxfattribs={"layer": "OUTLINE"})
    source = tmp_path / "src.dxf"
    doc.saveas(source)

    target = tmp_path / "part.dwg"
    if shutil.which("dxf2dwg") is None:
        pytest.skip("dxf2dwg not available to build a DWG fixture")
    subprocess.run(["dxf2dwg", "-y", "-o", str(target), str(source)], capture_output=True, timeout=120)
    if not target.exists():
        pytest.skip("could not produce a DWG fixture")
    return target


def test_converter_discovery_reports_licence():
    """The licence is part of the answer: it decides how it may be used."""
    text = dwg.describe_backends()
    assert "LibreDWG" in text
    assert "ODA File Converter" in text
    if HAS_DWG:
        assert "not linked" in text  # arm's-length invocation is the whole point


@needs_dwg
def test_dwg_geometry_survives_exactly(dwg_file):
    """Conversion must not cost precision: the bore is still 6.35."""
    faces, _, report = dwg.load_faces(dwg_file)
    assert faces, report.describe()

    face = max(faces, key=lambda f: f.area())
    assert len(face.inners) == 2
    assert face.is_area_exact()

    radii = [c.radius for loop in face.inners for c in loop.curves if hasattr(c, "radius")]
    assert radii == [6.35, 6.35]  # bit-for-bit through DWG


@needs_dwg
def test_lost_layers_are_reported_not_silently_empty(dwg_file):
    """LibreDWG drops layer names; filtering on one must not look like an empty file.

    Measured behaviour, not speculation: the round trip returns every
    entity on layer ''. Returning nothing at all would read as "the
    drawing is empty" instead of "your layer filter cannot work here".
    """
    faces, _, report = dwg.load_faces(dwg_file, layer="OUTLINE")

    lost = report.of(Code.CONVERSION_METADATA_LOST)
    assert lost, "layer loss went unreported"
    assert lost[0].severity is Severity.ERROR
    assert lost[0].remedy

    # And the geometry still came back, rather than being filtered to nothing.
    assert faces


@needs_dwg
def test_dwg_reaches_step_with_real_cylinders(dwg_file):
    import re

    from blueprint23d import extrude_face
    from blueprint23d.step_writer import StepWriter, summarize_step, validate_step

    faces, _, _ = dwg.load_faces(dwg_file)
    solid = extrude_face(max(faces, key=lambda f: f.area()), 8.0, name="from_dwg")
    text = StepWriter(name="from_dwg").render(solid)

    assert validate_step(text) == []
    assert summarize_step(text)["CYLINDRICAL_SURFACE"] == 2
    assert "6.35" in set(re.findall(r"CYLINDRICAL_SURFACE\('',#\d+,([0-9.E+-]+)\)", text))


def test_missing_file_reports_rather_than_raising(tmp_path):
    result = dwg.convert_to_dxf(tmp_path / "nope.dwg")
    assert not result.ok
    assert result.report.of(Code.FILE_UNREADABLE)


def test_oversized_file_is_refused_before_conversion(tmp_path):
    """A converter is a C program; do not hand it absurd input."""
    fake = tmp_path / "huge.dwg"
    fake.write_bytes(b"\0" * 2048)
    result = dwg.convert_to_dxf(fake, max_bytes=1024)
    assert not result.ok
    assert result.report.of(Code.FILE_TOO_LARGE)


def test_load_face_raises_with_the_blocking_reason(tmp_path):
    fake = tmp_path / "broken.dwg"
    fake.write_bytes(b"not a dwg at all")
    with pytest.raises(ValueError):
        dwg.load_face(fake)
