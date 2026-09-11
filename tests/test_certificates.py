"""What a reported number is worth, and proving the file is the file.

The failure these tests guard against is not a wrong number. It is a right
number printed to more digits than it has, handed to somebody who quotes
against it. A radius read from a DXF and a radius recovered from a scan
look identical on screen and are worth completely different amounts, and
the certificate has to say which one it is holding.
"""

import math
import os
import subprocess
import sys

import ezdxf
import pytest

from blueprint23d import certificates as certs
from blueprint23d.brep import Face2D, Loop2D, extrude_face
from blueprint23d.certificates import BOUNDED, ESTIMATED, EXACT
from blueprint23d.curves import Arc2D, BezierCurve2D, Line2D
from blueprint23d.fitting import FitReport
from blueprint23d.step_writer import write_step


def rectangle(x0, y0, x1, y1):
    return Loop2D(
        [
            Line2D((x0, y0), (x1, y0)),
            Line2D((x1, y0), (x1, y1)),
            Line2D((x1, y1), (x0, y1)),
            Line2D((x0, y1), (x0, y0)),
        ]
    )


def plate(holes=()):
    inners = [Loop2D([Arc2D.full_circle(c, r)]) for c, r in holes]
    return Face2D.create(rectangle(0.0, 0.0, 100.0, 60.0), inners)


# ==========================================================================
# Uncertainty of a recovered radius
# ==========================================================================


def test_a_shallow_arcs_radius_is_uncertain_however_well_the_points_fit():
    """The finding that matters, and the one a residual alone cannot express.

    Points can sit on a nearly-straight arc to a thousandth of a pixel and
    the radius still be meaningless, because the radius depends on the
    sagitta like 1/s² and the sagitta is what has gone to zero. A tool that
    reports only the residual will certify r = 3000 on a straight edge.
    """
    deep = certs.radius_uncertainty(sagitta=5.0, span=40.0, residual=0.01, points=200)
    shallow = certs.radius_uncertainty(sagitta=0.02, span=40.0, residual=0.01, points=200)

    assert deep < 0.2
    assert shallow > 100.0
    assert shallow / deep > 1000.0


def test_radius_uncertainty_diverges_as_the_arc_flattens():
    previous = 0.0
    for sagitta in (1.0, 0.5, 0.25, 0.125):
        value = certs.radius_uncertainty(sagitta, span=40.0, residual=0.01, points=100)
        assert value > previous
        previous = value
    assert not math.isfinite(certs.radius_uncertainty(0.0, 40.0, 0.01, 100))


def test_radius_uncertainty_falls_as_one_over_root_n():
    """More points average the noise down; they do nothing about the bias."""
    few = certs.radius_uncertainty(2.0, 40.0, 0.01, points=100)
    many = certs.radius_uncertainty(2.0, 40.0, 0.01, points=400)
    assert many == pytest.approx(few / 2.0, rel=1e-12)


def test_position_uncertainty_is_the_standard_error():
    assert certs.position_uncertainty(0.5, 100) == pytest.approx(0.05)
    assert not math.isfinite(certs.position_uncertainty(0.5, 0))


# ==========================================================================
# A measurement knows what it is worth
# ==========================================================================


def test_an_exact_measurement_supports_every_digit():
    measurement = certs.exact("radius", 6.35, "carried from the source entity")
    assert measurement.exact
    assert measurement.uncertainty == 0.0
    assert measurement.confidence(tolerance=1e-9) == 1.0
    assert measurement.digits() >= 9


def test_digits_tracks_the_uncertainty():
    assert certs.Measurement("r", 6.35, 0.01, ESTIMATED, "fit").digits() == 2
    assert certs.Measurement("r", 6.35, 0.0001, ESTIMATED, "fit").digits() == 4
    assert certs.Measurement("r", 6.35, 5.0, ESTIMATED, "fit").digits() == 0


def test_a_value_is_printed_only_to_the_precision_it_has():
    """Five decimals on a number good to two is a lie told in formatting."""
    measurement = certs.Measurement("radius", 6.3498712, 0.01, ESTIMATED, "fit")
    assert measurement.format() == "6.35"


def test_confidence_is_relative_to_a_stated_tolerance():
    """No standalone score: the same fit is fine for plasma and hopeless
    for a reamed bore, and one number cannot mean both."""
    measurement = certs.Measurement("radius", 6.35, 0.05, ESTIMATED, "fit")
    assert measurement.confidence(tolerance=1.0) == pytest.approx(0.95)
    assert measurement.confidence(tolerance=0.1) == pytest.approx(0.5)
    assert measurement.confidence(tolerance=0.05) == 0.0
    assert measurement.confidence(tolerance=0.01) == 0.0


def test_an_infinite_uncertainty_is_zero_confidence():
    measurement = certs.Measurement("radius", 3000.0, math.inf, ESTIMATED, "fit")
    assert measurement.confidence(1.0) == 0.0
    assert measurement.digits() == 0


# ==========================================================================
# Vector input: nothing was measured, so nothing is estimated
# ==========================================================================


def test_a_vector_part_is_certified_exact():
    certificate = certs.certify_face(plate([((50.0, 30.0), 6.35)]), tolerance=1e-3, depth=8.0)
    assert certificate.provenance == "vector"
    assert certificate.exact
    assert certificate.of("area").basis == EXACT
    assert certificate.of("volume").basis == EXACT
    assert certificate.confidence() > 0.999


def test_an_arcs_radius_is_carried_not_measured():
    certificate = certs.certify_face(plate([((50.0, 30.0), 6.35)]))
    arcs = [f for f in certificate.features if f.kind == "Arc2D"]
    assert arcs
    radius = arcs[0].of("radius")
    assert radius.basis == EXACT
    assert radius.value == 6.35  # bit for bit, not approx


def test_tessellation_deviation_is_reported_but_not_charged_to_the_part():
    """A mesh asked for at 1e-3 deviates by just under 1e-3 by construction.

    Rolling that into the part's confidence would make every certificate
    read as 1% confident, which would be a bug in the metric rather than a
    finding about the drawing.
    """
    certificate = certs.certify_face(plate([((50.0, 30.0), 6.35)]), tolerance=1e-3)
    deviation = certificate.of("mesh deviation")
    assert deviation.budgeted
    assert deviation.uncertainty == pytest.approx(1e-3, rel=0.05)
    assert certificate.worst_uncertainty < 1e-10
    assert certificate.confidence() > 0.999


def test_an_area_uncertainty_is_not_measured_against_a_length_tolerance():
    """Square millimetres against millimetres is a units error, not a score."""
    certificate = certs.certify_face(plate(), tolerance=1e-3)
    area = certificate.of("area")
    assert area.dimension == "area"
    assert certificate.of("width").dimension == "length"


def test_a_free_form_edge_makes_the_area_an_estimate():
    """No elementary antiderivative, so the area is quadrature and says so."""
    outline = Loop2D(
        [
            Line2D((0.0, 0.0), (100.0, 0.0)),
            Line2D((100.0, 0.0), (100.0, 40.0)),
            BezierCurve2D([(100.0, 40.0), (70.0, 80.0), (30.0, 0.0), (0.0, 40.0)]),
            Line2D((0.0, 40.0), (0.0, 0.0)),
        ]
    )
    certificate = certs.certify_face(Face2D.create(outline))
    area = certificate.of("area")
    assert area.basis == ESTIMATED
    assert area.uncertainty > 0.0
    assert not certificate.exact


# ==========================================================================
# Raster input: everything is a measurement
# ==========================================================================


def raster_reports(n=200, rms=0.05, radius=40.0, sagitta=None):
    sagitta = radius if sagitta is None else sagitta
    return [
        FitReport("arc", n, rms, rms * 3, sagitta=sagitta, radius=radius),
        FitReport("line", n, rms, rms * 3),
    ]


def test_a_raster_part_is_never_certified_exact():
    certificate = certs.certify_face(plate(), fit_reports=raster_reports())
    assert certificate.provenance == "raster"
    assert not certificate.exact
    assert certificate.of("area").basis == ESTIMATED


def test_the_systematic_bias_is_reported_separately_from_the_noise():
    """Averaging more points kills the noise and leaves the bias untouched,
    so folding them together would understate the error."""
    certificate = certs.certify_face(plate(), fit_reports=raster_reports(n=10000, rms=0.01))
    bias = certificate.of("systematic contour bias")
    assert bias is not None
    assert bias.basis == BOUNDED
    assert bias.value == pytest.approx(certs.RASTER_SYSTEMATIC_BIAS)
    # With 10,000 points the standard error is negligible and the bias is
    # the whole story -- which is exactly when reporting only the residual
    # would be most misleading.
    assert certificate.worst_uncertainty > certs.RASTER_SYSTEMATIC_BIAS


def test_the_bias_scales_with_the_pixel_size():
    coarse = certs.certify_face(plate(), fit_reports=raster_reports(), pixel_scale=0.5)
    assert coarse.of("systematic contour bias").value == pytest.approx(
        certs.RASTER_SYSTEMATIC_BIAS * 0.5
    )


def test_an_unresolvable_arc_is_flagged_and_says_why():
    """Sagitta below the noise floor: the curvature is not measurable."""
    reports = [FitReport("arc", 100, 0.5, 1.2, sagitta=0.2, radius=3000.0)]
    certificate = certs.certify_face(plate(), fit_reports=reports)
    assert certificate.unresolvable
    note = certificate.unresolvable[0].note
    assert "not measurable" in note
    assert "should not be quoted" in note


def test_a_well_resolved_arc_is_not_flagged():
    certificate = certs.certify_face(plate(), fit_reports=raster_reports(rms=0.02))
    assert not certificate.unresolvable


def test_an_extent_carries_twice_the_boundary_offset():
    """Both opposite edges are displaced, so the width is short by two."""
    certificate = certs.certify_face(plate(), fit_reports=raster_reports(rms=0.0))
    width = certificate.of("width")
    assert width.uncertainty == pytest.approx(2.0 * certs.RASTER_SYSTEMATIC_BIAS)


# ==========================================================================
# Fingerprints
# ==========================================================================


def test_the_same_geometry_fingerprints_the_same():
    a = plate([((50.0, 30.0), 6.35)])
    b = plate([((50.0, 30.0), 6.35)])
    assert certs.face_fingerprint(a) == certs.face_fingerprint(b)


def test_a_different_radius_fingerprints_differently():
    a = plate([((50.0, 30.0), 6.35)])
    b = plate([((50.0, 30.0), 6.36)])
    assert certs.face_fingerprint(a) != certs.face_fingerprint(b)


def test_a_change_of_one_ulp_changes_the_fingerprint():
    """Not a hash of a rounded summary -- a hash of the actual coordinates."""
    a = plate([((50.0, 30.0), 6.35)])
    b = plate([((50.0, 30.0), math.nextafter(6.35, math.inf))])
    assert certs.face_fingerprint(a) != certs.face_fingerprint(b)


def test_the_fingerprint_ignores_where_a_loop_starts_and_which_way_it_runs():
    """Export artefacts, not design changes."""
    outline = rectangle(0.0, 0.0, 100.0, 60.0)
    rotated = Loop2D(
        [
            Line2D((100.0, 60.0), (0.0, 60.0)),
            Line2D((0.0, 60.0), (0.0, 0.0)),
            Line2D((0.0, 0.0), (100.0, 0.0)),
            Line2D((100.0, 0.0), (100.0, 60.0)),
        ]
    )
    assert certs.loop_fingerprint(outline) == certs.loop_fingerprint(rotated)
    assert certs.loop_fingerprint(outline) == certs.loop_fingerprint(outline.reverse())


def test_a_drawing_fingerprint_does_not_depend_on_the_order_of_its_faces():
    a = Face2D.create(rectangle(0.0, 0.0, 10.0, 10.0))
    b = Face2D.create(rectangle(50.0, 0.0, 60.0, 10.0))
    assert certs.drawing_fingerprint([a, b]) == certs.drawing_fingerprint([b, a])


def test_holes_may_arrive_in_any_order():
    a = plate([((30.0, 30.0), 3.0), ((70.0, 30.0), 4.0)])
    b = plate([((70.0, 30.0), 4.0), ((30.0, 30.0), 3.0)])
    assert certs.face_fingerprint(a) == certs.face_fingerprint(b)


# ==========================================================================
# Determinism of the output
# ==========================================================================


def test_the_same_solid_writes_byte_identical_step(tmp_path):
    """With the clock pinned, nothing in the file is a function of anything
    but the geometry -- so two runs can be compared by hash."""
    solid = extrude_face(plate([((50.0, 30.0), 6.35)]), 8.0, name="plate")
    first = tmp_path / "a.step"
    second = tmp_path / "b.step"
    write_step(solid, first, name="plate", timestamp="2020-01-01T00:00:00")
    write_step(solid, second, name="plate", timestamp="2020-01-01T00:00:00")
    assert first.read_bytes() == second.read_bytes()


def test_source_date_epoch_pins_the_timestamp(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1577836800")  # 2020-01-01T00:00:00Z
    solid = extrude_face(plate(), 8.0, name="plate")
    path = tmp_path / "pinned.step"
    write_step(solid, path, name="plate")
    assert "2020-01-01T00:00:00" in path.read_text()


def test_a_nonsense_source_date_epoch_does_not_break_the_write(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "not-a-number")
    solid = extrude_face(plate(), 8.0, name="plate")
    path = tmp_path / "fallback.step"
    write_step(solid, path, name="plate")
    assert path.read_text().startswith("ISO-10303-21;")


def test_only_the_timestamp_varies_between_unpinned_writes(tmp_path):
    solid = extrude_face(plate([((50.0, 30.0), 6.35)]), 8.0, name="plate")
    a = tmp_path / "a.step"
    b = tmp_path / "b.step"
    write_step(solid, a, name="plate", timestamp="2020-01-01T00:00:00")
    write_step(solid, b, name="plate", timestamp="2031-06-15T12:34:56")
    differing = [
        (x, y)
        for x, y in zip(a.read_text().splitlines(), b.read_text().splitlines())
        if x != y
    ]
    assert len(differing) == 1
    assert differing[0][0].startswith("FILE_NAME")


def test_reading_the_same_drawing_twice_gives_the_same_fingerprint(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 60), (0, 60)], close=True)
    msp.add_circle((50, 30), 6.35)
    path = tmp_path / "plate.dxf"
    doc.saveas(path)

    from blueprint23d.loaders import load_faces

    first, _ = load_faces(path)
    second, _ = load_faces(path)
    assert certs.drawing_fingerprint(first) == certs.drawing_fingerprint(second)


def test_the_pipeline_is_deterministic_across_processes(tmp_path):
    """A fresh interpreter, so nothing carries over -- no warm caches, no
    hash seed, no dictionary order from the last run."""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 60), (0, 60)], close=True)
    for x in (20, 40, 60, 80):
        msp.add_circle((x, 30), 4.0)
    path = tmp_path / "many.dxf"
    doc.saveas(path)

    script = (
        "from blueprint23d.loaders import load_faces\n"
        "from blueprint23d.certificates import drawing_fingerprint\n"
        f"faces, _ = load_faces({str(path)!r})\n"
        "print(drawing_fingerprint(faces))\n"
    )
    seeds = ["0", "12345"]
    outputs = []
    for seed in seeds:
        environment = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        )
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1]
    assert len(outputs[0]) == 64


# ==========================================================================
# The whole document
# ==========================================================================


def test_a_certificate_serialises_to_json(tmp_path):
    import json

    certificate = certs.certify_face(plate([((50.0, 30.0), 6.35)]), depth=8.0)
    payload = json.loads(certificate.to_json())
    assert payload["exact"] is True
    assert payload["fingerprint"]
    names = [m["name"] for m in payload["measurements"]]
    assert "area" in names and "volume" in names
    assert all("basis" in m and "method" in m for m in payload["measurements"])


def test_certify_drawing_reads_a_file_end_to_end(tmp_path):
    doc = ezdxf.new(setup=True)
    doc.modelspace().add_lwpolyline([(0, 0), (100, 0), (100, 60), (0, 60)], close=True)
    path = tmp_path / "part.dxf"
    doc.saveas(path)

    certificate = certs.certify_drawing(path, depth=8.0)
    assert certificate.exact
    assert certificate.of("volume").value == pytest.approx(100 * 60 * 8)
    assert certificate.source == str(path)


def test_certify_drawing_refuses_a_file_with_no_profile(tmp_path):
    doc = ezdxf.new(setup=True)
    doc.modelspace().add_line((0, 0), (10, 0))
    path = tmp_path / "bare.dxf"
    doc.saveas(path)
    with pytest.raises(ValueError, match="no closed profile"):
        certs.certify_drawing(path)


def test_the_description_names_the_weakest_number():
    certificate = certs.certify_face(
        plate(), tolerance=0.1, fit_reports=raster_reports(rms=0.2)
    )
    text = certificate.describe()
    assert "raster" in text
    assert "confidence" in text
    assert "fingerprint" in text
