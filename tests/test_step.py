"""STEP AP214 export: structural validity and preservation of nominal geometry."""

import math
import re

import pytest

from blueprint23d.brep import Face2D, Loop2D, extrude_face
from blueprint23d.curves import Arc2D, BezierCurve2D, EllipseArc2D, Line2D
from blueprint23d.step_writer import (
    StepWriter,
    parse_entities,
    summarize_step,
    validate_step,
    write_step,
)


def rectangle(w=50.0, h=30.0):
    return Loop2D(
        [
            Line2D((0.0, 0.0), (w, 0.0)),
            Line2D((w, 0.0), (w, h)),
            Line2D((w, h), (0.0, h)),
            Line2D((0.0, h), (0.0, 0.0)),
        ]
    )


def bracket(radius=6.0):
    return Face2D.create(rectangle(), [Loop2D([Arc2D.full_circle((25.0, 15.0), radius)])])


def rounded_plate():
    r, w, h = 3.0, 40.0, 20.0
    loop = Loop2D(
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
    holes = [Loop2D([Arc2D.full_circle((10.0, 10.0), 2.5)]), Loop2D([Arc2D.full_circle((30.0, 10.0), 2.5)])]
    return Face2D.create(loop, holes)


def render(face, depth=8.0, name="part"):
    return StepWriter(name=name).render(extrude_face(face, depth, name=name))


# --------------------------------------------------------------------------


def test_emitted_file_is_structurally_valid():
    assert validate_step(render(bracket())) == []


def test_rounded_plate_with_holes_is_valid():
    assert validate_step(render(rounded_plate())) == []


def test_spline_part_is_valid():
    loop = Loop2D(
        [
            BezierCurve2D(((0.0, 0.0), (10.0, 20.0), (30.0, 20.0), (40.0, 0.0))),
            Line2D((40.0, 0.0), (0.0, 0.0)),
        ]
    )
    text = render(Face2D.create(loop), name="spline_part")
    assert validate_step(text) == []
    counts = summarize_step(text)
    assert counts.get("B_SPLINE_CURVE_WITH_KNOTS", 0) >= 1
    assert counts.get("SURFACE_OF_LINEAR_EXTRUSION", 0) == 1


def test_every_reference_resolves():
    """Dangling references are the classic way a hand-written STEP file fails."""
    text = render(rounded_plate())
    entities = parse_entities(text)
    for entity_id, body in entities.items():
        for ref in re.findall(r"#(\d+)", body):
            assert int(ref) in entities, f"#{entity_id} points at missing #{ref}"


def test_hole_is_exported_as_a_cylinder_not_facets():
    text = render(bracket(radius=6.0))
    counts = summarize_step(text)
    assert counts["CYLINDRICAL_SURFACE"] == 1
    assert counts.get("B_SPLINE_SURFACE_WITH_KNOTS", 0) == 0  # not approximated
    # And the bounding edges are circles, not polylines.
    assert counts["CIRCLE"] == 2


def test_nominal_diameter_survives_into_the_file_text():
    """The number a machinist reads must be the number that was drawn.

    6.35 mm is a quarter inch; it appears in the file exactly, not as
    6.3499999 or as a chord length implied by triangles.
    """
    text = render(bracket(radius=6.35))
    radii = set(re.findall(r"CYLINDRICAL_SURFACE\('',#\d+,([0-9.E+-]+)\)", text))
    assert radii == {"6.35"}
    circle_radii = set(re.findall(r"CIRCLE\('',#\d+,([0-9.E+-]+)\)", text))
    assert circle_radii == {"6.35"}


def test_fillets_and_holes_all_become_cylinders():
    counts = summarize_step(render(rounded_plate()))
    assert counts["CYLINDRICAL_SURFACE"] == 6  # 4 fillets + 2 holes
    assert counts["CIRCLE"] == 12  # top and bottom edge of each


def test_ellipse_edge_exports_as_ellipse():
    loop = Loop2D([EllipseArc2D((0.0, 0.0), (10.0, 0.0), 0.5)])
    text = render(Face2D.create(loop), name="elliptical")
    assert validate_step(text) == []
    assert summarize_step(text).get("ELLIPSE", 0) >= 1


def test_required_ap214_scaffolding_is_present():
    counts = summarize_step(render(bracket()))
    for required in (
        "MANIFOLD_SOLID_BREP",
        "CLOSED_SHELL",
        "ADVANCED_BREP_SHAPE_REPRESENTATION",
        "SHAPE_DEFINITION_REPRESENTATION",
        "PRODUCT_DEFINITION_SHAPE",
        "APPLICATION_CONTEXT",
    ):
        assert counts.get(required, 0) >= 1, f"missing {required}"


def test_face_count_matches_the_brep():
    solid = extrude_face(bracket(), 8.0)
    counts = summarize_step(StepWriter().render(solid))
    assert counts["ADVANCED_FACE"] == len(solid.faces())
    assert counts["CLOSED_SHELL"] == 1


def test_real_literals_are_valid_part21():
    """Part 21 reals must carry a point or an exponent; bare integers are invalid."""
    text = render(bracket())
    for body in parse_entities(text).values():
        if body.startswith("CARTESIAN_POINT"):
            # CARTESIAN_POINT('',(x,y,z)) -- the coordinate list is the
            # final parenthesised group, after the name field.
            coords = re.search(r",\(([^)]*)\)\)$", body)
            assert coords, body
            for value in coords.group(1).split(","):
                value = value.strip()
                assert "." in value or "E" in value, f"invalid STEP real {value!r}"


def test_header_and_terminator():
    text = render(bracket())
    assert text.startswith("ISO-10303-21;")
    assert text.rstrip().endswith("END-ISO-10303-21;")
    assert "AUTOMOTIVE_DESIGN" in text


def test_validator_catches_a_dangling_reference():
    """The validator must actually fail on a broken file, or it proves nothing."""
    text = render(bracket())
    broken = text.replace("CLOSED_SHELL('',(#", "CLOSED_SHELL('',(#99999,#", 1)
    assert validate_step(broken) != []


def test_validator_catches_missing_scaffolding():
    text = render(bracket())
    stripped = "\n".join(l for l in text.splitlines() if "APPLICATION_CONTEXT" not in l)
    assert validate_step(stripped) != []


def test_write_step_creates_a_file(tmp_path):
    solid = extrude_face(bracket(), 8.0, name="bracket")
    path = write_step(solid, tmp_path / "out" / "bracket.step")
    assert path.exists()
    assert validate_step(path.read_text()) == []


def test_rejects_unsupported_units():
    with pytest.raises(ValueError):
        StepWriter(units="FURLONG")
