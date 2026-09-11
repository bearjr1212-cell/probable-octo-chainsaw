"""Bends, and the reason a folded part comes out the wrong size.

A bracket drawn 50 + 30 + 50 is cut from a blank about 7 mm shorter,
because bending consumes material. Nobody quotes a 7 mm tolerance. These
tests pin the arithmetic that gets there and the four rules that stop a
press brake.
"""

import math

import ezdxf
import pytest

from blueprint23d import sheetmetal as sm
from blueprint23d.brep import Face2D, Loop2D
from blueprint23d.curves import Arc2D, Line2D
from blueprint23d.diagnostics import Code, Severity

RIGHT = math.pi / 2


def rectangle(x0, y0, x1, y1):
    return Loop2D(
        [
            Line2D((x0, y0), (x1, y0)),
            Line2D((x1, y0), (x1, y1)),
            Line2D((x1, y1), (x0, y1)),
            Line2D((x0, y1), (x0, y0)),
        ]
    )


def blank(width=200.0, height=100.0, holes=()):
    inners = [Loop2D([Arc2D.full_circle(c, r)]) for c, r in holes]
    return Face2D.create(rectangle(0.0, 0.0, width, height), inners)


# ==========================================================================
# The K-factor
# ==========================================================================


def test_k_rises_with_the_radius_because_a_gentle_bend_strains_less():
    values = [sm.k_factor(r, 2.0) for r in (2.0, 4.0, 6.0, 10.0)]
    assert values == sorted(values)
    assert values[0] < values[-1]


def test_k_is_bounded_by_the_physics_at_both_ends():
    """0.325 is the tightest practical bend; 0.5 is no strain at all, which
    is the geometric middle of the section and cannot be passed."""
    assert sm.k_factor(0.1, 10.0) == pytest.approx(0.325)
    assert sm.k_factor(1000.0, 1.0) == pytest.approx(0.5)
    for radius in (0.05, 0.5, 5.0, 50.0, 500.0):
        assert 0.325 <= sm.k_factor(radius, 2.0) <= 0.5


def test_k_depends_on_the_ratio_and_not_the_absolute_size():
    """A bend is the same bend scaled up: R/T is what matters."""
    assert sm.k_factor(4.0, 2.0) == pytest.approx(sm.k_factor(40.0, 20.0))


def test_a_shop_that_has_measured_its_tooling_can_pin_k():
    measured = sm.material("mild_steel").but(k_override=0.42)
    assert sm.k_factor(2.0, 2.0, measured) == 0.42
    assert sm.k_factor(2.0, 2.0) != 0.42  # the default is untouched


def test_a_zero_thickness_is_refused():
    with pytest.raises(ValueError):
        sm.k_factor(2.0, 0.0)


# ==========================================================================
# Allowance, setback and deduction
# ==========================================================================


def test_the_allowance_is_the_neutral_lines_own_arc_length():
    """Not a fitted number: it is θ·(R + K·T), the arc the neutral line
    traces, which is the one length the bend does not change."""
    k = 0.33
    assert sm.bend_allowance(RIGHT, 2.0, 2.0, k) == pytest.approx(RIGHT * (2.0 + k * 2.0))


def test_the_setback_of_a_right_angle_is_the_outside_radius():
    """tan(45°) is 1, so a 90° bend sets back by exactly R + T."""
    assert sm.outside_setback(RIGHT, 2.0, 3.0) == pytest.approx(5.0)


def test_a_flat_bend_deducts_nothing():
    assert sm.bend_deduction(0.0, 2.0, 2.0, 0.4) == pytest.approx(0.0)


def test_the_deduction_of_a_standard_right_angle_matches_the_tables():
    """2 mm mild steel, R2, 90°: shop tables put this a shade under 4."""
    k = sm.k_factor(2.0, 2.0)
    assert sm.bend_deduction(RIGHT, 2.0, 2.0, k) == pytest.approx(3.84, abs=0.05)


def test_a_sharper_angle_deducts_less_than_a_right_angle():
    k = sm.k_factor(2.0, 2.0)
    gentle = sm.bend_deduction(math.radians(30.0), 2.0, 2.0, k)
    right = sm.bend_deduction(RIGHT, 2.0, 2.0, k)
    assert 0.0 < gentle < right


def test_a_bend_describes_its_own_arithmetic():
    text = sm.Bend(RIGHT, 2.0, direction=-1).describe(2.0)
    assert "90° down" in text
    assert "K=" in text and "deduction" in text


# ==========================================================================
# The flat pattern
# ==========================================================================


def test_the_blank_is_shorter_than_the_finished_part_by_enough_to_matter():
    """The whole point: 130 of outside dimensions comes off a 122 blank,
    and a shop that cuts 130 has scrapped the sheet."""
    pattern = sm.flat_pattern([50.0, 30.0, 50.0], [sm.Bend(RIGHT, 2.0)] * 2, 2.0)

    assert pattern.folded_length == 130.0
    assert pattern.flat_length == pytest.approx(122.3, abs=0.1)
    assert pattern.shortfall == pytest.approx(7.67, abs=0.05)
    assert pattern.shortfall > 1.0  # far outside any tolerance anyone quotes


def test_every_bend_contributes_one_deduction():
    pattern = sm.flat_pattern(
        [20.0, 20.0, 20.0, 20.0], [sm.Bend(RIGHT, 1.0)] * 3, 1.0
    )
    assert len(pattern.deductions) == 3
    assert pattern.flat_length == pytest.approx(80.0 - sum(pattern.deductions))


def test_a_part_with_no_bends_is_its_own_blank():
    pattern = sm.flat_pattern([120.0], [], 2.0)
    assert pattern.flat_length == 120.0
    assert pattern.shortfall == 0.0


def test_flanges_and_bends_have_to_agree_in_number():
    """A silently wrong blank length is worse than an exception."""
    with pytest.raises(ValueError, match="flanges"):
        sm.flat_pattern([50.0, 30.0], [sm.Bend(RIGHT, 2.0)] * 2, 2.0)


def test_a_tighter_radius_needs_a_longer_blank():
    tight = sm.flat_pattern([50.0, 50.0], [sm.Bend(RIGHT, 1.0)], 2.0)
    generous = sm.flat_pattern([50.0, 50.0], [sm.Bend(RIGHT, 6.0)], 2.0)
    assert tight.flat_length > generous.flat_length


def test_the_pattern_shows_its_working():
    pattern = sm.flat_pattern([50.0, 30.0, 50.0], [sm.Bend(RIGHT, 2.0)] * 2, 2.0)
    text = pattern.describe()
    assert "50 + 30 + 50" in text
    assert "K=" in text
    assert "blank" in text

    payload = pattern.to_dict()
    assert payload["material"] == "mild steel"
    assert len(payload["k_factors"]) == 2


# ==========================================================================
# Bend radius against the material
# ==========================================================================


def test_a_radius_fine_in_5052_cracks_in_6061():
    """The same drawing, two aluminium alloys, two answers."""
    part = blank()
    bends = [sm.Bend(RIGHT, 2.0, line=((50.0, 0.0), (50.0, 100.0)))]

    formable = sm.check_sheet_metal(part, bends, 2.0, sm.material("aluminium_5052"))
    brittle = sm.check_sheet_metal(part, bends, 2.0, sm.material("aluminium_6061_t6"))

    assert not formable.of(Code.BEND_RADIUS_TOO_SMALL)
    defect = brittle.of(Code.BEND_RADIUS_TOO_SMALL)[0]
    assert defect.measurements["minimum"] == pytest.approx(6.0)
    assert "crack" in defect.message


def test_a_radius_far_below_the_minimum_is_an_error():
    part = blank()
    bends = [sm.Bend(RIGHT, 0.2, line=((50.0, 0.0), (50.0, 100.0)))]
    report = sm.check_sheet_metal(part, bends, 2.0)
    assert report.of(Code.BEND_RADIUS_TOO_SMALL)[0].severity is Severity.ERROR
    assert not report.cuttable


def test_a_generous_radius_raises_nothing():
    part = blank()
    bends = [sm.Bend(RIGHT, 3.0, line=((50.0, 0.0), (50.0, 100.0)))]
    assert not sm.check_sheet_metal(part, bends, 2.0).of(Code.BEND_RADIUS_TOO_SMALL)


# ==========================================================================
# Flange length
# ==========================================================================


def test_a_flange_too_short_to_hold_is_caught_and_measured():
    """Under about four thicknesses the part drops into the die."""
    part = blank(width=200.0)
    bends = [sm.Bend(RIGHT, 2.0, line=((5.0, 0.0), (5.0, 100.0)))]
    report = sm.check_sheet_metal(part, bends, 3.0)

    defect = report.of(Code.FLANGE_TOO_SHORT)[0]
    assert defect.measurements["width"] == pytest.approx(5.0, abs=0.2)
    assert defect.measurements["minimum"] == pytest.approx(12.0)


def test_a_bend_in_the_middle_of_a_plate_leaves_plenty():
    part = blank(width=200.0)
    bends = [sm.Bend(RIGHT, 2.0, line=((100.0, 0.0), (100.0, 100.0)))]
    assert not sm.check_sheet_metal(part, bends, 3.0).of(Code.FLANGE_TOO_SHORT)


def test_the_minimum_flange_is_an_argument_not_a_constant():
    part = blank(width=200.0)
    bends = [sm.Bend(RIGHT, 2.0, line=((5.0, 0.0), (5.0, 100.0)))]
    assert not sm.check_sheet_metal(
        part, bends, 3.0, min_flange_ratio=1.0
    ).of(Code.FLANGE_TOO_SHORT)


def test_flange_widths_ignore_boundary_off_the_end_of_the_bend():
    """Only the stretch of edge the bend actually faces is a flange."""
    part = blank(width=200.0, height=100.0)
    bend = sm.Bend(RIGHT, 2.0, line=((60.0, 0.0), (60.0, 100.0)))
    left, right = sm.flange_widths(part, bend)
    assert left == pytest.approx(60.0, abs=0.5) or right == pytest.approx(60.0, abs=0.5)
    assert {round(left), round(right)} == {60, 140}


# ==========================================================================
# Bend relief
# ==========================================================================


def test_a_bend_running_out_at_a_straight_edge_wants_a_relief():
    part = blank()
    bends = [sm.Bend(RIGHT, 2.0, line=((100.0, 0.0), (100.0, 100.0)))]
    report = sm.check_sheet_metal(part, bends, 2.0)
    reliefs = report.of(Code.BEND_RELIEF_MISSING)
    assert len(reliefs) == 2  # both ends run out at an edge
    assert "tear" in reliefs[0].message


def test_a_notched_edge_counts_as_a_relief():
    """A relief is a notch, and a notch is a departure from a straight edge."""
    notched = Loop2D(
        [
            Line2D((0.0, 0.0), (97.0, 0.0)),
            Line2D((97.0, 0.0), (97.0, 5.0)),
            Line2D((97.0, 5.0), (103.0, 5.0)),
            Line2D((103.0, 5.0), (103.0, 0.0)),
            Line2D((103.0, 0.0), (200.0, 0.0)),
            Line2D((200.0, 0.0), (200.0, 100.0)),
            Line2D((200.0, 100.0), (0.0, 100.0)),
            Line2D((0.0, 100.0), (0.0, 0.0)),
        ]
    )
    part = Face2D.create(notched)
    assert sm.has_relief(part, (100.0, 5.0), window=8.0, thickness=2.0)
    assert not sm.has_relief(part, (100.0, 100.0), window=8.0, thickness=2.0)


def test_a_bend_that_stops_short_of_the_edge_needs_no_relief():
    part = blank()
    bends = [sm.Bend(RIGHT, 2.0, line=((100.0, 30.0), (100.0, 70.0)))]
    assert not sm.check_sheet_metal(part, bends, 2.0).of(Code.BEND_RELIEF_MISSING)


# ==========================================================================
# Holes near a bend
# ==========================================================================


def test_a_hole_too_close_to_a_bend_is_reported():
    """Inside 2.5×t + R the hole draws into an oval."""
    part = blank(holes=[((104.0, 50.0), 3.0)])
    bends = [sm.Bend(RIGHT, 2.0, line=((100.0, 0.0), (100.0, 100.0)))]
    report = sm.check_sheet_metal(part, bends, 2.0)

    defect = report.of(Code.HOLE_NEAR_BEND)[0]
    assert defect.measurements["distance"] == pytest.approx(1.0, abs=0.05)
    assert defect.measurements["minimum"] == pytest.approx(7.0)
    assert defect.location.point == pytest.approx((104.0, 50.0), abs=1e-6)


def test_a_hole_well_clear_of_the_bend_is_fine():
    part = blank(holes=[((150.0, 50.0), 3.0)])
    bends = [sm.Bend(RIGHT, 2.0, line=((100.0, 0.0), (100.0, 100.0)))]
    assert not sm.check_sheet_metal(part, bends, 2.0).of(Code.HOLE_NEAR_BEND)


def test_the_clearance_grows_with_thickness_and_radius():
    part = blank(holes=[((112.0, 50.0), 3.0)])
    bends_thin = [sm.Bend(RIGHT, 1.0, line=((100.0, 0.0), (100.0, 100.0)))]
    bends_thick = [sm.Bend(RIGHT, 6.0, line=((100.0, 0.0), (100.0, 100.0)))]
    assert not sm.check_sheet_metal(part, bends_thin, 1.0).of(Code.HOLE_NEAR_BEND)
    assert sm.check_sheet_metal(part, bends_thick, 4.0).of(Code.HOLE_NEAR_BEND)


# ==========================================================================
# Reading bend lines from a drawing
# ==========================================================================


def test_a_layer_name_can_carry_the_angle_radius_and_direction():
    angle, radius, direction = sm.parse_bend_layer("BEND_DOWN_45_R3", RIGHT, 1.0)
    assert angle == pytest.approx(math.radians(45.0))
    assert radius == 3.0
    assert direction == -1


def test_a_bare_bend_layer_falls_back_on_the_defaults():
    angle, radius, direction = sm.parse_bend_layer("BEND", RIGHT, 1.5)
    assert angle == pytest.approx(RIGHT)
    assert radius == 1.5
    assert direction == 1


def test_bend_lines_are_read_from_the_drawing(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (200, 0), (200, 100), (0, 100)], close=True)
    msp.add_line((60, 0), (60, 100), dxfattribs={"layer": "BEND_UP_90_R2"})
    msp.add_line((140, 0), (140, 100), dxfattribs={"layer": "BEND_DOWN_90_R2"})
    path = tmp_path / "bracket.dxf"
    doc.saveas(path)

    bends, report = sm.load_bends(path)
    assert len(bends) == 2
    assert [b.direction for b in bends] == [1, -1]
    assert all(b.radius == 2.0 for b in bends)
    assert report.metrics["bends"] == 2


def test_geometry_on_other_layers_is_not_mistaken_for_a_bend(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (200, 0), (200, 100), (0, 100)], close=True)
    msp.add_line((10, 10), (190, 10), dxfattribs={"layer": "CENTRELINE"})
    path = tmp_path / "noline.dxf"
    doc.saveas(path)

    bends, report = sm.load_bends(path)
    assert bends == []
    assert report.of(Code.BEND_LINE_INVALID)[0].severity is Severity.INFO


def test_the_whole_check_runs_from_a_file(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (200, 0), (200, 100), (0, 100)], close=True)
    msp.add_circle((104, 50), 3.0)
    msp.add_line((100, 0), (100, 100), dxfattribs={"layer": "BEND_UP_90_R2"})
    path = tmp_path / "full.dxf"
    doc.saveas(path)

    bends, report = sm.check_drawing(path, thickness=2.0)
    assert len(bends) == 1
    assert report.of(Code.HOLE_NEAR_BEND)
    assert report.of(Code.BEND_RELIEF_MISSING)
    assert report.metrics["material"] == "mild steel"


def test_an_unreadable_file_reports_rather_than_raising(tmp_path):
    bends, report = sm.check_drawing(tmp_path / "nope.dxf", thickness=2.0)
    assert bends == []
    assert report.at_least(Severity.CRITICAL)


def test_an_unknown_material_says_what_is_available():
    with pytest.raises(ValueError, match="mild_steel"):
        sm.material("unobtanium")
