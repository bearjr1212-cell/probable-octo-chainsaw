"""Manufacturability against a real machine, and what the cut costs.

A drawing is not manufacturable in the abstract. These tests build parts
that are fine on one process and impossible on the next, and check that
the difference is reported rather than averaged away -- because the
cheapest machine in a quote is frequently one that cannot make the part.
"""

import math

import ezdxf
import pytest

from blueprint23d import processes as proc
from blueprint23d.brep import Face2D, Loop2D
from blueprint23d.curves import Arc2D, Line2D
from blueprint23d.diagnostics import Code, Severity


def rectangle(x0, y0, x1, y1):
    return Loop2D(
        [
            Line2D((x0, y0), (x1, y0)),
            Line2D((x1, y0), (x1, y1)),
            Line2D((x1, y1), (x0, y1)),
            Line2D((x0, y1), (x0, y0)),
        ]
    )


def plate(holes=(), cutouts=(), width=200.0, height=120.0):
    inners = [Loop2D([Arc2D.full_circle(c, r)]) for c, r in holes]
    inners += list(cutouts)
    return Face2D.create(rectangle(0.0, 0.0, width, height), inners)


LASER = proc.process("fiber_laser")
PLASMA = proc.process("plasma")
WATERJET = proc.process("waterjet")
ROUTER = proc.process("router")


# ==========================================================================
# Measuring the geometry the rules depend on
# ==========================================================================


def test_a_circles_narrowest_width_is_its_diameter():
    """The easy case, to within the conservative slack: a circle's width
    is the diameter, reported a hair under rather than a hair over."""
    width, _ = proc.narrowest_width(Loop2D([Arc2D.full_circle((0.0, 0.0), 10.0)]))
    assert 19.9 <= width <= 20.0


def test_a_thin_slot_is_measured_across_not_along():
    width, where = proc.narrowest_width(rectangle(10.0, 10.0, 90.0, 10.15))
    assert width == pytest.approx(0.15, abs=1e-9)
    assert where[1] == pytest.approx(10.075, abs=1e-6)


def test_the_narrow_leg_of_an_l_shaped_cutout_is_found():
    """The bounding box says 50 x 50 and tells you nothing about the 2 mm leg."""
    l_cutout = Loop2D(
        [
            Line2D((0.0, 0.0), (50.0, 0.0)),
            Line2D((50.0, 0.0), (50.0, 2.0)),
            Line2D((50.0, 2.0), (2.0, 2.0)),
            Line2D((2.0, 2.0), (2.0, 50.0)),
            Line2D((2.0, 50.0), (0.0, 50.0)),
            Line2D((0.0, 50.0), (0.0, 0.0)),
        ]
    )
    width, _ = proc.narrowest_width(l_cutout)
    assert width == pytest.approx(2.0, abs=1e-9)


def test_a_tessellation_chord_is_not_mistaken_for_a_width():
    """Sample a circle finely enough and every step is a short distance
    between two points on the boundary. None of them is a width.

    The answer is a lower bound on the true 50 and stays there as the
    sampling refines -- erring toward flagging, never toward missing, and
    never collapsing onto a tessellation step.
    """
    circle = Loop2D([Arc2D.full_circle((0.0, 0.0), 25.0)])
    widths = [proc.narrowest_width(circle, t)[0] for t in (0.5, 0.05, 0.005)]
    for width in widths:
        assert 49.5 <= width <= 50.0
    assert max(widths) - min(widths) < 0.2


def test_clearance_to_a_long_edge_is_measured_to_the_edge():
    """A straight edge tessellates to two endpoints fifty apart. A hole
    half a millimetre off the middle of it is half a millimetre from the
    part, not fifty."""
    edge = Loop2D([Line2D((0.0, 0.0), (100.0, 0.0)), Line2D((100.0, 0.0), (0.0, 0.0))])
    hole = Loop2D([Arc2D.full_circle((50.0, 5.5), 5.0)])
    gap, where = proc.loop_clearance(edge, hole)
    assert 0.49 <= gap <= 0.5  # a lower bound on the true 0.5
    assert where[0] == pytest.approx(50.0, abs=0.5)


def test_corner_angles_find_the_reflex_corner_of_an_l_shape():
    l_shape = Loop2D(
        [
            Line2D((0.0, 0.0), (100.0, 0.0)),
            Line2D((100.0, 0.0), (100.0, 40.0)),
            Line2D((100.0, 40.0), (40.0, 40.0)),
            Line2D((40.0, 40.0), (40.0, 100.0)),
            Line2D((40.0, 100.0), (0.0, 100.0)),
            Line2D((0.0, 100.0), (0.0, 0.0)),
        ]
    )
    reflex = [c for c in proc.corner_angles(l_shape) if c[1] > math.pi + 1e-9]
    assert len(reflex) == 1
    assert reflex[0][0] == pytest.approx((40.0, 40.0))
    assert reflex[0][1] == pytest.approx(1.5 * math.pi)


# ==========================================================================
# The same part, judged by different machines
# ==========================================================================


def test_a_hole_fine_on_a_laser_is_a_blocker_on_plasma():
    """The whole point of a process profile. Ø6 through 6 mm plate is a
    routine laser hole and a plasma arc will close it on pierce."""
    part = plate(holes=[((100.0, 60.0), 3.0)])

    assert proc.check_manufacturability(part, LASER, 6.0).cuttable
    plasma_report = proc.check_manufacturability(part, PLASMA, 6.0)
    assert not plasma_report.cuttable
    defect = plasma_report.of(Code.HOLE_TOO_SMALL_FOR_THICKNESS)[0]
    assert defect.measurements["diameter"] == pytest.approx(6.0)
    assert defect.measurements["minimum"] == pytest.approx(9.0)


def test_a_hole_too_small_for_every_thermal_process_is_fine_on_a_waterjet():
    part = plate(holes=[((100.0, 60.0), 2.0)])
    assert not proc.check_manufacturability(part, LASER, 8.0).cuttable
    assert proc.check_manufacturability(part, WATERJET, 8.0).cuttable


def test_a_marginal_hole_is_a_warning_and_a_hopeless_one_is_an_error():
    """Thirty per cent under the rule is a conversation; half is a no."""
    marginal = proc.check_manufacturability(plate(holes=[((100.0, 60.0), 2.6)]), LASER, 6.0)
    hopeless = proc.check_manufacturability(plate(holes=[((100.0, 60.0), 1.5)]), LASER, 6.0)
    assert marginal.of(Code.HOLE_TOO_SMALL_FOR_THICKNESS)[0].severity is Severity.WARNING
    assert hopeless.of(Code.HOLE_TOO_SMALL_FOR_THICKNESS)[0].severity is Severity.ERROR


def test_a_feature_narrower_than_the_kerf_is_an_error_not_a_warning():
    """There is no judgement call here: the cutter does not fit."""
    part = plate(cutouts=[rectangle(50.0, 50.0, 150.0, 50.15)])
    report = proc.check_manufacturability(part, LASER, 3.0)
    defect = report.of(Code.FEATURE_BELOW_KERF)[0]
    assert defect.severity is Severity.ERROR
    assert defect.measurements["width"] == pytest.approx(0.15, abs=1e-6)
    assert defect.measurements["kerf"] == pytest.approx(LASER.kerf)


def test_the_same_slot_is_producible_with_a_finer_kerf():
    part = plate(cutouts=[rectangle(50.0, 50.0, 150.0, 50.4)])
    assert proc.check_manufacturability(part, LASER, 3.0).cuttable
    assert not proc.check_manufacturability(part, PLASMA, 3.0).cuttable


def test_a_web_that_will_burn_through_is_found_and_measured():
    part = plate(holes=[((100.0, 115.6), 4.0)])  # 0.4 mm from the top edge
    report = proc.check_manufacturability(part, LASER, 6.0)
    defect = report.of(Code.BRIDGE_TOO_NARROW)[0]
    assert 0.39 <= defect.measurements["gap"] <= 0.4  # lower bound on 0.4
    assert defect.measurements["minimum"] == pytest.approx(6.0)
    assert not report.cuttable


def test_two_holes_too_close_together_are_caught():
    part = plate(holes=[((90.0, 60.0), 5.0), ((101.0, 60.0), 5.0)])
    report = proc.check_manufacturability(part, LASER, 3.0)
    assert report.of(Code.BRIDGE_TOO_NARROW)


def test_generous_spacing_raises_nothing():
    part = plate(holes=[((50.0, 60.0), 5.0), ((150.0, 60.0), 5.0)])
    assert proc.check_manufacturability(part, LASER, 3.0).cuttable


def test_a_sharp_internal_corner_is_a_router_problem_and_not_a_laser_one():
    """A beam has no radius. A 6 mm cutter has 3 mm of one, and the corner
    comes out at R3 however the drawing shows it."""
    l_shape = Face2D.create(
        Loop2D(
            [
                Line2D((0.0, 0.0), (100.0, 0.0)),
                Line2D((100.0, 0.0), (100.0, 40.0)),
                Line2D((100.0, 40.0), (40.0, 40.0)),
                Line2D((40.0, 40.0), (40.0, 100.0)),
                Line2D((40.0, 100.0), (0.0, 100.0)),
                Line2D((0.0, 100.0), (0.0, 0.0)),
            ]
        )
    )
    assert not proc.check_manufacturability(l_shape, LASER, 10.0).of(
        Code.INTERNAL_CORNER_TOO_SHARP
    )
    router = proc.check_manufacturability(l_shape, ROUTER, 10.0)
    corners = router.of(Code.INTERNAL_CORNER_TOO_SHARP)
    assert len(corners) == 1
    assert corners[0].location.point == pytest.approx((40.0, 40.0))
    assert corners[0].measurements["tool_radius"] == pytest.approx(3.0)


def test_a_corner_radiused_to_the_cutter_is_accepted():
    radiused = Face2D.create(
        Loop2D(
            [
                Line2D((0.0, 0.0), (100.0, 0.0)),
                Line2D((100.0, 0.0), (100.0, 40.0)),
                Line2D((100.0, 40.0), (43.0, 40.0)),
                Arc2D((43.0, 43.0), 3.0, -math.pi / 2, math.pi),
                Line2D((40.0, 43.0), (40.0, 100.0)),
                Line2D((40.0, 100.0), (0.0, 100.0)),
                Line2D((0.0, 100.0), (0.0, 0.0)),
            ]
        )
    )
    assert not proc.check_manufacturability(radiused, ROUTER, 10.0).of(
        Code.INTERNAL_CORNER_TOO_SHARP
    )


def test_a_part_bigger_than_the_sheet_is_reported():
    big = plate(width=3500.0, height=1000.0)
    report = proc.check_manufacturability(big, LASER, 3.0)
    assert report.of(Code.PART_EXCEEDS_SHEET)
    assert not report.cuttable


def test_a_part_that_only_fits_turned_is_accepted():
    """2800 x 1400 will not fit 1500 x 3000 the way it is drawn, and will."""
    turned = plate(width=1400.0, height=2800.0)
    assert not proc.check_manufacturability(turned, LASER, 3.0).of(Code.PART_EXCEEDS_SHEET)


def test_material_thicker_than_the_process_will_cut():
    report = proc.check_manufacturability(plate(), LASER, 30.0)
    defect = report.of(Code.THICKNESS_EXCEEDS_PROCESS)[0]
    assert defect.measurements["limit"] == pytest.approx(20.0)
    assert proc.check_manufacturability(plate(), WATERJET, 30.0).cuttable


# ==========================================================================
# Feed models
# ==========================================================================


def test_a_thermal_feed_falls_with_thickness():
    rates = [LASER.feed.rate(t) for t in (1.0, 3.0, 6.0, 12.0)]
    assert rates == sorted(rates, reverse=True)
    assert rates[0] == pytest.approx(8000.0)


def test_edm_speed_is_a_fixed_cut_area_per_minute():
    """Not a fitted exponent: the machine erodes area, so the path rate is
    exactly inversely proportional to the thickness."""
    edm = proc.process("wire_edm").feed
    assert edm.rate(10.0) == pytest.approx(edm.rate(20.0) * 2.0)


def test_routing_adds_a_pass_every_few_millimetres():
    feed = proc.PassFeed(feed=2500.0, pass_depth=3.0)
    assert feed.passes(3.0) == 1
    assert feed.passes(3.1) == 2
    assert feed.passes(12.0) == 4
    assert feed.rate(12.0) == pytest.approx(625.0)


def test_a_zero_thickness_is_refused_rather_than_dividing_by_zero():
    with pytest.raises(ValueError):
        LASER.feed.rate(0.0)


# ==========================================================================
# What it costs
# ==========================================================================


def test_cut_length_is_exact_because_arcs_have_closed_form_length():
    part = plate(holes=[((100.0, 60.0), 10.0)], width=200.0, height=120.0)
    estimate = proc.quote(part, LASER, 6.0)
    expected = 2 * (200.0 + 120.0) + 2 * math.pi * 10.0
    assert estimate.cut_length == pytest.approx(expected, rel=1e-12)


def test_every_loop_costs_a_pierce():
    part = plate(holes=[((60.0, 60.0), 5.0), ((100.0, 60.0), 5.0), ((140.0, 60.0), 5.0)])
    estimate = proc.quote(part, LASER, 6.0)
    assert estimate.pierces == 4  # the profile plus three holes
    assert estimate.pierce_seconds == pytest.approx(4 * LASER.pierce_seconds)


def test_piercing_dominates_a_part_that_is_mostly_holes_on_a_waterjet():
    """Eight seconds a pierce is why a perforated part is quoted on a laser."""
    many = plate(holes=[((20.0 + 20.0 * i, 60.0), 4.0) for i in range(8)])
    laser = proc.quote(many, LASER, 6.0)
    jet = proc.quote(many, WATERJET, 6.0)
    assert jet.pierce_seconds > 8 * laser.pierce_seconds
    assert jet.cycle_seconds > laser.cycle_seconds


def test_setup_is_charged_once_so_a_hundred_are_not_a_hundred_times_one():
    estimate = proc.quote(plate(holes=[((100.0, 60.0), 5.0)]), LASER, 6.0)
    assert estimate.batch_seconds(100) == pytest.approx(
        estimate.setup_seconds + 100 * estimate.cycle_seconds
    )
    assert estimate.unit_cost(100) < estimate.unit_cost(1)
    assert estimate.unit_cost(1) == pytest.approx(estimate.machine_cost)


def test_a_thicker_plate_takes_longer_on_the_same_machine():
    part = plate(holes=[((100.0, 60.0), 8.0)])
    assert proc.quote(part, LASER, 12.0).cut_seconds > proc.quote(part, LASER, 3.0).cut_seconds


def test_the_quote_serialises():
    estimate = proc.quote(plate(holes=[((100.0, 60.0), 5.0)]), LASER, 6.0)
    payload = estimate.to_dict()
    assert payload["pierces"] == 2
    assert payload["seconds"]["cycle"] > 0
    assert payload["cost"]["each_at_100"] < payload["cost"]["one"]
    assert payload["process"]["kerf"] == pytest.approx(LASER.kerf)


# ==========================================================================
# Comparing machines
# ==========================================================================


def test_the_cheapest_machine_is_not_always_one_that_can_make_the_part():
    """The finding the whole module exists for.

    Plasma is the fastest and cheapest here and cannot produce the holes.
    The comparison has to sort producible ahead of cheap, or it is worse
    than useless -- it is a quote somebody will send.
    """
    part = plate(holes=[((60.0 + 60.0 * i, 60.0), 3.0) for i in range(3)])
    results = proc.compare(part, 6.0)

    plasma_quote = next(q for q, _ in results if "plasma" in q.process.name)
    best_quote, best_report = results[0]

    assert best_report.cuttable
    assert plasma_quote.machine_cost < best_quote.machine_cost  # cheaper
    plasma_report = next(r for q, r in results if "plasma" in q.process.name)
    assert not plasma_report.cuttable  # and cannot do it


def test_comparison_can_be_narrowed_to_the_machines_a_shop_owns():
    results = proc.compare(plate(), 6.0, names=["fiber_laser", "plasma"])
    assert len(results) == 2


def test_the_comparison_table_names_the_blocker():
    part = plate(holes=[((100.0, 60.0), 3.0)])
    text = proc.describe_comparison(proc.compare(part, 6.0), quantity=50)
    assert "HOLE_TOO_SMALL_FOR_THICKNESS" in text
    assert "OK" in text


# ==========================================================================
# Overriding the defaults with shop data
# ==========================================================================


def test_a_shop_can_replace_any_default_and_keep_the_rest():
    """The built-in numbers are rules of thumb; a shop's own are better."""
    mine = LASER.but(kerf=0.12, machine_rate=140.0)
    assert mine.kerf == 0.12
    assert mine.machine_rate == 140.0
    assert mine.max_thickness == LASER.max_thickness
    assert LASER.kerf == 0.2  # the original is untouched

    part = plate(cutouts=[rectangle(50.0, 50.0, 150.0, 50.15)])
    assert not proc.check_manufacturability(part, LASER, 3.0).cuttable
    assert proc.check_manufacturability(part, mine, 3.0).cuttable


def test_an_unknown_process_says_what_is_available():
    with pytest.raises(ValueError, match="fiber_laser"):
        proc.process("laser_beam_of_doom")


# ==========================================================================
# End to end
# ==========================================================================


def test_quote_a_drawing_from_disk(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (200, 0), (200, 120), (0, 120)], close=True)
    msp.add_circle((100, 60), 10.0)
    path = tmp_path / "part.dxf"
    doc.saveas(path)

    estimate, report = proc.quote_drawing(path, "fiber_laser", 6.0)
    assert estimate is not None
    assert estimate.pierces == 2
    assert report.cuttable
    assert report.metrics["process"].startswith("fibre laser")


def test_quoting_an_unreadable_file_reports_rather_than_raising(tmp_path):
    estimate, report = proc.quote_drawing(tmp_path / "nope.dxf", "fiber_laser", 6.0)
    assert estimate is None
    assert report.at_least(Severity.CRITICAL)
