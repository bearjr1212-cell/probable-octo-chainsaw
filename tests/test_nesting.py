"""How many come off a sheet, and how much of it is scrap.

The estimate has to be a genuine lower bound -- an achievable layout, not
a hopeful one -- and it has to beat rows of bounding boxes on the parts
where that matters, because a bounding box is a terrible model of a
triangle.
"""

import math

import pytest

from blueprint23d import nesting, processes
from blueprint23d.brep import Face2D, Loop2D
from blueprint23d.curves import Arc2D, Line2D


def rectangle(x0, y0, x1, y1):
    return Loop2D(
        [
            Line2D((x0, y0), (x1, y0)),
            Line2D((x1, y0), (x1, y1)),
            Line2D((x1, y1), (x0, y1)),
            Line2D((x0, y1), (x0, y0)),
        ]
    )


def box(width, height):
    return Face2D.create(rectangle(0.0, 0.0, width, height))


def right_triangle(width, height):
    return Face2D.create(
        Loop2D(
            [
                Line2D((0.0, 0.0), (width, 0.0)),
                Line2D((width, 0.0), (0.0, height)),
                Line2D((0.0, height), (0.0, 0.0)),
            ]
        )
    )


def disc(radius):
    return Face2D.create(Loop2D([Arc2D.full_circle((0.0, 0.0), radius)]))


SHEET = (3000.0, 1500.0)


# ==========================================================================
# The bound has to hold
# ==========================================================================


@pytest.mark.parametrize(
    "face",
    [box(200.0, 120.0), right_triangle(200.0, 120.0), disc(100.0), box(40.0, 40.0)],
)
def test_the_estimate_never_claims_more_than_the_area_allows(face):
    """The one invariant that cannot be negotiated: parts do not overlap,
    so no layout places more than the sheet area divided by the part area.

    An estimate that breaches this is not optimistic, it is wrong, and the
    breach is silent -- a utilisation over 100% is the only symptom.
    """
    result = nesting.estimate(face, SHEET)
    assert result.per_sheet <= result.area_bound
    assert 0.0 <= result.utilisation <= 1.0


def test_a_part_larger_than_the_sheet_yields_nothing():
    result = nesting.estimate(box(4000.0, 200.0), SHEET)
    assert result.per_sheet == 0
    assert result.sheets_for(10) == 0
    assert not math.isfinite(result.material_per_part(200.0))


def test_a_part_that_only_fits_turned_is_found():
    """1400 x 2800 does not fit a 3000 x 1500 sheet as drawn."""
    result = nesting.estimate(box(1400.0, 2800.0), SHEET, gap=0.0, margin=0.0)
    assert result.per_sheet == 1
    assert result.layout.turned


# ==========================================================================
# Interlocking
# ==========================================================================


def test_two_triangles_make_a_rectangle_and_the_estimate_finds_it():
    """The case the whole module exists for.

    A right triangle is half its bounding box. Packed as boxes it yields
    exactly what the box yields; packed properly, its end-for-end twin
    fits against it and the sheet holds twice as many. The utilisation
    should land on the rectangle's, because that is literally the layout.
    """
    rectangle_result = nesting.estimate(box(200.0, 120.0), SHEET)
    triangle_result = nesting.estimate(right_triangle(200.0, 120.0), SHEET)

    assert triangle_result.per_sheet == 2 * rectangle_result.per_sheet
    assert triangle_result.layout.flipped
    assert triangle_result.utilisation == pytest.approx(
        rectangle_result.utilisation, rel=1e-9
    )


def test_the_paired_pitches_are_very_unequal_for_a_triangle():
    """A triangle sits almost on top of its twin and then the next pair has
    to clear the whole height. One uniform pitch would throw half of it
    away."""
    layout = nesting.estimate(right_triangle(200.0, 120.0), SHEET).layout
    assert layout.pitch_up < 0.2 * layout.pitch_back


def test_a_rectangle_gains_nothing_from_interlocking():
    """Nothing nests into a rectangle, and the estimate should not pretend
    otherwise: both pitches are the full height plus the gap."""
    layout = nesting.estimate(box(200.0, 120.0), SHEET, gap=5.0).layout
    assert layout.pitch_up == pytest.approx(layout.pitch_back)
    assert layout.pitch_up == pytest.approx(205.0, abs=5.0)


def test_discs_are_packed_in_offset_rows():
    """Rows of circles drop into the hollows of the row below. The pitch
    should come out near the hexagonal r√3, not the diameter."""
    result = nesting.estimate(disc(100.0), SHEET, gap=5.0)
    assert result.layout.shift > 0.0
    assert result.layout.pitch_up == pytest.approx(205.0 * math.sqrt(3) / 2, rel=0.05)
    assert result.utilisation > 0.75


def test_a_row_shift_cannot_slide_a_part_onto_its_neighbour():
    """A row is a train of parts, not one part.

    Shift a row of discs by nearly a whole pitch and the disc above lands
    on the *next* one along rather than in the hollow. Checking only the
    part directly opposite misses that entirely, and the estimate silently
    claims more area in parts than the sheet has.
    """
    profile = nesting.silhouette(disc(100.0))
    span = int(round(205.0 / profile.column_width))
    # A shift of nearly a full part pitch is the same as a tiny shift the
    # other way, and must cost the same vertical room.
    nearly_full = nesting._pair_pitch(profile, profile, span - 2, span, 5.0)
    tiny = nesting._pair_pitch(profile, profile, 2, span, 5.0)
    assert nearly_full == pytest.approx(tiny, rel=0.05)


# ==========================================================================
# The silhouette
# ==========================================================================


def test_a_rectangles_skyline_is_flat():
    profile = nesting.silhouette(box(200.0, 120.0))
    assert all(t == pytest.approx(120.0) for t in profile.columns)
    assert all(b == pytest.approx(0.0) for b in profile.floors)


def test_a_long_edge_reaches_every_column_it_crosses():
    """Two endpoints a hundred apart contribute to two columns unless the
    edge is densified, and the skyline comes out full of holes."""
    profile = nesting.silhouette(box(200.0, 120.0), columns=96)
    assert all(t is not None for t in profile.columns)


def test_flipping_a_profile_turns_it_end_for_end():
    profile = nesting.silhouette(right_triangle(200.0, 120.0))
    flipped = profile.flipped()
    assert flipped.columns[0] == pytest.approx(profile.height)
    assert flipped.floors[0] == pytest.approx(profile.height - profile.columns[-1])


def test_required_pitch_of_a_shape_against_itself_is_its_own_height():
    profile = nesting.silhouette(box(200.0, 120.0))
    assert nesting.required_pitch(profile, profile, 0, 5.0) == pytest.approx(125.0)


def test_a_shift_clear_of_the_part_reports_no_constraint():
    profile = nesting.silhouette(box(200.0, 120.0))
    assert nesting.required_pitch(profile, profile, 999, 5.0) == math.inf


# ==========================================================================
# Honesty about what is not known
# ==========================================================================


def test_a_concave_part_reports_headroom_a_real_nester_could_use():
    """The estimate is a lower bound, and says how much room is left above
    it rather than implying it is the answer."""
    result = nesting.estimate(
        Face2D.create(
            Loop2D(
                [
                    Line2D((0.0, 0.0), (200.0, 0.0)),
                    Line2D((200.0, 0.0), (200.0, 40.0)),
                    Line2D((200.0, 40.0), (40.0, 40.0)),
                    Line2D((40.0, 40.0), (40.0, 160.0)),
                    Line2D((40.0, 160.0), (0.0, 160.0)),
                    Line2D((0.0, 160.0), (0.0, 0.0)),
                ]
            )
        ),
        SHEET,
    )
    assert result.compactness < 0.5
    assert result.headroom > 0
    assert "headroom" in result.describe()


def test_a_rectangle_has_no_headroom_worth_mentioning():
    result = nesting.estimate(box(300.0, 250.0), SHEET, gap=0.0, margin=0.0)
    assert result.compactness == pytest.approx(1.0)
    assert result.headroom == 0
    assert "none" in result.describe()


def test_the_gap_and_margin_cost_real_parts():
    tight = nesting.estimate(box(200.0, 120.0), SHEET, gap=0.0, margin=0.0)
    loose = nesting.estimate(box(200.0, 120.0), SHEET, gap=20.0, margin=50.0)
    assert loose.per_sheet < tight.per_sheet


# ==========================================================================
# What it means for a price
# ==========================================================================


def test_sheets_needed_rounds_up_because_you_cannot_buy_two_thirds_of_one():
    result = nesting.estimate(box(200.0, 120.0), SHEET)
    per = result.per_sheet
    assert result.sheets_for(per) == 1
    assert result.sheets_for(per + 1) == 2


def test_material_per_part_carries_the_offcut():
    result = nesting.estimate(box(200.0, 120.0), SHEET)
    assert result.material_per_part(180.0) == pytest.approx(180.0 / result.per_sheet)


def test_the_estimate_uses_the_processes_own_sheet_and_kerf():
    laser = processes.process("fiber_laser")
    plasma = processes.process("plasma")
    part = box(200.0, 120.0)
    # A 2.5 mm plasma kerf eats more room between parts than a 0.2 laser one.
    assert nesting.estimate_for(part, plasma).gap > nesting.estimate_for(part, laser).gap
    assert nesting.estimate_for(part, laser).sheet == laser.sheet


def test_a_quote_costs_material_only_when_a_sheet_price_is_given():
    """Nothing is guessed: without a price there is no material line."""
    laser = processes.process("fiber_laser")
    part = box(200.0, 120.0)

    bare = processes.quote(part, laser, 6.0)
    assert bare.material_cost == 0.0
    assert bare.total_cost == pytest.approx(bare.machine_cost)

    priced = processes.quote(part, laser, 6.0, sheet_cost=180.0)
    assert priced.per_sheet > 0
    assert priced.material_cost == pytest.approx(180.0 / priced.per_sheet)
    assert priced.unit_cost(100) > bare.unit_cost(100)


def test_material_dominates_a_large_part_in_quantity():
    """At a hundred off, the sheet is the bill and the cut time is noise --
    which is why a quote from cut time alone is wrong in the expensive
    direction."""
    laser = processes.process("fiber_laser")
    priced = processes.quote(box(700.0, 700.0), laser, 6.0, sheet_cost=180.0)
    assert priced.material_cost > priced.batch_cost(100) / 100


def test_the_estimate_serialises():
    payload = nesting.estimate(right_triangle(200.0, 120.0), SHEET).to_dict()
    assert payload["per_sheet"] > 0
    assert payload["layout"]["flipped"] is True
    assert payload["layout"]["pitch_up"] < payload["layout"]["pitch_back"]
    assert payload["headroom"] >= 0
