"""Comparing two revisions of the same part.

The expensive failure here is quiet: rev C arrives, looks like rev B, and
one hole has moved half a millimetre. These tests build pairs of drawings
that differ in each of the ways a real revision does, and check that the
comparison names the change rather than merely noticing that something is
different.
"""

import math

import ezdxf
import pytest

from blueprint23d import revisions
from blueprint23d.brep import Face2D, Loop2D
from blueprint23d.curves import Arc2D, Line2D
from blueprint23d.diagnostics import Code, Severity


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def rectangle(x0, y0, x1, y1):
    return Loop2D(
        [
            Line2D((x0, y0), (x1, y0)),
            Line2D((x1, y0), (x1, y1)),
            Line2D((x1, y1), (x0, y1)),
            Line2D((x0, y1), (x0, y0)),
        ]
    )


def plate(holes=(), x1=100.0, y1=60.0):
    """A plate with circular holes given as ``(centre, radius)``."""
    inners = [Loop2D([Arc2D.full_circle(centre, radius)]) for centre, radius in holes]
    return [Face2D.create(rectangle(0.0, 0.0, x1, y1), inners)]


def write_plate(tmp_path, name, holes=(), x1=100.0, y1=60.0):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (x1, 0), (x1, y1), (0, y1)], close=True)
    for centre, radius in holes:
        msp.add_circle(centre, radius)
    path = tmp_path / name
    doc.saveas(path)
    return path


# --------------------------------------------------------------------------
# Nothing changed
# --------------------------------------------------------------------------


def test_the_same_drawing_twice_is_reported_identical():
    before = plate([((25.0, 30.0), 3.175)])
    diff = revisions.diff_faces(before, plate([((25.0, 30.0), 3.175)]))
    assert diff.identical
    assert not diff.changed and not diff.added and not diff.removed
    assert "identical" in diff.describe()
    assert diff.report.cuttable


def test_unchanged_means_exactly_unchanged_not_approximately():
    """A hole 1e-12 off is not the same hole, and is not claimed to be."""
    before = plate([((25.0, 30.0), 3.175)])
    after = plate([((25.0, 30.0), 3.175 + 1e-12)])
    diff = revisions.diff_faces(before, after)
    assert not diff.identical
    assert diff.changed


def test_a_loop_that_starts_at_a_different_vertex_is_still_the_same_loop():
    """Export order is not a design change."""
    a = Face2D.create(rectangle(0.0, 0.0, 100.0, 60.0))
    rotated = Loop2D(
        [
            Line2D((100.0, 60.0), (0.0, 60.0)),
            Line2D((0.0, 60.0), (0.0, 0.0)),
            Line2D((0.0, 0.0), (100.0, 0.0)),
            Line2D((100.0, 0.0), (100.0, 60.0)),
        ]
    )
    assert revisions.diff_faces([a], [Face2D.create(rotated)]).identical


def test_a_loop_traced_the_other_way_round_is_still_the_same_loop():
    outline = rectangle(0.0, 0.0, 100.0, 60.0)
    assert revisions.diff_faces(
        [Face2D.create(outline)], [Face2D.create(outline.reverse())]
    ).identical


# --------------------------------------------------------------------------
# A feature changed size
# --------------------------------------------------------------------------


def test_a_hole_that_grew_is_named_with_both_diameters():
    """The flagship case: same place, different drill."""
    before = plate([((25.0, 30.0), 3.175)])
    after = plate([((25.0, 30.0), 3.25)])
    diff = revisions.diff_faces(before, after, before_label="rev B", after_label="rev C")

    changed = diff.changed
    assert len(changed) == 1
    change = changed[0]
    assert change.before.diameter == pytest.approx(6.35)
    assert change.after.diameter == pytest.approx(6.5)
    assert change.size_delta == pytest.approx(0.15)
    assert change.moved == 0.0
    assert "Ø6.35 → Ø6.5" in "; ".join(change.reasons)

    defects = diff.report.of(Code.FEATURE_CHANGED)
    assert defects and defects[0].severity is Severity.WARNING
    assert defects[0].measurements["diameter_before"] == pytest.approx(6.35)


def test_matching_is_by_position_so_a_size_change_can_be_seen_at_all():
    """Pairing on diameter would silently make this test impossible.

    Two holes of different size swap diameters in place. Matched by
    position, that is two changes; matched by size it would look unchanged.
    """
    before = plate([((25.0, 30.0), 3.0), ((75.0, 30.0), 5.0)])
    after = plate([((25.0, 30.0), 5.0), ((75.0, 30.0), 3.0)])
    diff = revisions.diff_faces(before, after)
    assert len(diff.changed) == 2
    assert not diff.added and not diff.removed


# --------------------------------------------------------------------------
# A feature moved
# --------------------------------------------------------------------------


def test_a_hole_that_moved_slightly_is_caught_with_its_displacement():
    """0.4 mm: invisible on a print, fatal in an assembly."""
    before = plate([((25.0, 30.0), 3.175)])
    after = plate([((25.4, 30.0), 3.175)])
    diff = revisions.diff_faces(before, after)

    change = diff.changed[0]
    assert change.moved == pytest.approx(0.4)
    assert change.size_delta is None
    assert "+0.4" in "; ".join(change.reasons)
    assert diff.report.of(Code.FEATURE_CHANGED)[0].measurements["moved"] == pytest.approx(0.4)


def test_a_hole_that_moved_further_than_its_neighbours_are_spaced_is_not_guessed_at():
    """Beyond that distance, "moved" and "deleted plus added" are the same
    observation, and the honest answer is the one that does not invent a
    correspondence."""
    before = plate([((20.0, 30.0), 3.0), ((40.0, 30.0), 3.0)])
    after = plate([((20.0, 30.0), 3.0), ((80.0, 30.0), 3.0)])
    diff = revisions.diff_faces(before, after)
    assert len(diff.removed) == 1
    assert len(diff.added) == 1
    assert diff.removed[0].before.anchor == pytest.approx((40.0, 30.0))
    assert diff.added[0].after.anchor == pytest.approx((80.0, 30.0))


def test_a_lone_feature_is_recognised_however_far_it_moved():
    """With nothing to confuse it with, there is no ambiguity to protect."""
    before = plate([((10.0, 10.0), 3.0)])
    after = plate([((90.0, 50.0), 3.0)])
    diff = revisions.diff_faces(before, after)
    assert len(diff.changed) == 1
    assert diff.changed[0].moved == pytest.approx(math.dist((10, 10), (90, 50)))


def test_a_coin_toss_pairing_says_so():
    before = plate([((30.0, 30.0), 3.0), ((40.0, 30.0), 3.0)])
    after = plate([((35.0, 30.0), 3.0), ((45.0, 30.0), 3.0)])
    diff = revisions.diff_faces(before, after)
    assert diff.report.of(Code.REVISION_AMBIGUOUS)
    assert any(c.ambiguous for c in diff.changes)


# --------------------------------------------------------------------------
# Features came and went
# --------------------------------------------------------------------------


def test_an_added_hole_is_reported_as_added():
    before = plate([((25.0, 30.0), 3.0)])
    after = plate([((25.0, 30.0), 3.0), ((75.0, 30.0), 4.0)])
    diff = revisions.diff_faces(before, after, after_label="rev C")

    assert len(diff.added) == 1
    assert diff.added[0].after.anchor == pytest.approx((75.0, 30.0))
    defect = diff.report.of(Code.FEATURE_ADDED)[0]
    assert "new in rev C" in defect.message
    assert not diff.report.of(Code.FEATURE_REMOVED)


def test_a_removed_hole_is_reported_as_removed():
    before = plate([((25.0, 30.0), 3.0), ((75.0, 30.0), 4.0)])
    after = plate([((25.0, 30.0), 3.0)])
    diff = revisions.diff_faces(before, after)
    assert len(diff.removed) == 1
    assert diff.report.of(Code.FEATURE_REMOVED)
    assert not diff.report.of(Code.FEATURE_ADDED)


def test_the_outer_profile_is_compared_too():
    diff = revisions.diff_faces(plate(x1=100.0), plate(x1=97.0))
    change = diff.changed[0]
    assert change.feature.kind == "outer profile"
    assert change.size_delta == pytest.approx((97.0 - 100.0) * 60.0)


def test_a_profile_never_pairs_with_a_hole():
    """Different roles are not interchangeable, however close they sit."""
    before = plate([((50.0, 30.0), 3.0)])
    after = plate()
    diff = revisions.diff_faces(before, after)
    assert len(diff.removed) == 1
    assert diff.removed[0].before.role == "opening"
    assert diff.unchanged  # the profile itself is untouched


# --------------------------------------------------------------------------
# Redrawn rather than redesigned
# --------------------------------------------------------------------------


def test_a_circle_exported_as_a_polyline_is_not_called_unchanged():
    """Same nominal hole, rebuilt out of 64 chords.

    It is a real difference and the comparison says so: the inscribed
    polygon is genuinely smaller, it is no longer a circle, and a cutting
    program made from it has 64 moves where it had one arc.
    """
    circle = Loop2D([Arc2D.full_circle((50.0, 30.0), 5.0)])
    n = 64
    points = [
        (50.0 + 5.0 * math.cos(2 * math.pi * i / n), 30.0 + 5.0 * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    faceted = Loop2D([Line2D(points[i], points[(i + 1) % n]) for i in range(n)])

    before = [Face2D.create(rectangle(0.0, 0.0, 100.0, 60.0), [circle])]
    after = [Face2D.create(rectangle(0.0, 0.0, 100.0, 60.0), [faceted])]
    diff = revisions.diff_faces(before, after)

    change = diff.changed[0]
    reasons = "; ".join(change.reasons)
    assert "hole → cutout" in reasons
    assert "1 → 64 edges" in reasons
    assert change.before.edges == 1
    assert change.after.edges == 64
    assert change.size_delta < 0  # an inscribed polygon is the smaller hole


def test_a_circle_that_starts_at_a_different_angle_is_the_same_circle():
    """Phase is an artefact of the exporter, not a property of the hole."""
    a = Loop2D([Arc2D((50.0, 30.0), 5.0, 0.0, 2.0 * math.pi)])
    b = Loop2D([Arc2D((50.0, 30.0), 5.0, math.pi, 3.0 * math.pi)])
    before = [Face2D.create(rectangle(0.0, 0.0, 100.0, 60.0), [a])]
    after = [Face2D.create(rectangle(0.0, 0.0, 100.0, 60.0), [b])]
    assert revisions.diff_faces(before, after).identical


def test_a_hole_that_became_a_slot_names_both_kinds():
    hole = Loop2D([Arc2D.full_circle((50.0, 30.0), 5.0)])
    slot = Loop2D(
        [
            Arc2D((45.0, 30.0), 5.0, math.pi / 2, 1.5 * math.pi),
            Line2D((45.0, 25.0), (55.0, 25.0)),
            Arc2D((55.0, 30.0), 5.0, 1.5 * math.pi, 2.5 * math.pi),
            Line2D((55.0, 35.0), (45.0, 35.0)),
        ]
    )
    before = [Face2D.create(rectangle(0.0, 0.0, 100.0, 60.0), [hole])]
    after = [Face2D.create(rectangle(0.0, 0.0, 100.0, 60.0), [slot])]
    diff = revisions.diff_faces(before, after)
    reasons = "; ".join(diff.changed[0].reasons)
    assert "hole → slot" in reasons


# --------------------------------------------------------------------------
# Feature extraction and signatures
# --------------------------------------------------------------------------


def test_a_circular_hole_anchors_exactly_at_its_centre():
    features = revisions.extract_features(plate([((25.0, 30.0), 3.175)]))
    hole = next(f for f in features if f.kind == "hole")
    assert hole.anchor == (25.0, 30.0)
    assert hole.diameter == 6.35


def test_extraction_order_does_not_depend_on_input_order():
    a = plate([((25.0, 30.0), 3.0), ((75.0, 30.0), 4.0)])
    b = plate([((75.0, 30.0), 4.0), ((25.0, 30.0), 3.0)])
    assert [f.anchor for f in revisions.extract_features(a)] == [
        f.anchor for f in revisions.extract_features(b)
    ]


def test_signatures_distinguish_curve_types_at_the_same_place():
    line = revisions.curve_signature(Line2D((0.0, 0.0), (10.0, 0.0)))
    arc = revisions.curve_signature(Arc2D((5.0, 0.0), 5.0, math.pi, 2.0 * math.pi))
    assert line != arc


# --------------------------------------------------------------------------
# Whole files
# --------------------------------------------------------------------------


def test_two_files_compare_end_to_end(tmp_path):
    before = write_plate(tmp_path, "revB.dxf", [((25.0, 30.0), 3.175)])
    after = write_plate(tmp_path, "revC.dxf", [((25.0, 30.0), 3.25)])
    diff = revisions.diff_drawings(before, after)
    assert diff.before_label == "revB"
    assert diff.after_label == "revC"
    assert len(diff.changed) == 1
    assert "revB → revC" in diff.describe()
    assert diff.to_dict()["counts"]["changed"] == 1


def test_two_files_marked_the_same_revision_that_differ_is_an_error(tmp_path):
    """Two prints both stamped rev C describing different parts. Somebody
    is about to cut the wrong one."""
    before = write_plate(tmp_path, "from_customer.dxf", [((25.0, 30.0), 3.175)])
    after = write_plate(tmp_path, "from_engineering.dxf", [((25.0, 30.0), 3.25)])
    diff = revisions.diff_drawings(before, after, revision_before="C", revision_after="C")

    undocumented = diff.report.of(Code.REVISION_UNDOCUMENTED)
    assert undocumented and undocumented[0].severity is Severity.ERROR
    assert not diff.report.cuttable


def test_matching_files_at_the_same_revision_raise_nothing(tmp_path):
    before = write_plate(tmp_path, "a.dxf", [((25.0, 30.0), 3.175)])
    after = write_plate(tmp_path, "b.dxf", [((25.0, 30.0), 3.175)])
    diff = revisions.diff_drawings(before, after, revision_before="C", revision_after="C")
    assert diff.identical
    assert not diff.report.of(Code.REVISION_UNDOCUMENTED)


def test_an_unreadable_file_reports_rather_than_raising(tmp_path):
    diff = revisions.diff_drawings(tmp_path / "nope.dxf", tmp_path / "also_nope.dxf")
    assert diff.changes == []
    assert diff.report.at_least(Severity.CRITICAL)
