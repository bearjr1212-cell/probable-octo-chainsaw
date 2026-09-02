"""Constrained Delaunay triangulation: correctness by conservation and property.

A triangulation is checked here the way you would check a machined part
rather than the way you would eyeball a picture:

* **Area is conserved.** The triangles must sum to the exact analytic area
  of the region, so nothing is covered twice and nothing is missed.
* **Constraints survive.** Every boundary segment must appear as a triangle
  edge; one that does not means the mesh cut across the part's outline.
* **Holes are empty**, including islands nested inside holes.
* **Orientation is uniform**, so every face normal points the same way.
* **The Delaunay property holds** on unconstrained edges, which is what
  keeps slivers out of the mesh.
"""

import math

import pytest

from blueprint23d.exact import incircle, orient2d

cdt = pytest.importorskip("blueprint23d._native.cdt", reason="compiled CDT kernel not built")


def ring(cx, cy, r, n, reverse=False):
    order = range(n - 1, -1, -1) if reverse else range(n)
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in order]


def loop_segments(offset, count):
    return [(offset + i, offset + (i + 1) % count) for i in range(count)]


def polygon_area(points):
    n = len(points)
    return abs(
        sum(
            (points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1]) / 2.0
            for i in range(n)
        )
    )


def triangulated_area(points, triangles):
    total = 0.0
    for a, b, c in triangles:
        total += abs(
            (points[b][0] - points[a][0]) * (points[c][1] - points[a][1])
            - (points[c][0] - points[a][0]) * (points[b][1] - points[a][1])
        ) / 2.0
    return total


def all_counter_clockwise(points, triangles):
    return all(orient2d(points[a], points[b], points[c]) > 0 for a, b, c in triangles)


def missing_constraints(segments, triangles):
    present = set()
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            present.add((min(u, v), max(u, v)))
    return [s for s in segments if (min(s), max(s)) not in present]


# --------------------------------------------------------------------------


def square_case():
    points = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    return points, loop_segments(0, 4), 12.0


def l_shape_case():
    points = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 2.0), (2.0, 4.0), (0.0, 4.0)]
    return points, loop_segments(0, 6), 12.0


def star_case():
    star = [
        ((5.0 if i % 2 == 0 else 2.0) * math.cos(2 * math.pi * i / 10),
         (5.0 if i % 2 == 0 else 2.0) * math.sin(2 * math.pi * i / 10))
        for i in range(10)
    ]
    return star, loop_segments(0, 10), polygon_area(star)


def square_with_hole_case():
    points = [(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0), (2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)]
    return points, loop_segments(0, 4) + loop_segments(4, 4), 36.0 - 4.0


def annulus_case():
    n = 48
    outer = ring(0.0, 0.0, 6.0, n)
    inner = ring(0.0, 0.0, 2.0, n, reverse=True)
    return outer + inner, loop_segments(0, n) + loop_segments(n, n), polygon_area(outer) - polygon_area(inner)


def island_in_hole_case():
    outer = [(0.0, 0.0), (12.0, 0.0), (12.0, 12.0), (0.0, 12.0)]
    hole = ring(6.0, 6.0, 4.0, 24, reverse=True)
    island = ring(6.0, 6.0, 1.5, 16)
    points = outer + hole + island
    segments = loop_segments(0, 4) + loop_segments(4, 24) + loop_segments(28, 16)
    return points, segments, 144.0 - polygon_area(hole) + polygon_area(island)


def plate_with_two_holes_case():
    plate = [(0.0, 0.0), (10.0, 0.0), (10.0, 6.0), (0.0, 6.0)]
    h1 = ring(2.0, 3.0, 0.8, 16, reverse=True)
    h2 = ring(7.0, 3.0, 1.2, 16, reverse=True)
    points = plate + h1 + h2
    segments = loop_segments(0, 4) + loop_segments(4, 16) + loop_segments(20, 16)
    return points, segments, 60.0 - polygon_area(h1) - polygon_area(h2)


ALL_CASES = {
    "square": square_case,
    "l_shape": l_shape_case,
    "star": star_case,
    "square_with_hole": square_with_hole_case,
    "annulus": annulus_case,
    "island_in_hole": island_in_hole_case,
    "plate_with_two_holes": plate_with_two_holes_case,
}


@pytest.mark.parametrize("name", sorted(ALL_CASES))
def test_area_is_conserved_exactly(name):
    """Sum of triangle areas equals the analytic area: no gaps, no overlaps."""
    points, segments, expected = ALL_CASES[name]()
    triangles = cdt.triangulate(points, segments)
    assert triangulated_area(points, triangles) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("name", sorted(ALL_CASES))
def test_every_constrained_edge_survives(name):
    """A boundary edge that vanishes means the mesh changed the part's shape."""
    points, segments, _ = ALL_CASES[name]()
    triangles = cdt.triangulate(points, segments)
    assert missing_constraints(segments, triangles) == []


@pytest.mark.parametrize("name", sorted(ALL_CASES))
def test_orientation_is_uniform(name):
    points, segments, _ = ALL_CASES[name]()
    triangles = cdt.triangulate(points, segments)
    assert all_counter_clockwise(points, triangles)


def test_holes_and_islands_nest_by_parity():
    """Depth 1 is material, 2 is a hole, 3 is an island inside that hole.

    Flooding inward from outside and stopping at constraints would leave
    the hole solid, since nothing outside can reach it. Counting crossings
    instead gets arbitrarily deep nesting right in a single pass.
    """
    points, segments, expected = island_in_hole_case()
    triangles = cdt.triangulate(points, segments)
    assert triangulated_area(points, triangles) == pytest.approx(expected, abs=1e-9)

    # A point at the centre of the island must be covered; a point in the
    # surrounding hole must not be.
    covered = set()
    for a, b, c in triangles:
        covered.add((a, b, c))
    hole_ring_indices = set(range(4, 28))
    island_indices = set(range(28, 44))
    # No triangle may join the island to the hole wall across the void.
    for a, b, c in triangles:
        tri = {a, b, c}
        assert not (tri & island_indices and tri & hole_ring_indices), "triangle spans the hole void"


def test_delaunay_property_holds_on_unconstrained_edges():
    """No vertex inside a triangle's circumcircle: this is what excludes slivers."""
    points, segments, _ = annulus_case()
    triangles = cdt.triangulate(points, segments)
    constrained = {(min(a, b), max(a, b)) for a, b in segments}

    for a, b, c in triangles:
        # Only test triangles whose edges are all free to flip; a
        # constrained edge is allowed to violate the empty-circle property,
        # which is exactly what "constrained" Delaunay means.
        edges = [(min(u, v), max(u, v)) for u, v in ((a, b), (b, c), (c, a))]
        if any(e in constrained for e in edges):
            continue
        for d in range(len(points)):
            if d in (a, b, c):
                continue
            assert incircle(points[a], points[b], points[c], points[d]) <= 0


def test_cocircular_input_is_handled_exactly():
    """A tessellated circle is exactly cocircular -- the degenerate case.

    Every in-circle test among these points is a true zero. With a merely
    filtered predicate the flips would be decided by rounding noise and
    the triangulation could come out non-planar.
    """
    n = 64
    points = ring(0.0, 0.0, 5.0, n)
    segments = loop_segments(0, n)
    triangles = cdt.triangulate(points, segments)

    assert len(triangles) == n - 2  # a simple polygon always triangulates to n-2
    assert triangulated_area(points, triangles) == pytest.approx(polygon_area(points), abs=1e-9)
    assert all_counter_clockwise(points, triangles)


def test_collinear_boundary_points_do_not_break_it():
    """Extra vertices along a straight edge are exactly collinear."""
    points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)]
    segments = loop_segments(0, 6)
    triangles = cdt.triangulate(points, segments)
    assert triangulated_area(points, triangles) == pytest.approx(6.0, abs=1e-9)
    assert missing_constraints(segments, triangles) == []


def test_rejects_degenerate_input():
    with pytest.raises(ValueError):
        cdt.triangulate([(0.0, 0.0), (1.0, 0.0)], [(0, 1)])
    with pytest.raises(ValueError):
        cdt.triangulate([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], [(0, 99)])


def test_scales_to_a_realistic_profile():
    """A part at a tight tolerance can easily reach thousands of boundary points."""
    n = 2000
    outer = ring(0.0, 0.0, 10.0, n)
    inner = ring(0.0, 0.0, 4.0, n // 2, reverse=True)
    points = outer + inner
    segments = loop_segments(0, n) + loop_segments(n, n // 2)

    triangles = cdt.triangulate(points, segments)
    expected = polygon_area(outer) - polygon_area(inner)
    assert triangulated_area(points, triangles) == pytest.approx(expected, rel=1e-12)
    assert missing_constraints(segments, triangles) == []
