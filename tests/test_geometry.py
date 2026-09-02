import math

from blueprint23d.geometry import (
    closed_loops,
    flatten_arc,
    flatten_circle,
    loops_to_polygons,
    weld_chains,
)


def square(x0, y0, size):
    return [(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size), (x0, y0)]


def test_weld_chains_joins_open_segments_into_a_loop():
    # A 10x10 square drawn as four independent LINE-like segments, in a
    # shuffled order and mixed directions, as a real CAD export would.
    segments = [
        [(0, 0), (10, 0)],
        [(10, 10), (10, 0)],  # reversed
        [(10, 10), (0, 10)],
        [(0, 10), (0, 0)],
    ]
    welded = weld_chains(segments)
    loops = closed_loops(welded)
    assert len(loops) == 1
    assert len(loops[0]) == 5  # 4 unique points + closing point


def test_loops_to_polygons_handles_hole_and_island_nesting():
    outer = square(0, 0, 20)
    hole = square(5, 5, 10)
    island = square(7, 7, 2)

    profile = loops_to_polygons([outer, hole, island])
    polys = list(profile.geoms)

    # outer-with-hole is one solid, the island inside the hole is another
    assert len(polys) == 2
    areas = sorted(p.area for p in polys)
    assert math.isclose(areas[0], 4.0, rel_tol=1e-6)  # the island, 2x2
    assert math.isclose(areas[1], 400.0 - 100.0, rel_tol=1e-6)  # 20x20 minus 10x10 hole
    with_hole = max(polys, key=lambda p: p.area)
    assert len(with_hole.interiors) == 1


def test_flatten_arc_and_circle_endpoints():
    arc = flatten_arc((0, 0), 5, 0, 90)
    assert math.isclose(arc[0][0], 5.0, abs_tol=1e-9)
    assert math.isclose(arc[0][1], 0.0, abs_tol=1e-9)
    assert math.isclose(arc[-1][0], 0.0, abs_tol=1e-9)
    assert math.isclose(arc[-1][1], 5.0, abs_tol=1e-9)

    circle = flatten_circle((0, 0), 3, segments=128)
    assert math.isclose(circle[0][0], circle[-1][0], abs_tol=1e-9)
    assert math.isclose(circle[0][1], circle[-1][1], abs_tol=1e-9)
    radii = [math.hypot(x, y) for x, y in circle]
    assert all(math.isclose(r, 3.0, rel_tol=1e-6) for r in radii)
