import math

import pytest
from shapely.geometry import box

from blueprint23d.reconstruct import ViewSpec, from_views


def test_from_views_two_view_intersection_matches_analytic_box():
    # top view: 10x10 square extruded 10 deep (Z) -> a 10x10x10 cube.
    # front view: 6x20 rectangle (X x Z) extruded 6 deep (Y) -> a 6x6x20 box.
    # Their intersection is exactly a 6x6x10 box (bounded by the narrower
    # footprint in X/Y and the shorter height in Z), the textbook check for
    # this reconstruction technique.
    top = ViewSpec(profile=box(0, 0, 10, 10), depth=10)
    front = ViewSpec(profile=box(0, 0, 6, 20), depth=6)

    mesh = from_views(top=top, front=front)

    assert mesh.is_watertight
    assert math.isclose(mesh.volume, 6 * 6 * 10, rel_tol=1e-6)
    (minx, miny, minz), (maxx, maxy, maxz) = mesh.bounds
    assert math.isclose(maxx - minx, 6, rel_tol=1e-6)
    assert math.isclose(maxy - miny, 6, rel_tol=1e-6)
    assert math.isclose(maxz - minz, 10, rel_tol=1e-6)


def test_from_views_three_views_agree():
    top = ViewSpec(profile=box(0, 0, 10, 10), depth=10)
    front = ViewSpec(profile=box(0, 0, 6, 20), depth=6)
    side = ViewSpec(profile=box(0, 0, 6, 10), depth=6)  # redundant, same constraint

    mesh = from_views(top=top, front=front, side=side)

    assert mesh.is_watertight
    assert math.isclose(mesh.volume, 6 * 6 * 10, rel_tol=1e-6)


def test_from_views_min_align_matches_center_align_up_to_translation():
    top = ViewSpec(profile=box(0, 0, 10, 10), depth=10)
    front = ViewSpec(profile=box(0, 0, 6, 20), depth=6)

    mesh_min = from_views(top=top, front=front, align="min")

    assert mesh_min.is_watertight
    assert math.isclose(mesh_min.volume, 6 * 6 * 10, rel_tol=1e-6)
    (minx, miny, minz), (maxx, maxy, maxz) = mesh_min.bounds
    assert math.isclose(minx, 0, abs_tol=1e-6)
    assert math.isclose(miny, 0, abs_tol=1e-6)
    assert math.isclose(minz, 0, abs_tol=1e-6)


def test_from_views_requires_at_least_two_views():
    with pytest.raises(ValueError):
        from_views(top=ViewSpec(profile=box(0, 0, 1, 1), depth=1))


def test_from_views_raises_on_non_overlapping_profiles():
    # A square frame (hole dead in the middle) as the top view, and a tiny
    # square that lands entirely inside that hole as the front view: after
    # centering, no point can satisfy both silhouettes at once.
    ring = box(0, 0, 10, 10).difference(box(4, 4, 6, 6))
    top = ViewSpec(profile=ring, depth=10)
    front = ViewSpec(profile=box(0, 0, 1, 1), depth=1)
    with pytest.raises(ValueError):
        from_views(top=top, front=front)
