"""Exactness certificates for the predicate layer.

These tests do not check that the predicates are *accurate*; they check
that they are *exact*. The distinction matters: an accurate predicate that
is occasionally wrong about a sign will still build a topologically broken
solid. Every case below compares against an independent ground truth
computed in rational arithmetic, where no rounding occurs at all.
"""

import math
import random
from fractions import Fraction

import pytest

from blueprint23d import exact
from blueprint23d.exact import (
    BACKEND,
    Interval,
    incircle,
    incircle_python,
    orient2d,
    orient2d_python,
    point_in_polygon,
    polygon_orientation,
    segments_properly_intersect,
)

EPS = 2.0**-53


def truth_orient2d(a, b, c):
    """Ground truth in exact rationals: every float is a dyadic rational."""
    ax, ay = Fraction(a[0]), Fraction(a[1])
    bx, by = Fraction(b[0]), Fraction(b[1])
    cx, cy = Fraction(c[0]), Fraction(c[1])
    det = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
    return (det > 0) - (det < 0)


def truth_incircle(a, b, c, d):
    ax, ay = Fraction(a[0]) - Fraction(d[0]), Fraction(a[1]) - Fraction(d[1])
    bx, by = Fraction(b[0]) - Fraction(d[0]), Fraction(b[1]) - Fraction(d[1])
    cx, cy = Fraction(c[0]) - Fraction(d[0]), Fraction(c[1]) - Fraction(d[1])
    det = (
        (ax * ax + ay * ay) * (bx * cy - cx * by)
        + (bx * bx + by * by) * (cx * ay - ax * cy)
        + (cx * cx + cy * cy) * (ax * by - bx * ay)
    )
    return (det > 0) - (det < 0)


def naive_orient2d(a, b, c):
    det = (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])
    return (det > 0) - (det < 0)


def kettner_lattice(rng, count):
    """Near-collinear points on a fine lattice, the classic failure family.

    Coordinates that differ only in their last few mantissa bits make the
    orientation determinant catastrophically cancel, so a naive evaluation
    returns the sign of the rounding error rather than of the geometry.
    """
    for _ in range(count):
        i, j, k, m = (rng.randint(0, 256) for _ in range(4))
        a = (0.5 + i * EPS, 0.5 + j * EPS)
        b = (12.0, 12.0)
        c = (24.0 + k * EPS * 32, 24.0 + m * EPS * 32)
        yield a, b, c


def test_naive_orientation_really_does_fail_here():
    """Guard the guard: confirm this input family defeats naive floating point.

    If this ever stops failing, the adversarial cases below have gone
    stale and the exactness tests are no longer testing anything.
    """
    rng = random.Random(7)
    failures = sum(
        1 for a, b, c in kettner_lattice(rng, 20000) if naive_orient2d(a, b, c) != truth_orient2d(a, b, c)
    )
    assert failures > 100, f"expected many naive failures, saw {failures}"


def test_orient2d_is_exact_on_adversarial_input():
    rng = random.Random(7)
    for a, b, c in kettner_lattice(rng, 20000):
        assert orient2d(a, b, c) == truth_orient2d(a, b, c)


def test_orient2d_exact_across_magnitudes():
    rng = random.Random(99)
    scales = [1e-12, 1e-6, 1.0, 1e6, 1e12]
    for _ in range(5000):
        s = rng.choice(scales)
        pts = [(rng.uniform(-1, 1) * s, rng.uniform(-1, 1) * s) for _ in range(3)]
        assert orient2d(*pts) == truth_orient2d(*pts)


def test_orient2d_exact_zero_on_collinear():
    # Exactly collinear points must return exactly 0, not "small".
    for k in range(2, 500):
        assert orient2d((0.0, 0.0), (1.0, 1.0), (float(k), float(k))) == 0
    assert orient2d((0.0, 0.0), (0.1, 0.3), (0.2, 0.6)) == 0
    assert orient2d((1e10, 1e10), (2e10, 2e10), (3e10, 3e10)) == 0


def test_incircle_is_exact():
    rng = random.Random(3)
    for _ in range(5000):
        pts = [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(4)]
        assert incircle(*pts) == truth_incircle(*pts)


def test_incircle_exact_on_cocircular_tessellation_points():
    """The degenerate case this kernel actually meets in production.

    Sampling a circle at uniform angles places points that are exactly
    cocircular, so every in-circle test among them is a true zero. A
    filtered-only implementation would return an arbitrary sign here and
    produce a non-Delaunay triangulation of every arc in the drawing.
    """
    rng = random.Random(5)
    for segments in (8, 16, 64, 128):
        ring = [
            (math.cos(2 * math.pi * k / segments), math.sin(2 * math.pi * k / segments))
            for k in range(segments)
        ]
        for _ in range(400):
            quad = rng.sample(ring, 4)
            assert incircle(*quad) == truth_incircle(*quad)

    # The canonical exact zero.
    assert incircle((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)) == 0


def test_incircle_exact_when_scaled_and_translated():
    """Precision must not depend on where the part sits in the coordinate system."""
    rng = random.Random(13)
    for scale in (1e-8, 1.0, 1e8):
        for offset in (0.0, 1e6):
            ring = [
                (
                    offset + scale * math.cos(2 * math.pi * k / 12),
                    offset + scale * math.sin(2 * math.pi * k / 12),
                )
                for k in range(12)
            ]
            for _ in range(200):
                quad = rng.sample(ring, 4)
                assert incircle(*quad) == truth_incircle(*quad)


@pytest.mark.skipif(BACKEND != "c", reason="compiled kernel not built")
def test_c_backend_matches_python_oracle():
    """The two backends reach the same answers by different routes.

    C resolves hard cases with Shewchuk expansions; Python resolves them
    with rational arithmetic. Agreement across adversarial input is
    therefore evidence about the C build -- including that the compiler
    honoured the strict-IEEE flags and did not contract a multiply-add
    into an FMA, which would silently break the error-free transforms.
    """
    rng = random.Random(21)
    for a, b, c in kettner_lattice(rng, 8000):
        assert orient2d(a, b, c) == orient2d_python(a, b, c)

    for segments in (12, 64):
        ring = [
            (math.cos(2 * math.pi * k / segments), math.sin(2 * math.pi * k / segments))
            for k in range(segments)
        ]
        for _ in range(1500):
            quad = rng.sample(ring, 4)
            assert incircle(*quad) == incircle_python(*quad)


def test_point_in_polygon_handles_rays_through_vertices():
    """Vertices exactly on the test ray are the classic crossing-count bug."""
    diamond = [(0.0, 0.0), (2.0, 2.0), (4.0, 0.0), (2.0, -2.0)]
    assert point_in_polygon((2.0, 0.0), diamond) == 1  # ray exits through 2 vertices
    assert point_in_polygon((-1.0, 0.0), diamond) == -1
    assert point_in_polygon((5.0, 0.0), diamond) == -1
    assert point_in_polygon((0.0, 0.0), diamond) == 0  # on a vertex
    assert point_in_polygon((1.0, 1.0), diamond) == 0  # on an edge

    square = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    assert point_in_polygon((2.0, 2.0), square) == 1
    assert point_in_polygon((4.0, 2.0), square) == 0
    assert point_in_polygon((4.0 + 1e-15, 2.0), square) == -1


def test_polygon_orientation_exact_on_slivers():
    """A sliver whose float area underflows still has a well-defined winding."""
    sliver = [(0.0, 0.0), (1.0, 0.0), (1.0, 1e-300)]
    assert polygon_orientation(sliver) == 1
    assert polygon_orientation(sliver[::-1]) == -1


def test_segment_crossing_distinguishes_touching_from_crossing():
    assert segments_properly_intersect((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0))
    # Shared endpoint is adjacency, not a crossing.
    assert not segments_properly_intersect((0.0, 0.0), (1.0, 1.0), (1.0, 1.0), (2.0, 0.0))
    # Collinear overlap is not a proper crossing either.
    assert not segments_properly_intersect((0.0, 0.0), (2.0, 0.0), (1.0, 0.0), (3.0, 0.0))


def test_interval_encloses_true_value():
    """An interval bound is a theorem: the true value is inside, always."""
    acc = Interval(0.0)
    for _ in range(10):
        acc = acc + Interval(0.1)
    assert acc.lo <= 1.0 <= acc.hi  # despite 0.1 being inexact in binary
    assert acc.width < 1e-14

    # Accumulated 0.1s in plain float drift off 1.0; the interval still covers it.
    naive = sum(0.1 for _ in range(10))
    assert acc.lo <= naive <= acc.hi

    two = Interval(2.0)
    assert two.sqrt().lo <= math.sqrt(2.0) <= two.sqrt().hi


def test_interval_sign_refuses_to_guess():
    assert Interval(2.0, 3.0).sign() == 1
    assert Interval(-3.0, -2.0).sign() == -1
    assert Interval(-1.0, 1.0).sign() is None  # straddles zero: no certificate


def test_predicate_stats_track_fallbacks():
    exact.reset_predicate_stats()
    stats = exact.predicate_stats()
    if BACKEND != "c":
        assert stats == {}
        return
    # Start at k=2 so all three points stay distinct. At k=0 and k=1 the
    # third point coincides with one of the first two, which zeroes a
    # product and lets the sign-agreement shortcut settle it without ever
    # reaching the adaptive cascade.
    for k in range(2, 202):
        orient2d((0.0, 0.0), (1.0, 1.0), (float(k), float(k)))  # all collinear
    stats = exact.predicate_stats()
    assert stats["orient2d_calls"] >= 200
    # Genuinely collinear distinct points cannot be resolved by the filter,
    # so every one of these must have escalated to the adaptive path.
    assert stats["orient2d_adaptive"] >= 200
