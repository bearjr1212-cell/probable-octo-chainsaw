"""2D profile extraction and cleanup shared by all blueprint parsers.

Every parser (DXF, SVG, raster image) reduces its input down to a flat list
of point chains ("did the pen touch down here and trace this path"). This
module turns that loose, possibly-unordered, possibly-open collection of
chains into clean :class:`shapely.geometry.MultiPolygon` profiles with holes
correctly nested -- the shared representation the rest of the pipeline
(extrusion, multi-view intersection, export) builds on.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

from shapely.geometry import MultiPolygon, Polygon

Point2D = Tuple[float, float]
Chain = List[Point2D]

#: Default distance (in drawing units) below which two points are
#: considered coincident. Blueprints are rarely drawn with perfectly
#: matching endpoints, so a small tolerance is needed to weld segments
#: back into closed loops.
DEFAULT_TOLERANCE = 1e-6


def flatten_arc(
    center: Point2D,
    radius: float,
    start_deg: float,
    end_deg: float,
    segments_per_revolution: int = 64,
) -> Chain:
    """Sample a circular arc into a polyline, from ``start_deg`` to ``end_deg``.

    Angles are in degrees, counter-clockwise, matching DXF/SVG convention.
    """
    start = math.radians(start_deg)
    end = math.radians(end_deg)
    if end <= start:
        end += 2 * math.pi
    steps = max(2, round(segments_per_revolution * (end - start) / (2 * math.pi)))
    return [
        (
            center[0] + radius * math.cos(start + (end - start) * i / steps),
            center[1] + radius * math.sin(start + (end - start) * i / steps),
        )
        for i in range(steps + 1)
    ]


def flatten_circle(center: Point2D, radius: float, segments: int = 64) -> Chain:
    """Sample a full circle into a closed polyline."""
    return flatten_arc(center, radius, 0, 360, segments_per_revolution=segments)


def _key(p: Point2D, tolerance: float) -> Point2D:
    if tolerance <= 0:
        return p
    decimals = max(0, -int(math.floor(math.log10(tolerance))))
    return (round(p[0], decimals), round(p[1], decimals))


def weld_chains(chains: Iterable[Chain], tolerance: float = DEFAULT_TOLERANCE) -> List[Chain]:
    """Join open polylines that share endpoints (within ``tolerance``).

    Blueprints are frequently authored as independent LINE/ARC entities
    rather than a single closed polyline, so raw geometry rarely arrives
    pre-joined into loops. This repeatedly stitches chains together at
    matching endpoints until nothing more can be joined; the result may
    still contain open chains (dangling construction lines, dimensions,
    text) which callers should discard.
    """
    remaining = [list(c) for c in chains if len(c) >= 2]
    result: List[Chain] = []

    while remaining:
        chain = remaining.pop(0)
        grew = True
        while grew:
            grew = False
            for i, other in enumerate(remaining):
                if _key(chain[-1], tolerance) == _key(other[0], tolerance):
                    chain = chain + other[1:]
                elif _key(chain[-1], tolerance) == _key(other[-1], tolerance):
                    chain = chain + list(reversed(other))[1:]
                elif _key(chain[0], tolerance) == _key(other[-1], tolerance):
                    chain = other + chain[1:]
                elif _key(chain[0], tolerance) == _key(other[0], tolerance):
                    chain = list(reversed(other)) + chain[1:]
                else:
                    continue
                remaining.pop(i)
                grew = True
                break
        result.append(chain)

    return result


def closed_loops(chains: Sequence[Chain], tolerance: float = DEFAULT_TOLERANCE) -> List[Chain]:
    """Filter welded chains down to the ones that form closed loops."""
    loops = []
    for chain in chains:
        if len(chain) >= 3 and _key(chain[0], tolerance) == _key(chain[-1], tolerance):
            loops.append(chain)
    return loops


def _clean_loop(loop: Chain, tolerance: float) -> Chain:
    """Collapse near-duplicate consecutive points and snap the ring shut.

    Arc/circle sampling accumulates floating-point error, so the point
    meant to close a loop back onto its start is often off by a few ULPs
    rather than exactly equal. Left alone, that produces a near-zero-length
    edge that makes downstream triangulation (the extrusion cap and side
    walls) non-watertight, so it is snapped away here rather than trusting
    every caller to avoid it.
    """
    cleaned = [loop[0]]
    for point in loop[1:]:
        prev = cleaned[-1]
        if math.hypot(point[0] - prev[0], point[1] - prev[1]) > tolerance:
            cleaned.append(point)
    if len(cleaned) > 1 and math.hypot(cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]) <= tolerance:
        cleaned[-1] = cleaned[0]
    elif cleaned[-1] != cleaned[0]:
        cleaned.append(cleaned[0])
    return cleaned


def loops_to_polygons(loops: Iterable[Chain], tolerance: float = DEFAULT_TOLERANCE) -> MultiPolygon:
    """Turn a set of closed point loops into polygons-with-holes.

    Loops nest by point-in-polygon containment: a loop directly inside a
    solid becomes a hole in it; a loop directly inside a hole becomes a new
    solid island (e.g. a boss cast inside a pocket); and so on to arbitrary
    depth. This mirrors the even-odd nesting rule used for font glyphs and
    is the standard way to recover holes from unordered CAD/SVG loop data.
    """
    solids: List[Polygon] = []
    for loop in loops:
        cleaned = _clean_loop(loop, tolerance)
        if len(cleaned) < 4:  # 3 unique points + closing point
            continue
        poly = Polygon(cleaned)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < tolerance:
            continue
        if isinstance(poly, MultiPolygon):
            solids.extend(g for g in poly.geoms if g.area >= tolerance)
        else:
            solids.append(poly)

    if not solids:
        return MultiPolygon()

    n = len(solids)
    parent = [-1] * n
    for i in range(n):
        point = solids[i].representative_point()
        best_parent, best_area = -1, math.inf
        for j in range(n):
            if j == i or solids[j].area <= solids[i].area:
                continue
            if solids[j].area < best_area and solids[j].contains(point):
                best_parent, best_area = j, solids[j].area
        parent[i] = best_parent

    def depth(i: int) -> int:
        d, seen = 0, set()
        while parent[i] != -1 and i not in seen:
            seen.add(i)
            i = parent[i]
            d += 1
        return d

    depths = [depth(i) for i in range(n)]
    children = [[] for _ in range(n)]
    for i in range(n):
        if parent[i] != -1:
            children[parent[i]].append(i)

    output = []
    for i in range(n):
        if depths[i] % 2 != 0:
            continue  # holes are consumed by their parent solid below
        holes = [solids[c].exterior.coords for c in children[i] if depths[c] % 2 == 1]
        output.append(Polygon(solids[i].exterior.coords, holes))

    return MultiPolygon(output)


def profile_bounds(profile: MultiPolygon) -> Tuple[float, float, float, float]:
    """Return ``(min_x, min_y, max_x, max_y)`` for a profile, or zeros if empty."""
    if profile.is_empty:
        return (0.0, 0.0, 0.0, 0.0)
    return profile.bounds
