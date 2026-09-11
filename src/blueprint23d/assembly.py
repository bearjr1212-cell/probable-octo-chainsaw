"""Assembling loose exact curves into nested faces.

Parsers hand back a bag of curves in no particular order: a DXF drawing is
a pile of LINE, ARC and SPLINE entities, and nothing in the file says which
of them form the outer boundary and which bound a hole. This module
recovers that structure without ever flattening the curves.

Two steps:

**Welding.** Curves are chained head-to-tail into closed loops by matching
endpoints within a tolerance. A spatial hash keyed on the rounded endpoint
makes this linear rather than quadratic, which matters because a detailed
drawing easily runs to thousands of entities.

**Nesting.** A loop directly inside a solid is a hole; a loop inside a hole
is a solid island; and so on. Depth parity settles it, the same rule used
for font glyphs and in the triangulator. Containment is decided on the
tessellated loops with the exact predicates, so the answer is exactly right
for the tessellation and can only differ from the true curved region within
the tessellation tolerance of the boundary -- and since loops that nest are
not near-touching in any sane drawing, that is never the deciding factor.

The curves themselves are untouched throughout: an arc that arrives as an
arc leaves as an arc, with its centre and radius bit-for-bit intact.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .brep import Face2D, Loop2D
from .curves import Curve2D, DEFAULT_CHORD_TOLERANCE
from .exact import point_in_polygon

Point2 = Tuple[float, float]

#: Endpoints closer than this are treated as the same vertex.
DEFAULT_WELD_TOLERANCE = 1e-7

#: Loops enclosing less area than this are treated as degenerate artefacts.
_DEGENERATE_AREA = 1e-12


def _key(point: Point2, tolerance: float) -> Tuple[int, int]:
    scale = 1.0 / max(tolerance, 1e-15)
    return (int(round(point[0] * scale)), int(round(point[1] * scale)))


def _close(a: Point2, b: Point2, tolerance: float) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tolerance


def weld_curves(
    curves: Iterable[Curve2D],
    tolerance: float = DEFAULT_WELD_TOLERANCE,
) -> Tuple[List[Loop2D], List[List[Curve2D]]]:
    """Chain curves into closed loops.

    Returns the closed loops and, separately, any chains that would not
    close. Open chains are handed back rather than silently dropped or
    force-closed: in a real drawing they are usually dimension lines,
    centre marks or leader arrows, and the caller is better placed to
    decide whether their presence means the drawing was misread.
    """
    remaining = [c for c in curves]
    if not remaining:
        return [], []

    # Bucket curve endpoints so a partner lookup is a hash hit, not a scan.
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for index, curve in enumerate(remaining):
        for point in (curve.start, curve.end):
            buckets.setdefault(_key(point, tolerance), []).append(index)

    used = [False] * len(remaining)
    loops: List[Loop2D] = []
    open_chains: List[List[Curve2D]] = []

    def find_partner(point: Point2, exclude: int) -> Optional[Tuple[int, bool]]:
        """Find an unused curve starting or ending at ``point``."""
        kx, ky = _key(point, tolerance)
        # Check the neighbouring cells too: a point can round either way
        # across a bucket boundary.
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for index in buckets.get((kx + dx, ky + dy), ()):
                    if used[index] or index == exclude:
                        continue
                    candidate = remaining[index]
                    if _close(candidate.start, point, tolerance):
                        return index, True
                    if _close(candidate.end, point, tolerance):
                        return index, False
        return None

    for seed in range(len(remaining)):
        if used[seed]:
            continue
        used[seed] = True
        chain: List[Curve2D] = [remaining[seed]]

        # Grow forwards from the chain's end until it closes or runs out.
        while True:
            tail = chain[-1].end
            if _close(tail, chain[0].start, tolerance) and len(chain) >= 1:
                break
            found = find_partner(tail, exclude=-1)
            if found is None:
                break
            index, forward = found
            used[index] = True
            chain.append(remaining[index] if forward else remaining[index].reverse())

        if len(chain) >= 1 and _close(chain[-1].end, chain[0].start, tolerance):
            try:
                loops.append(Loop2D(chain, tolerance=max(tolerance, 1e-9)))
                continue
            except ValueError:
                pass
        open_chains.append(chain)

    return loops, open_chains


def nest_loops(
    loops: Sequence[Loop2D],
    tolerance: float = DEFAULT_CHORD_TOLERANCE,
) -> List[Face2D]:
    """Group loops into faces by containment parity.

    Even depth is material and becomes a face; odd depth is a hole in the
    face directly containing it. An island inside a hole is even again and
    becomes a face of its own, which is why arbitrarily deep nesting works
    without special cases.
    """
    # Drop loops that enclose nothing. Real drawings contain zero-length
    # artefacts -- a doubled vertex, a degenerate segment left by an
    # editor -- and letting one through produces a "face" with no area that
    # then fails every downstream operation for reasons that look unrelated.
    loops = [loop for loop in loops if abs(loop.signed_area()) > _DEGENERATE_AREA]
    if not loops:
        return []

    polygons = [loop.tessellate(tolerance) for loop in loops]
    areas = [abs(loop.signed_area()) for loop in loops]
    count = len(loops)

    parent = [-1] * count
    for i in range(count):
        # Any vertex of loop i serves as a probe; loops in a valid drawing
        # do not cross, so one point decides containment for the whole loop.
        probe = polygons[i][0]
        best, best_area = -1, math.inf
        for j in range(count):
            if i == j or areas[j] <= areas[i]:
                continue
            if areas[j] < best_area and point_in_polygon(probe, polygons[j]) > 0:
                best, best_area = j, areas[j]
        parent[i] = best

    def depth_of(index: int) -> int:
        depth, seen = 0, set()
        while parent[index] != -1 and index not in seen:
            seen.add(index)
            index = parent[index]
            depth += 1
        return depth

    depths = [depth_of(i) for i in range(count)]
    children: List[List[int]] = [[] for _ in range(count)]
    for i in range(count):
        if parent[i] != -1:
            children[parent[i]].append(i)

    faces: List[Face2D] = []
    for i in range(count):
        if depths[i] % 2 != 0:
            continue  # a hole; consumed by its parent below
        holes = [loops[c] for c in children[i] if depths[c] % 2 == 1]
        faces.append(Face2D.create(loops[i], holes))
    return faces


def faces_from_curves(
    curves: Iterable[Curve2D],
    weld_tolerance: float = DEFAULT_WELD_TOLERANCE,
    nest_tolerance: float = DEFAULT_CHORD_TOLERANCE,
) -> Tuple[List[Face2D], List[List[Curve2D]]]:
    """Weld and nest in one step; also returns the chains that did not close."""
    loops, open_chains = weld_curves(curves, weld_tolerance)
    return nest_loops(loops, nest_tolerance), open_chains


def largest_face(faces: Sequence[Face2D]) -> Face2D:
    """The face with the greatest area -- usually the part, on a busy sheet."""
    if not faces:
        raise ValueError("no faces were assembled from the drawing")
    return max(faces, key=lambda f: f.area())
