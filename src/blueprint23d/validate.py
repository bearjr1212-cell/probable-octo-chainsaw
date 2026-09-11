"""Is this drawing actually cuttable?

The three faults that stop a job, in the order they cost money:

**Open contours.** A cutter follows a closed path. A profile with a gap in
it is not a path, and the machine either refuses the file or wanders off.
The gap is usually tiny — a vertex snapped to the wrong place, a trim that
left 0.0004 mm behind — which is exactly why nobody spots it by eye. So the
report gives the coordinates and the gap size, and the size is what decides
whether it is a modelling slip or a genuinely unfinished outline.

**Self-intersections.** An outline that crosses itself has no consistent
inside, so offsetting for kerf produces garbage and the part comes out
wrong rather than failing loudly.

**Duplicate and overlapping edges.** Two entities along the same stretch of
outline make the machine cut it twice. On a laser that means a burnt,
out-of-tolerance edge and doubled cycle time; on a waterjet it is just
wasted money. This is the single most common defect in exported DXF, it is
completely invisible on screen, and almost nothing checks for it.

All three decisions bottom out in the exact predicates, which matters more
here than anywhere else in the package: the whole question is whether two
things that are *nearly* the same are the same, and that is precisely where
naive floating point answers at random.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .brep import Face2D, Loop2D
from .curves import Arc2D, Curve2D, Line2D
from .diagnostics import Code, Location, Report, Severity
from .exact import orient2d, segments_properly_intersect

Point2 = Tuple[float, float]
Segment = Tuple[Point2, Point2]

#: How finely curves are sampled before the segment-level checks. Fine
#: enough that a real crossing cannot hide between samples on any part
#: this tool handles.
DEFAULT_SAMPLE_TOLERANCE = 1e-3

#: Endpoints closer than this count as the same point.
DEFAULT_TOLERANCE = 1e-7


# --------------------------------------------------------------------------
# Spatial pruning
# --------------------------------------------------------------------------


def _bbox(segment: Segment) -> Tuple[float, float, float, float]:
    (x0, y0), (x1, y1) = segment
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _candidate_pairs(segments: Sequence[Segment], cell: float) -> Iterable[Tuple[int, int]]:
    """Index pairs whose bounding boxes could touch.

    A uniform grid rather than a sweep line: far less code to get wrong,
    and the pruning is what matters — the exact predicate then decides
    every surviving pair properly.
    """
    if cell <= 0.0:
        cell = 1.0
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for index, segment in enumerate(segments):
        minx, miny, maxx, maxy = _bbox(segment)
        for gx in range(int(math.floor(minx / cell)), int(math.floor(maxx / cell)) + 1):
            for gy in range(int(math.floor(miny / cell)), int(math.floor(maxy / cell)) + 1):
                buckets.setdefault((gx, gy), []).append(index)

    seen = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                pair = (a, b) if a < b else (b, a)
                if pair not in seen:
                    seen.add(pair)
                    yield pair


def _grid_cell(segments: Sequence[Segment]) -> float:
    if not segments:
        return 1.0
    lengths = [math.dist(a, b) for a, b in segments]
    average = sum(lengths) / len(lengths)
    return max(average * 2.0, 1e-9)


# --------------------------------------------------------------------------
# Self-intersection
# --------------------------------------------------------------------------


def _close(a: Point2, b: Point2, tolerance: float) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tolerance


def _segment_intersection_point(p1: Point2, p2: Point2, q1: Point2, q2: Point2) -> Optional[Point2]:
    dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
    dx2, dy2 = q2[0] - q1[0], q2[1] - q1[1]
    denom = dx1 * dy2 - dy1 * dx2
    if denom == 0.0:
        return None
    t = ((q1[0] - p1[0]) * dy2 - (q1[1] - p1[1]) * dx2) / denom
    return (p1[0] + t * dx1, p1[1] + t * dy1)


def find_self_intersections(
    loop: Loop2D,
    sample_tolerance: float = DEFAULT_SAMPLE_TOLERANCE,
    tolerance: float = DEFAULT_TOLERANCE,
) -> List[Point2]:
    """Points where a loop crosses itself.

    The loop is sampled to segments and every candidate pair is decided by
    :func:`~blueprint23d.exact.segments_properly_intersect`, which ignores
    shared endpoints and collinear touching — those are how a closed
    outline is *supposed* to join, not faults.
    """
    points = loop.tessellate(sample_tolerance)
    count = len(points)
    if count < 4:
        return []

    segments: List[Segment] = [(points[i], points[(i + 1) % count]) for i in range(count)]
    hits: List[Point2] = []

    for i, j in _candidate_pairs(segments, _grid_cell(segments)):
        # Neighbouring segments share an endpoint by construction.
        if j == i + 1 or (i == 0 and j == count - 1):
            continue
        (p1, p2), (q1, q2) = segments[i], segments[j]
        if not segments_properly_intersect(p1, p2, q1, q2):
            continue
        point = _segment_intersection_point(p1, p2, q1, q2)
        if point is None:
            continue
        if not any(_close(point, existing, sample_tolerance * 4) for existing in hits):
            hits.append(point)
    return hits


# --------------------------------------------------------------------------
# Duplicate and overlapping edges
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Duplicate:
    """Two curves describing the same stretch of outline."""

    first: int
    second: int
    kind: str  # "duplicate" or "overlap"
    at: Point2
    length: float


def _line_overlap(a: Line2D, b: Line2D, tolerance: float) -> Optional[Tuple[Point2, float]]:
    """Collinear lines sharing more than a point."""
    # Exactly collinear, judged by the exact predicate rather than a slope.
    if orient2d(a.p0, a.p1, b.p0) != 0 or orient2d(a.p0, a.p1, b.p1) != 0:
        return None

    dx, dy = a.p1[0] - a.p0[0], a.p1[1] - a.p0[1]
    length = math.hypot(dx, dy)
    if length <= tolerance:
        return None
    ux, uy = dx / length, dy / length

    def project(p: Point2) -> float:
        return (p[0] - a.p0[0]) * ux + (p[1] - a.p0[1]) * uy

    a0, a1 = 0.0, length
    b0, b1 = sorted((project(b.p0), project(b.p1)))
    low, high = max(a0, b0), min(a1, b1)
    if high - low <= tolerance:
        return None

    mid = 0.5 * (low + high)
    return ((a.p0[0] + ux * mid, a.p0[1] + uy * mid), high - low)


def _arc_overlap(a: Arc2D, b: Arc2D, tolerance: float) -> Optional[Tuple[Point2, float]]:
    """Concentric arcs of equal radius sharing an angular range."""
    if abs(a.radius - b.radius) > tolerance:
        return None
    if math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]) > tolerance:
        return None

    def span(arc: Arc2D) -> Tuple[float, float]:
        start = arc.start_angle
        sweep = arc.sweep
        return (start, start + sweep) if sweep >= 0 else (start + sweep, start)

    a0, a1 = span(a)
    b0, b1 = span(b)
    # Compare on a common branch.
    shift = round((b0 - a0) / (2 * math.pi))
    b0 -= shift * 2 * math.pi
    b1 -= shift * 2 * math.pi

    low, high = max(a0, b0), min(a1, b1)
    if high - low <= 1e-9:
        return None

    mid = 0.5 * (low + high)
    arc_length = (high - low) * a.radius
    if arc_length <= tolerance:
        return None
    return ((a.center[0] + a.radius * math.cos(mid), a.center[1] + a.radius * math.sin(mid)), arc_length)


def find_duplicates(
    curves: Sequence[Curve2D],
    tolerance: float = DEFAULT_TOLERANCE,
) -> List[Duplicate]:
    """Curves that retrace ground another curve already covers."""
    found: List[Duplicate] = []
    boxes = [curve.bounds() for curve in curves]

    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            bi, bj = boxes[i], boxes[j]
            # Cheap rejection before any real work.
            if bi[2] < bj[0] - tolerance or bj[2] < bi[0] - tolerance:
                continue
            if bi[3] < bj[1] - tolerance or bj[3] < bi[1] - tolerance:
                continue

            a, b = curves[i], curves[j]
            hit: Optional[Tuple[Point2, float]] = None
            if isinstance(a, Line2D) and isinstance(b, Line2D):
                hit = _line_overlap(a, b, tolerance)
            elif isinstance(a, Arc2D) and isinstance(b, Arc2D):
                hit = _arc_overlap(a, b, tolerance)
            else:
                # Mixed or free-form: fall back to endpoint coincidence,
                # which catches the common "exported twice" case without
                # claiming to find partial overlaps it cannot see.
                same = (_close(a.start, b.start, tolerance) and _close(a.end, b.end, tolerance)) or (
                    _close(a.start, b.end, tolerance) and _close(a.end, b.start, tolerance)
                )
                if same and type(a) is type(b):
                    hit = (a.point(0.5), a.length())

            if hit is None:
                continue
            at, length = hit
            identical = _close(a.start, b.start, tolerance) and _close(a.end, b.end, tolerance)
            reversed_identical = _close(a.start, b.end, tolerance) and _close(a.end, b.start, tolerance)
            kind = "duplicate" if (identical or reversed_identical) else "overlap"
            found.append(Duplicate(i, j, kind, at, length))
    return found


def find_zero_length(curves: Sequence[Curve2D], tolerance: float = DEFAULT_TOLERANCE) -> List[int]:
    return [i for i, curve in enumerate(curves) if curve.length() <= tolerance]


# --------------------------------------------------------------------------
# Open contours
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Gap:
    """Where an open chain fails to close, and by how much."""

    start: Point2
    end: Point2
    distance: float
    curves: int


def describe_open_chains(chains: Sequence[Sequence[Curve2D]]) -> List[Gap]:
    """Locate the break in each chain that would not close."""
    gaps: List[Gap] = []
    for chain in chains:
        if not chain:
            continue
        head = chain[0].start
        tail = chain[-1].end
        gaps.append(Gap(tail, head, math.dist(tail, head), len(chain)))
    return gaps


def nearest_endpoint_distance(gap: Gap, chains: Sequence[Sequence[Curve2D]]) -> float:
    """How far the loose end is from any other chain's end.

    A chain that ends 0.02 mm from another chain's start is a snapping
    failure and closes trivially. One that ends 40 mm away is a genuinely
    unfinished drawing, and saying which is which is the difference
    between a fixable file and a phone call to the customer.
    """
    best = math.inf
    for chain in chains:
        for candidate in (chain[0].start, chain[-1].end):
            distance = math.dist(gap.start, candidate)
            if distance > 1e-12:
                best = min(best, distance)
    return best


# --------------------------------------------------------------------------
# The intake check
# --------------------------------------------------------------------------


def validate_curves(
    curves: Sequence[Curve2D],
    open_chains: Sequence[Sequence[Curve2D]] = (),
    tolerance: float = DEFAULT_TOLERANCE,
    report: Optional[Report] = None,
    layer: Optional[str] = None,
) -> Report:
    """Check raw curves for duplicates, zero-length entities and open chains."""
    report = report or Report()

    for index in find_zero_length(curves, tolerance):
        curve = curves[index]
        report.add(
            Code.ZERO_LENGTH_EDGE,
            Severity.WARNING,
            f"{type(curve).__name__} has no length",
            Location(point=curve.start, layer=layer),
            length=curve.length(),
        )

    for duplicate in find_duplicates(curves, tolerance):
        severity = Severity.ERROR if duplicate.kind == "duplicate" else Severity.WARNING
        code = Code.DUPLICATE_EDGE if duplicate.kind == "duplicate" else Code.OVERLAPPING_EDGE
        report.add(
            code,
            severity,
            f"two entities cover the same {duplicate.length:.4g} of outline; "
            "the machine would cut it twice",
            Location(point=duplicate.at, layer=layer),
            overlap_length=duplicate.length,
        )

    for gap in describe_open_chains(open_chains):
        nearest = nearest_endpoint_distance(gap, open_chains)
        # A hair-thin gap is a snapping slip; a wide one is unfinished work.
        hint = ""
        if math.isfinite(nearest) and nearest <= tolerance * 1000:
            hint = f"; the nearest other endpoint is {nearest:.4g} away"
        report.add(
            Code.OPEN_CONTOUR,
            Severity.ERROR,
            f"a chain of {gap.curves} curve(s) does not close: "
            f"{gap.distance:.4g} between its ends{hint}",
            Location(point=gap.start, other_point=gap.end, layer=layer),
            gap=gap.distance,
            nearest_endpoint=nearest if math.isfinite(nearest) else -1.0,
        )

    return report


def validate_face(
    face: Face2D,
    sample_tolerance: float = DEFAULT_SAMPLE_TOLERANCE,
    tolerance: float = DEFAULT_TOLERANCE,
    report: Optional[Report] = None,
) -> Report:
    """Check an assembled face for self-intersection."""
    report = report or Report()

    for index, loop in enumerate(face.loops()):
        name = "outer boundary" if index == 0 else f"hole {index}"
        for point in find_self_intersections(loop, sample_tolerance, tolerance):
            report.add(
                Code.SELF_INTERSECTION,
                Severity.ERROR,
                f"the {name} crosses itself",
                Location(point=point),
            )
    return report


def validate_drawing(
    path,
    layer: Optional[str] = None,
    tolerance: float = DEFAULT_TOLERANCE,
    sample_tolerance: float = DEFAULT_SAMPLE_TOLERANCE,
) -> Tuple[List[Face2D], Report]:
    """Read a drawing and run every intake check on it."""
    from .loaders import load_faces

    report = Report(source=str(path))

    try:
        faces, open_chains = load_faces(path, layer=layer)
    except (ValueError, FileNotFoundError) as exc:
        report.add(Code.FILE_UNREADABLE, Severity.CRITICAL, str(exc))
        return [], report

    curves: List[Curve2D] = []
    for face in faces:
        for loop in face.loops():
            curves.extend(loop.curves)
    for chain in open_chains:
        curves.extend(chain)

    validate_curves(curves, open_chains, tolerance, report, layer)
    for face in faces:
        validate_face(face, sample_tolerance, tolerance, report)

    if not faces:
        report.add(
            Code.NO_CLOSED_PROFILE,
            Severity.CRITICAL,
            "no closed profile was found in the drawing",
        )

    report.metrics["faces"] = len(faces)
    report.metrics["curves"] = len(curves)
    if faces:
        largest = max(faces, key=lambda f: f.area())
        report.metrics["area"] = largest.area()
        report.metrics["holes"] = len(largest.inners)

    return faces, report
