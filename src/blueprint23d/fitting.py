"""Reconstructing exact CAD primitives from raster drawings.

A contour traced from pixels is a staircase: its vertices are quantised to
the sampling grid, and a drilled hole in it is a lumpy polygon with no
radius at all. This module recovers the geometry the drawing actually
describes, in three stages, each of which beats the pixel grid for a
different reason.

**1. Sub-pixel edge localisation.** An edge is not at a pixel; it is where
the intensity gradient peaks, which is almost always between pixels. The
gradient magnitude is sampled at the contour point and one pixel either
side *along the gradient direction*, and a parabola through those three
values is solved for its vertex. That places the edge to a fraction of a
pixel instead of the nearest whole one, and costs one interpolation per
point.

**2. Segmentation.** The refined chain is split into spans that are each
well explained by a single line or a single arc, greedily and with a
conditioning guard (see :mod:`_native.fitkernels`).

**3. Least-squares fitting.** Each span is fitted with total least squares
(lines) or Taubin plus Gauss-Newton (arcs). This is where the real
precision comes from: the per-point localisation error is independent, so
a fit over N points averages it down by roughly sqrt(N). Fitting a circle
to 200 points located to +/-0.5 px recovers its radius to a few
hundredths of a pixel -- an order of magnitude better than any individual
measurement in the input.

The output is ordinary :mod:`curves` primitives, so a scanned drawing
enters the same exact B-rep and STEP pipeline as a DXF, and a recovered
hole exports as a real cylinder.

Honest limits: precision depends on how much curvature the data actually
contains. A short, small-radius fillet under heavy noise is genuinely
unidentifiable -- its sagitta falls below the noise floor -- and this
module reports that condition rather than returning a confident number.
Every fit carries its RMS residual and the span's sagitta so a caller can
decide whether to trust it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .brep import Loop2D
from .curves import Arc2D, Curve2D, Line2D

Point2 = Tuple[float, float]

try:  # pragma: no cover - depends on whether the extension was built
    from ._native import fitkernels as _fitkernels
except ImportError:  # pragma: no cover
    _fitkernels = None

FIT_BACKEND = "c" if _fitkernels is not None else "unavailable"


@dataclass(frozen=True)
class FitReport:
    """Quality of one recovered primitive, so a caller can reject it."""

    kind: str
    points: int
    rms: float
    max_deviation: float
    sagitta: Optional[float] = None
    radius: Optional[float] = None

    def is_resolvable(self, noise_estimate: float) -> bool:
        """Whether the curvature was actually measurable above the noise."""
        if self.kind != "arc" or self.sagitta is None:
            return True
        return self.sagitta > 2.0 * noise_estimate

    def __str__(self) -> str:
        base = f"{self.kind} over {self.points} pts, rms={self.rms:.4g}, max={self.max_deviation:.4g}"
        if self.kind == "arc":
            return base + f", r={self.radius:.6g}, sagitta={self.sagitta:.4g}"
        return base


# --------------------------------------------------------------------------
# Sub-pixel edge localisation
# --------------------------------------------------------------------------


def _bilinear(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear sample of ``image`` at floating-point ``(x, y)``, clamped."""
    h, w = image.shape
    x = np.clip(xs, 0.0, w - 1.000001)
    y = np.clip(ys, 0.0, h - 1.000001)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    return (
        image[y0, x0] * (1 - fx) * (1 - fy)
        + image[y0, x1] * fx * (1 - fy)
        + image[y1, x0] * (1 - fx) * fy
        + image[y1, x1] * fx * fy
    )


#: How far along the normal to look for the gradient ridge, in pixels.
#:
#: It has to cover the distance between where the contour *starts* and
#: where the edge actually is. ``findContours`` returns the centres of the
#: outermost ink pixels, which sit about half a pixel inside the true
#: boundary, and the Gaussian used for the gradient spreads the ridge a
#: little further. Two pixels covers both with room to spare; going wider
#: only risks finding a neighbouring edge on a thin feature.
DEFAULT_SEARCH = 2.0


def refine_subpixel(
    gray: np.ndarray, points: Sequence[Point2], search: float = DEFAULT_SEARCH
) -> List[Point2]:
    """Move each contour point onto the true gradient ridge, to sub-pixel accuracy.

    For an area-sampled (anti-aliased) edge the intensity crosses its
    midpoint exactly at the geometric boundary, and blurring does not move
    that, so the gradient magnitude peaks *on* the edge. Finding that peak
    to sub-pixel accuracy is the whole job.

    Two steps, and the first one is the one that matters:

    **Walk to the peak.** The gradient magnitude is sampled at whole-pixel
    steps along the normal across ``±search``, and the largest interior
    sample is taken as the ridge. Skipping this and interpolating around
    the starting point instead is the obvious implementation and it is
    subtly, systematically wrong: ``findContours`` hands back the centres
    of boundary *pixels*, about half a pixel inside the edge, so the
    parabola's vertex frequently lands beyond the half-pixel where a
    three-sample fit is valid. Rejecting those (the only safe thing to do
    with them) leaves precisely the points that needed moving most sitting
    where they started, and the recovered contour comes out a third of a
    pixel small -- a bias no amount of averaging removes, because every
    point is displaced the same way.

    **Interpolate.** A parabola through the peak sample and its two
    neighbours has its vertex at

    .. math:: \\delta = \\frac{g_- - g_+}{2\\,(g_- - 2 g_0 + g_+)}

    which is the remaining sub-pixel offset. A degenerate (non-peaked)
    triple means the model does not hold there, and the sample stays at
    the integer peak rather than moving somewhere arbitrary.
    """
    import cv2

    blurred = cv2.GaussianBlur(gray.astype(np.float64), (0, 0), 1.0)
    # Scharr is more rotationally symmetric than Sobel, which matters when
    # the same edge must localise identically at any angle.
    gx = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
    magnitude = np.hypot(gx, gy)

    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return []
    xs, ys = pts[:, 0], pts[:, 1]

    gxi = _bilinear(gx, xs, ys)
    gyi = _bilinear(gy, xs, ys)
    norm = np.hypot(gxi, gyi)
    valid = norm > 1e-12
    nx = np.where(valid, gxi / np.where(valid, norm, 1.0), 0.0)
    ny = np.where(valid, gyi / np.where(valid, norm, 1.0), 0.0)

    steps = np.arange(-math.ceil(search), math.ceil(search) + 1, 1.0)
    profile = np.stack(
        [_bilinear(magnitude, xs + t * nx, ys + t * ny) for t in steps]
    )  # (steps, points)

    # The peak has to have a neighbour on each side for the parabola, so
    # the ends of the window are not candidates.
    peak = np.argmax(profile[1:-1], axis=0) + 1
    index = np.arange(pts.shape[0])
    g0 = profile[peak, index]
    gm = profile[peak - 1, index]
    gp = profile[peak + 1, index]

    denom = gm - 2.0 * g0 + gp
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(np.abs(denom) > 1e-12, 0.5 * (gm - gp) / denom, 0.0)
    delta = np.where(np.isfinite(delta), delta, 0.0)
    delta = np.where(np.abs(delta) <= 0.5, delta, 0.0)
    delta = np.where(denom < 0, delta, 0.0)  # a maximum, not a minimum

    offset = np.where(valid, steps[peak] + delta, 0.0)
    return list(zip(xs + offset * nx, ys + offset * ny))


# --------------------------------------------------------------------------
# Joining fitted primitives exactly
# --------------------------------------------------------------------------


def _line_line_intersection(p0, d0, p1, d1) -> Optional[Point2]:
    denom = d0[0] * d1[1] - d0[1] * d1[0]
    if abs(denom) < 1e-12:
        return None  # parallel
    t = ((p1[0] - p0[0]) * d1[1] - (p1[1] - p0[1]) * d1[0]) / denom
    return (p0[0] + t * d0[0], p0[1] + t * d0[1])


def _line_circle_intersection(p, d, center, radius, near) -> Optional[Point2]:
    fx, fy = p[0] - center[0], p[1] - center[1]
    a = d[0] * d[0] + d[1] * d[1]
    b = 2.0 * (fx * d[0] + fy * d[1])
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0 or a == 0.0:
        return None
    root = math.sqrt(disc)
    candidates = [(-b + root) / (2 * a), (-b - root) / (2 * a)]
    points = [(p[0] + t * d[0], p[1] + t * d[1]) for t in candidates]
    return min(points, key=lambda q: math.hypot(q[0] - near[0], q[1] - near[1]))


def _circle_circle_intersection(c0, r0, c1, r1, near) -> Optional[Point2]:
    dx, dy = c1[0] - c0[0], c1[1] - c0[1]
    d = math.hypot(dx, dy)
    if d < 1e-12 or d > r0 + r1 or d < abs(r0 - r1):
        return None
    a = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    h_sq = r0 * r0 - a * a
    if h_sq < 0.0:
        return None
    h = math.sqrt(h_sq)
    mx, my = c0[0] + a * dx / d, c0[1] + a * dy / d
    points = [(mx + h * dy / d, my - h * dx / d), (mx - h * dy / d, my + h * dx / d)]
    return min(points, key=lambda q: math.hypot(q[0] - near[0], q[1] - near[1]))


def _joint(prev: dict, nxt: dict, nominal: Point2) -> Point2:
    """Where two fitted primitives truly meet.

    Averaging their loose endpoints would put the joint on neither curve.
    Intersecting them puts it on both, which is what lets each primitive
    keep its own fitted geometry exactly -- the same thing a CAD system
    does when trimming a fillet against an edge. The root nearest the
    nominal meeting point is chosen, so the far intersection of two
    circles never gets picked by mistake.
    """
    if prev["kind"] == "line" and nxt["kind"] == "line":
        hit = _line_line_intersection(prev["point"], prev["direction"], nxt["point"], nxt["direction"])
    elif prev["kind"] == "line" and nxt["kind"] == "arc":
        hit = _line_circle_intersection(prev["point"], prev["direction"], nxt["center"], nxt["radius"], nominal)
    elif prev["kind"] == "arc" and nxt["kind"] == "line":
        hit = _line_circle_intersection(nxt["point"], nxt["direction"], prev["center"], prev["radius"], nominal)
    else:
        hit = _circle_circle_intersection(
            prev["center"], prev["radius"], nxt["center"], nxt["radius"], nominal
        )

    if hit is None:
        return nominal
    # A wild intersection (near-parallel lines) is worse than the nominal.
    if math.hypot(hit[0] - nominal[0], hit[1] - nominal[1]) > 10.0 * max(1.0, abs(nominal[0]) * 1e-6):
        span = math.hypot(hit[0] - nominal[0], hit[1] - nominal[1])
        if span > 50.0:
            return nominal
    return hit


def _project_to_circle(point: Point2, center: Point2, radius: float) -> float:
    return math.atan2(point[1] - center[1], point[0] - center[0])


def _build_curve(span: dict, start: Point2, end: Point2, sole: bool = False) -> Optional[Curve2D]:
    if span["kind"] == "line":
        if math.hypot(end[0] - start[0], end[1] - start[1]) < 1e-12:
            return None
        return Line2D(start, end)

    center = span["center"]
    radius = span["radius"]
    ccw = span.get("ccw", True)

    # A span that is the entire chain closes on itself, so its start and
    # end coincide. That is a complete circle -- a drilled hole, the common
    # case for a raster drawing -- not a degenerate zero-angle arc, and
    # dropping it would silently discard the feature.
    if sole or span.get("closed_circle"):
        return Arc2D.full_circle(center, radius, ccw)

    a0 = _project_to_circle(start, center, radius)
    a1 = _project_to_circle(end, center, radius)
    if abs(a1 - a0) < 1e-12:
        return None
    return Arc2D(center, radius, a0, a1, ccw)


def fit_chain(
    points: Sequence[Point2],
    tolerance: float,
    min_arc_points: int = 6,
    closed: bool = True,
) -> Tuple[List[Curve2D], List[FitReport]]:
    """Segment and fit a point chain into exact line and arc primitives.

    Consecutive primitives are joined at their true intersection, so the
    result is a continuous chain suitable for :class:`~blueprint23d.brep.Loop2D`.
    """
    if _fitkernels is None:
        raise RuntimeError("the compiled fitting kernel is not available")
    if len(points) < 4:
        raise ValueError("need at least four points to fit a chain")

    spans = _fitkernels.segment(list(points), tolerance, min_arc_points)
    if not spans:
        return [], []
    spans = _merge_compatible_spans(list(points), spans, tolerance)

    # Record traversal direction for arcs from the source points, since the
    # fitted centre alone does not say which way round the arc was drawn.
    for span in spans:
        if span["kind"] == "arc":
            cx, cy = span["center"]
            i0, i1 = span["start"], span["end"]
            mid = (i0 + i1) // 2
            a0 = math.atan2(points[i0][1] - cy, points[i0][0] - cx)
            am = math.atan2(points[mid][1] - cy, points[mid][0] - cx)
            a1 = math.atan2(points[i1][1] - cy, points[i1][0] - cx)
            span["ccw"] = _sweeps_ccw(a0, am, a1)

    nominal = []
    count = len(spans)
    for i in range(count):
        end_index = spans[i]["end"]
        nominal.append(points[end_index % len(points)])

    joints: List[Point2] = []
    for i in range(count):
        nxt = spans[(i + 1) % count]
        if i == count - 1 and not closed:
            joints.append(nominal[i])
        else:
            joints.append(_joint(spans[i], nxt, nominal[i]))

    curves: List[Curve2D] = []
    reports: List[FitReport] = []
    for i, span in enumerate(spans):
        start = joints[i - 1] if (closed or i > 0) else points[span["start"]]
        end = joints[i]
        curve = _build_curve(span, start, end, sole=(len(spans) == 1 and closed))
        if curve is None:
            continue
        curves.append(curve)
        reports.append(
            FitReport(
                kind=span["kind"],
                points=span["end"] - span["start"] + 1,
                rms=span["rms"],
                max_deviation=span["max_deviation"],
                sagitta=span.get("sagitta"),
                radius=span.get("radius"),
            )
        )
    return curves, reports


def _merge_compatible_spans(points: List[Point2], spans: List[dict], tolerance: float) -> List[dict]:
    """Rejoin spans the greedy pass split for no geometric reason.

    Segmentation starts wherever the contour happens to begin, which is
    usually part-way along a feature, so a single fillet often arrives as
    two arcs that agree on centre and radius. Refitting the union and
    keeping it when the residual is still inside tolerance recovers the
    feature as one primitive -- which matters because the merged fit also
    spans more points, and precision improves with the square root of the
    count.
    """
    merged = list(spans)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i in range(len(merged) - 1):
            a, b = merged[i], merged[i + 1]
            if a["kind"] != b["kind"]:
                continue
            window = points[a["start"] : b["end"] + 1]
            if len(window) < 3:
                continue

            if a["kind"] == "arc":
                # Only attempt the merge when the two fits already agree;
                # otherwise this is a genuine corner between two features.
                radius_gap = abs(a["radius"] - b["radius"])
                centre_gap = math.hypot(a["center"][0] - b["center"][0], a["center"][1] - b["center"][1])
                scale = max(a["radius"], b["radius"], 1e-9)
                if radius_gap > 0.15 * scale or centre_gap > 0.15 * scale:
                    continue
                fit = _fitkernels.fit_circle(window)
                if fit is None or fit["max_deviation"] > tolerance:
                    continue
                combined = dict(a)
                combined.update(
                    end=b["end"], center=fit["center"], radius=fit["radius"], rms=fit["rms"],
                    max_deviation=fit["max_deviation"], span=fit["span"], sagitta=fit["sagitta"],
                )
            else:
                dot = a["direction"][0] * b["direction"][0] + a["direction"][1] * b["direction"][1]
                if abs(dot) < 0.999:  # more than ~2.5 degrees apart: a real corner
                    continue
                fit = _fitkernels.fit_line(window)
                if fit is None or fit["max_deviation"] > tolerance:
                    continue
                combined = dict(a)
                combined.update(
                    end=b["end"], point=fit["point"], direction=fit["direction"], rms=fit["rms"],
                    max_deviation=fit["max_deviation"],
                )

            merged[i : i + 2] = [combined]
            changed = True
            break
    return merged


def _sweeps_ccw(a0: float, am: float, a1: float) -> bool:
    """Does going a0 -> am -> a1 turn counter-clockwise?"""
    def forward(x: float, y: float) -> float:
        d = y - x
        while d < 0.0:
            d += 2.0 * math.pi
        return d

    return forward(a0, am) <= forward(a0, a1)


def loop_from_points(
    points: Sequence[Point2],
    tolerance: float,
    min_arc_points: int = 6,
    join_tolerance: float = 1e-6,
) -> Tuple[Loop2D, List[FitReport]]:
    """Fit a closed chain and assemble it into a validated :class:`Loop2D`.

    Any residual gap between consecutive fitted primitives is bridged with
    a short line rather than silently widening the loop's join tolerance --
    an explicit tiny edge is honest about what happened, where a loosened
    tolerance would hide it.
    """
    curves, reports = fit_chain(points, tolerance, min_arc_points, closed=True)
    if not curves:
        raise ValueError("no primitives could be fitted to the chain")

    bridged: List[Curve2D] = []
    for i, curve in enumerate(curves):
        bridged.append(curve)
        nxt = curves[(i + 1) % len(curves)]
        gap = math.hypot(curve.end[0] - nxt.start[0], curve.end[1] - nxt.start[1])
        if gap > join_tolerance:
            bridged.append(Line2D(curve.end, nxt.start))

    return Loop2D(bridged, tolerance=max(join_tolerance, 1e-9)), reports
