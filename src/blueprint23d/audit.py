"""Checking a drawing against itself.

A drawing makes two independent claims about a part: the geometry says one
thing, and the dimensions written on it say another. Inside a live CAD
model those cannot disagree, because the dimension is *derived* from the
geometry. But a file that arrives from a customer has usually lost that
association somewhere — exported through a different system, scaled, edited
by hand, or had its text overridden — and then the two claims drift apart
with nothing anywhere checking them.

That is the error class this module exists for, and it is the worst one to
have, because nothing looks wrong. The geometry is valid. The dimensions
are legible. Every individual step of the job is correct. The part simply
comes out the wrong size, and nobody finds out until it is measured.

Two checks, in increasing depth:

**The dimension against its own extension lines.** Text reading ``99.50``
on a dimension whose definition points span ``100.0`` is an override that
contradicts itself. Cheap, and it needs nothing but the dimension.

**The dimension against the real geometry.** The definition points are
snapped onto actual curves and the distance re-measured there, which
catches the case where the dimension is internally consistent but the
geometry moved underneath it. For radial dimensions this is the valuable
one: a hole called ``Ø6.35`` whose arc measures ``Ø6.30`` is a quarter-inch
drill that will not fit.

What is *not* claimed: this cannot tell you which of the two is right. It
tells you they disagree, by how much, and where — which is the whole job,
because a shop that knows a drawing is inconsistent will pick up the phone
instead of cutting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from .annotations import DIAMETER, Dimension, RADIUS, extract_dimensions
from .brep import Face2D
from .curves import Arc2D, Curve2D
from .diagnostics import Code, Location, Report, Severity

Point2 = Tuple[float, float]

#: A dimension whose definition points land further than this from any
#: geometry was probably not attached to the part at all.
DEFAULT_ATTACH_TOLERANCE = 1.0

#: Relative disagreement below this is drafting round-off, not a conflict.
DEFAULT_RELATIVE_TOLERANCE = 1e-4

#: Absolute floor, so tiny features are not flagged over nothing.
DEFAULT_ABSOLUTE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Finding:
    """One dimension, and whether the part agrees with it."""

    dimension: Dimension
    stated: Optional[float]
    measured: Optional[float]
    #: What the measurement was taken from: "annotation" or "geometry".
    basis: str
    matched: bool
    #: How far the definition points sat from real geometry, when known.
    detachment: Optional[float] = None

    @property
    def difference(self) -> Optional[float]:
        if self.stated is None or self.measured is None:
            return None
        return self.measured - self.stated

    @property
    def relative(self) -> Optional[float]:
        difference = self.difference
        if difference is None or not self.stated:
            return None
        return abs(difference) / abs(self.stated)

    def disagrees(
        self,
        relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
        absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    ) -> bool:
        difference = self.difference
        if difference is None:
            return False
        if abs(difference) <= absolute_tolerance:
            return False
        relative = self.relative
        return relative is not None and relative > relative_tolerance

    def describe(self) -> str:
        if self.stated is None:
            return f"{self.dimension.kind_name}: states nothing checkable"
        if self.measured is None:
            return f"{self.dimension.kind_name} {self.stated:g}: no geometry matched"
        return (
            f"{self.dimension.kind_name} called {self.stated:g}, "
            f"{self.basis} measures {self.measured:g} "
            f"(off by {self.difference:+.4g})"
        )


# --------------------------------------------------------------------------
# Matching dimensions to geometry
# --------------------------------------------------------------------------


def _all_curves(faces: Sequence[Face2D]) -> List[Curve2D]:
    curves: List[Curve2D] = []
    for face in faces:
        for loop in face.loops():
            curves.extend(loop.curves)
    return curves


def _distance_to_curve(point: Point2, curve: Curve2D, samples: int = 24) -> float:
    """Closest approach of a point to a curve, near enough for matching."""
    if isinstance(curve, Arc2D):
        # Exact for arcs: the distance to the circle, clamped to the sweep.
        to_centre = math.dist(point, curve.center)
        radial = abs(to_centre - curve.radius)
        # Also consider the endpoints, in case the point is off the sweep.
        return min(radial, math.dist(point, curve.start), math.dist(point, curve.end))
    best = math.inf
    for i in range(samples + 1):
        best = min(best, math.dist(point, curve.point(i / samples)))
    return best


def _nearest_distance(point: Point2, curves: Sequence[Curve2D]) -> float:
    """Distance from a point to the closest geometry, however far."""
    return min((_distance_to_curve(point, curve) for curve in curves), default=math.inf)


def snap_to_geometry(
    point: Point2,
    curves: Sequence[Curve2D],
    tolerance: float = DEFAULT_ATTACH_TOLERANCE,
    samples: int = 64,
) -> Optional[Point2]:
    """The nearest point on any curve, if the dimension attaches at all."""
    best_point: Optional[Point2] = None
    best_distance = tolerance

    for curve in curves:
        if isinstance(curve, Arc2D):
            to_centre = math.dist(point, curve.center)
            if to_centre > 1e-12:
                scale = curve.radius / to_centre
                candidate = (
                    curve.center[0] + (point[0] - curve.center[0]) * scale,
                    curve.center[1] + (point[1] - curve.center[1]) * scale,
                )
                distance = math.dist(point, candidate)
                if distance < best_distance:
                    best_point, best_distance = candidate, distance
        for i in range(samples + 1):
            candidate = curve.point(i / samples)
            distance = math.dist(point, candidate)
            if distance < best_distance:
                best_point, best_distance = candidate, distance

    return best_point


def match_radial(
    dimension: Dimension,
    curves: Sequence[Curve2D],
    tolerance: float = DEFAULT_ATTACH_TOLERANCE,
) -> Optional[Arc2D]:
    """Find the arc a radial dimension is pointing at.

    Matched by proximity of the dimension's reference point to the arc, not
    by radius — matching on radius would assume the answer and could never
    detect a disagreement.
    """
    anchor = dimension.anchor
    if anchor is None:
        return None

    best: Optional[Arc2D] = None
    best_distance = math.inf
    for curve in curves:
        if not isinstance(curve, Arc2D):
            continue
        distance = _distance_to_curve(anchor, curve)
        if distance < best_distance:
            best, best_distance = curve, distance

    # Radial dimensions often anchor at the centre rather than on the arc.
    if best is not None and best_distance <= tolerance:
        return best
    for curve in curves:
        if isinstance(curve, Arc2D) and math.dist(anchor, curve.center) <= tolerance:
            return curve
    return None


def audit_dimension(
    dimension: Dimension,
    curves: Sequence[Curve2D],
    attach_tolerance: float = DEFAULT_ATTACH_TOLERANCE,
) -> Finding:
    """Check one dimension against the part."""
    stated = dimension.stated_value

    if dimension.is_radial:
        arc = match_radial(dimension, curves, attach_tolerance)
        if arc is None:
            return Finding(dimension, stated, None, "geometry", matched=False)
        measured = arc.radius * 2.0 if dimension.kind == DIAMETER else arc.radius
        return Finding(dimension, stated, measured, "geometry", matched=True)

    points = dimension.reference_points
    if len(points) >= 2:
        snapped = [snap_to_geometry(p, curves, attach_tolerance) for p in points[:2]]
        if all(s is not None for s in snapped):
            measured = math.dist(snapped[0], snapped[1])  # type: ignore[arg-type]
            return Finding(dimension, stated, measured, "geometry", matched=True, detachment=0.0)

        # The dimension does not land on the part. That is itself the
        # finding: a dimension floating free of the geometry usually means
        # the geometry was edited and the dimension was left behind, which
        # is precisely the drawing that gets cut to a stale number. Measure
        # how far off it is rather than quietly trusting the annotation.
        detachment = max(
            (_nearest_distance(p, curves) for p in points[:2]),
            default=math.inf,
        )
        return Finding(
            dimension,
            stated,
            dimension.measurement,
            "annotation",
            matched=False,
            detachment=detachment,
        )

    # Nothing to snap to: fall back on the dimension's own extension lines,
    # which still catches an override that contradicts itself.
    return Finding(dimension, stated, dimension.measurement, "annotation", matched=False)


def audit(
    faces: Sequence[Face2D],
    dimensions: Sequence[Dimension],
    attach_tolerance: float = DEFAULT_ATTACH_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    report: Optional[Report] = None,
) -> Tuple[List[Finding], Report]:
    """Check every dimension against the geometry."""
    report = report or Report()
    curves = _all_curves(faces)
    findings: List[Finding] = []

    for dimension in dimensions:
        finding = audit_dimension(dimension, curves, attach_tolerance)
        findings.append(finding)

        if finding.stated is None:
            continue

        if (
            finding.detachment is not None
            and math.isfinite(finding.detachment)
            and finding.detachment > attach_tolerance
        ):
            report.add(
                Code.DIMENSION_UNMATCHED,
                Severity.WARNING,
                f"a {dimension.kind_name} dimension of {finding.stated:g} does not "
                f"attach to the part: its extension lines start {finding.detachment:.4g} "
                "from any geometry, so it may have been left behind by an edit",
                Location(point=dimension.anchor, layer=dimension.layer,
                         entity_handles=(dimension.handle,) if dimension.handle else ()),
                stated=finding.stated,
                detachment=finding.detachment,
            )
            continue

        if finding.measured is None:
            report.add(
                Code.DIMENSION_UNMATCHED,
                Severity.INFO,
                f"a {dimension.kind_name} dimension of {finding.stated:g} could not be "
                "matched to any geometry",
                Location(point=dimension.anchor, layer=dimension.layer,
                         entity_handles=(dimension.handle,) if dimension.handle else ()),
                stated=finding.stated,
            )
            continue

        if finding.disagrees(relative_tolerance, absolute_tolerance):
            difference = finding.difference or 0.0
            noun = "diameter" if dimension.kind == DIAMETER else (
                "radius" if dimension.kind == RADIUS else "distance"
            )
            source = "the geometry" if finding.basis == "geometry" else "its own extension lines"
            report.add(
                Code.DIMENSION_MISMATCH,
                Severity.ERROR,
                f"drawing calls this {noun} {finding.stated:g} but {source} "
                f"measures {finding.measured:g} (off by {difference:+.4g})",
                Location(point=dimension.anchor, layer=dimension.layer,
                         entity_handles=(dimension.handle,) if dimension.handle else ()),
                stated=finding.stated,
                measured=finding.measured,
                difference=difference,
            )

    checked = sum(1 for f in findings if f.stated is not None and f.measured is not None)
    report.metrics["dimensions"] = len(dimensions)
    report.metrics["dimensions_checked"] = checked
    report.metrics["dimensions_matched_to_geometry"] = sum(1 for f in findings if f.matched)
    return findings, report


def audit_drawing(
    path: Union[str, Path],
    layer: Optional[str] = None,
    attach_tolerance: float = DEFAULT_ATTACH_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> Tuple[List[Finding], Report]:
    """Read a DXF and check what it says against what it draws."""
    import ezdxf

    from .loaders import load_faces

    report = Report(source=str(path))
    try:
        faces, _ = load_faces(path, layer=layer)
        dimensions = extract_dimensions(ezdxf.readfile(str(path)).modelspace())
    except Exception as exc:
        report.add(Code.FILE_UNREADABLE, Severity.CRITICAL, str(exc))
        return [], report

    if not dimensions:
        report.add(
            Code.DIMENSION_UNMATCHED,
            Severity.INFO,
            "the drawing carries no dimensions, so nothing could be cross-checked",
        )
        return [], report

    return audit(
        faces,
        dimensions,
        attach_tolerance=attach_tolerance,
        relative_tolerance=relative_tolerance,
        report=report,
    )


def describe_findings(findings: Sequence[Finding]) -> str:
    """A dimension-by-dimension table, for a human."""
    if not findings:
        return "no dimensions to check"
    lines = []
    for finding in findings:
        marker = "  " if not finding.disagrees() else "! "
        lines.append(marker + finding.describe())
    return "\n".join(lines)
