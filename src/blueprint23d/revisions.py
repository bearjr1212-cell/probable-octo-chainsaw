"""What actually changed between two revisions of a drawing.

A shop receives rev C of a part it already cut as rev B. The revision note
says "updated per customer request". Nothing else on the drawing says
which of the fifty features moved, and the cloud bubbles — if there are
any — were drawn by hand and are not to be trusted. So somebody puts the
two prints side by side on the bench and looks, and the hole that moved
0.4 mm does not get noticed until the assembly does not go together.

This module answers the question the drawing refuses to: feature by
feature, what is new, what is gone, and what moved or changed size.

Three design commitments make the answer worth reading.

**Features are matched by position, never by size.** Matching a Ø6.35 hole
to the other drawing's Ø6.35 hole would assume exactly the thing being
tested and would be structurally incapable of reporting a diameter change.
Features are paired by where they sit; their sizes are then compared.

**A pairing that is not obvious is not made.** Two features are paired only
when they are each other's nearest candidate and closer together than the
features themselves are spaced. A hole that moved further than its
neighbours are apart is genuinely indistinguishable from one hole deleted
and another added, so it is reported that way rather than guessed at, and
near-calls raise :data:`~blueprint23d.diagnostics.Code.REVISION_AMBIGUOUS`.

**"Unchanged" means unchanged.** The geometry in this package is held
exactly — an arc is a centre, a radius and two angles, not a polyline — so
two features can be compared for genuine identity rather than for
agreement within an epsilon somebody picked. A feature reported unchanged
has a bit-identical description, up to where the loop happens to start and
which way round it runs. Everything else is reported with numbers.

That last point has a useful consequence: a circle that was exported as a
64-segment polyline is *not* reported as unchanged. It is reported as
redrawn, at the same place and the same size, which is precisely the
finding you want when the second file came out of a different system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .brep import Face2D, Loop2D
from .curves import Arc2D, Curve2D, Line2D
from .diagnostics import Code, Location, Report, Severity

Point2 = Tuple[float, float]

#: Differences below this are floating-point noise from a file round-trip,
#: not an edit. Geometry here is exact, so this is deliberately tight: a
#: real CAD edit is never this small.
DEFAULT_TOLERANCE = 1e-9

#: A second candidate within this ratio of the best one means the pairing
#: was a coin toss and the comparison should say so.
AMBIGUITY_RATIO = 2.0


# --------------------------------------------------------------------------
# Exact identity
# --------------------------------------------------------------------------


def _freeze(value: Any) -> Any:
    """Hashable, comparable, order-preserving image of a curve field."""
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def curve_signature(curve: Curve2D) -> Tuple[Any, ...]:
    """Everything that defines a curve, exactly as stored.

    Two curves with equal signatures are the same curve bit for bit. No
    tolerance enters here, which is what lets an unchanged feature be
    reported as a fact rather than as an estimate.
    """
    from dataclasses import fields as dataclass_fields

    if isinstance(curve, Arc2D) and abs(abs(curve.sweep) - 2.0 * math.pi) < 1e-9:
        # A closed circle is the same circle whatever angle it starts at, so
        # the phase is dropped. Otherwise re-exporting a hole would read as
        # a design change.
        return ("Arc2D", _freeze(curve.center), curve.radius, 0.0, 2.0 * math.pi, True)

    try:
        values = tuple(_freeze(getattr(curve, f.name)) for f in dataclass_fields(curve))
    except TypeError:  # not a dataclass; fall back on a stable sampling
        values = tuple(_freeze(curve.point(i / 32.0)) for i in range(33))
    return (type(curve).__name__,) + values


def loop_signature(loop: Loop2D) -> Tuple[Any, ...]:
    """A loop's identity, independent of where it starts and which way it runs.

    The same closed profile exported twice can begin at a different vertex
    and wind the other way. Neither is a change to the part, so the
    signature is canonicalised over both.
    """
    forward = [curve_signature(c) for c in loop.curves]
    backward = [curve_signature(c) for c in loop.reverse().curves]

    def canonical(keys: List[Tuple[Any, ...]]) -> Tuple[Any, ...]:
        n = len(keys)
        if n == 0:
            return ()
        rotations = [tuple(keys[i:] + keys[:i]) for i in range(n)]
        try:
            return min(rotations)
        except TypeError:  # heterogeneous keys that will not order
            return rotations[0]

    try:
        return min(canonical(forward), canonical(backward))
    except TypeError:
        return canonical(forward)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def _is_full_circle(loop: Loop2D) -> Optional[Arc2D]:
    if len(loop.curves) != 1:
        return None
    curve = loop.curves[0]
    if isinstance(curve, Arc2D) and abs(abs(curve.sweep) - 2.0 * math.pi) < 1e-9:
        return curve
    return None


def _classify(loop: Loop2D, outer: bool) -> str:
    if outer:
        return "outer profile"
    if _is_full_circle(loop) is not None:
        return "hole"
    arcs = [c for c in loop.curves if isinstance(c, Arc2D)]
    lines = [c for c in loop.curves if isinstance(c, Line2D)]
    if len(arcs) == 2 and len(lines) == 2 and len(loop.curves) == 4:
        if abs(arcs[0].radius - arcs[1].radius) < 1e-9:
            return "slot"
    if not arcs and len(loop.curves) == 4:
        return "rectangular cutout"
    return "cutout"


@dataclass(frozen=True)
class Feature:
    """One identifiable thing on a drawing: a profile or an opening in it.

    The anchor is the centre of the exact bounding box, which for a
    circular hole is the centre of the circle exactly, and for everything
    else is a position that does not depend on a tessellation tolerance.
    """

    role: str  # "profile" or "opening" -- only like roles are compared
    kind: str  # "hole", "slot", "outer profile", ...
    anchor: Point2
    area: float
    perimeter: float
    bounds: Tuple[float, float, float, float]
    signature: Tuple[Any, ...]
    diameter: Optional[float] = None
    edges: int = 0
    area_is_exact: bool = True
    face_index: int = 0

    @classmethod
    def from_loop(cls, loop: Loop2D, outer: bool, face_index: int = 0) -> "Feature":
        x0, y0, x1, y1 = loop.bounds()
        circle = _is_full_circle(loop)
        return cls(
            role="profile" if outer else "opening",
            kind=_classify(loop, outer),
            anchor=(circle.center if circle is not None else ((x0 + x1) / 2.0, (y0 + y1) / 2.0)),
            area=loop.area(),
            perimeter=loop.length(),
            bounds=(x0, y0, x1, y1),
            signature=loop_signature(loop),
            diameter=(2.0 * circle.radius) if circle is not None else None,
            edges=len(loop.curves),
            area_is_exact=loop.is_area_exact(),
            face_index=face_index,
        )

    @property
    def extent(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.bounds
        return (x1 - x0, y1 - y0)

    @property
    def size(self) -> float:
        """The one number that best describes how big this feature is."""
        return self.diameter if self.diameter is not None else self.area

    def describe_size(self) -> str:
        if self.diameter is not None:
            return f"Ø{self.diameter:g}"
        width, height = self.extent
        return f"{width:g}×{height:g}"

    def describe(self) -> str:
        return f"{self.kind} at ({self.anchor[0]:g}, {self.anchor[1]:g}) {self.describe_size()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "kind": self.kind,
            "anchor": [self.anchor[0], self.anchor[1]],
            "area": self.area,
            "perimeter": self.perimeter,
            "diameter": self.diameter,
            "extent": list(self.extent),
            "edges": self.edges,
        }


def extract_features(faces: Sequence[Face2D]) -> List[Feature]:
    """Every profile and opening in a drawing, in a deterministic order."""
    features: List[Feature] = []
    for index, face in enumerate(faces):
        features.append(Feature.from_loop(face.outer, outer=True, face_index=index))
        for loop in face.inners:
            features.append(Feature.from_loop(loop, outer=False, face_index=index))
    features.sort(key=lambda f: (f.role, f.anchor[0], f.anchor[1], f.kind))
    return features


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------


def _minimum_spacing(features: Sequence[Feature]) -> float:
    """Closest approach between two features of the same role.

    This sets the scale at which "it moved" stops being a better
    explanation than "one went and another arrived".
    """
    best = math.inf
    for i, a in enumerate(features):
        for b in features[i + 1 :]:
            if a.role == b.role:
                best = min(best, math.dist(a.anchor, b.anchor))
    return best


def pairing_tolerance(
    before: Sequence[Feature], after: Sequence[Feature]
) -> float:
    """How far a feature may move and still be recognised as itself.

    Half the tightest spacing between features of the same role in either
    drawing. Beyond that, a move is genuinely indistinguishable from a
    deletion plus an addition, and this module refuses to guess. With only
    one feature of a role there is nothing to confuse it with, so any
    distance is accepted.
    """
    spacing = min(_minimum_spacing(before), _minimum_spacing(after))
    if not math.isfinite(spacing):
        return math.inf
    return max(0.5 * spacing, DEFAULT_TOLERANCE)


@dataclass(frozen=True)
class Pairing:
    before_index: int
    after_index: int
    distance: float
    #: Distance to the runner-up, when one exists. A close second means the
    #: pairing was a judgement call.
    runner_up: float = math.inf

    @property
    def ambiguous(self) -> bool:
        return (
            math.isfinite(self.runner_up)
            and self.distance > DEFAULT_TOLERANCE
            and self.runner_up < AMBIGUITY_RATIO * self.distance
        )


def pair_features(
    before: Sequence[Feature],
    after: Sequence[Feature],
    tolerance: Optional[float] = None,
) -> Tuple[List[Pairing], List[int], List[int]]:
    """Match features between revisions by position alone.

    Returns the accepted pairings plus the indices left over on each side,
    which are the removals and the additions.
    """
    if tolerance is None:
        tolerance = pairing_tolerance(before, after)

    candidates: List[Tuple[float, int, int]] = []
    for i, a in enumerate(before):
        for j, b in enumerate(after):
            if a.role != b.role:
                continue
            distance = math.dist(a.anchor, b.anchor)
            if distance <= tolerance:
                candidates.append((distance, i, j))
    # Shortest first; ties broken by index so the result never depends on
    # dictionary or float ordering.
    candidates.sort()

    taken_before: Dict[int, int] = {}
    taken_after: Dict[int, int] = {}
    pairings: List[Pairing] = []
    for distance, i, j in candidates:
        if i in taken_before or j in taken_after:
            continue
        runner_up = min(
            (d for d, k, m in candidates if (k == i) != (m == j)),
            default=math.inf,
        )
        taken_before[i] = j
        taken_after[j] = i
        pairings.append(Pairing(i, j, distance, runner_up))

    removed = [i for i in range(len(before)) if i not in taken_before]
    added = [j for j in range(len(after)) if j not in taken_after]
    pairings.sort(key=lambda p: (p.before_index, p.after_index))
    return pairings, removed, added


# --------------------------------------------------------------------------
# The diff
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Change:
    """One feature's history across the revision."""

    status: str  # "added" | "removed" | "changed" | "unchanged"
    before: Optional[Feature] = None
    after: Optional[Feature] = None
    moved: float = 0.0
    size_delta: Optional[float] = None
    reasons: Tuple[str, ...] = ()
    ambiguous: bool = False

    @property
    def feature(self) -> Feature:
        """Whichever side exists, preferring the later revision."""
        result = self.after or self.before
        assert result is not None
        return result

    @property
    def anchor(self) -> Point2:
        return self.feature.anchor

    def describe(self) -> str:
        head = f"{self.feature.kind} ({self.anchor[0]:g}, {self.anchor[1]:g})"
        detail = ", ".join(self.reasons)
        return f"{head:<34}{detail:<34}{self.status}"

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "status": self.status,
            "kind": self.feature.kind,
            "anchor": [self.anchor[0], self.anchor[1]],
            "reasons": list(self.reasons),
        }
        if self.before is not None:
            data["before"] = self.before.to_dict()
        if self.after is not None:
            data["after"] = self.after.to_dict()
        if self.moved:
            data["moved"] = self.moved
        if self.size_delta is not None:
            data["size_delta"] = self.size_delta
        if self.ambiguous:
            data["ambiguous"] = True
        return data


def _compare(
    before: Feature, after: Feature, pairing: Pairing, tolerance: float
) -> Change:
    if before.signature == after.signature:
        return Change("unchanged", before, after, ambiguous=pairing.ambiguous)

    reasons: List[str] = []
    moved = math.dist(before.anchor, after.anchor)
    if moved > tolerance:
        dx = after.anchor[0] - before.anchor[0]
        dy = after.anchor[1] - before.anchor[1]
        reasons.append(f"moved {moved:g} ({dx:+g}, {dy:+g})")

    size_delta: Optional[float] = None
    if before.diameter is not None and after.diameter is not None:
        if abs(after.diameter - before.diameter) > tolerance:
            size_delta = after.diameter - before.diameter
            reasons.append(f"Ø{before.diameter:g} → Ø{after.diameter:g}")
    else:
        if abs(after.area - before.area) > tolerance:
            size_delta = after.area - before.area
            reasons.append(f"{before.describe_size()} → {after.describe_size()}")

    if before.kind != after.kind:
        reasons.append(f"{before.kind} → {after.kind}")
    if before.edges != after.edges:
        # How an exact arc becomes a polyline, and a cutting program
        # silently gains a thousand moves where it had one.
        reasons.append(f"{before.edges} → {after.edges} edges")
    if not reasons:
        # Same place, same size, same construction, different description:
        # the profile was rebuilt out of different curves.
        reasons.append(f"redrawn ({before.edges} edges, same size and position)")

    return Change(
        "changed",
        before,
        after,
        moved=moved,
        size_delta=size_delta,
        reasons=tuple(reasons),
        ambiguous=pairing.ambiguous,
    )


@dataclass
class Diff:
    """The whole comparison: every feature, and what became of it."""

    changes: List[Change] = field(default_factory=list)
    report: Report = field(default_factory=Report)
    before_label: str = "before"
    after_label: str = "after"
    tolerance: float = math.inf

    def of(self, *statuses: str) -> List[Change]:
        wanted = set(statuses)
        return [c for c in self.changes if c.status in wanted]

    @property
    def added(self) -> List[Change]:
        return self.of("added")

    @property
    def removed(self) -> List[Change]:
        return self.of("removed")

    @property
    def changed(self) -> List[Change]:
        return self.of("changed")

    @property
    def unchanged(self) -> List[Change]:
        return self.of("unchanged")

    @property
    def identical(self) -> bool:
        """True when the two revisions describe the same part exactly."""
        return all(c.status == "unchanged" for c in self.changes)

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for change in self.changes:
            out[change.status] = out.get(change.status, 0) + 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "before": self.before_label,
            "after": self.after_label,
            "identical": self.identical,
            "counts": self.counts(),
            "pairing_tolerance": (
                self.tolerance if math.isfinite(self.tolerance) else None
            ),
            "changes": [c.to_dict() for c in self.changes],
        }

    def describe(self, include_unchanged: bool = True) -> str:
        lines = [f"{self.before_label} → {self.after_label}"]
        if self.identical:
            lines.append("  identical: every feature matches exactly")
            return "\n".join(lines)
        for change in self.changes:
            if change.status == "unchanged" and not include_unchanged:
                continue
            lines.append("  " + change.describe().rstrip())
            if change.ambiguous:
                lines.append("    (pairing uncertain: another feature is nearly as close)")
        return "\n".join(lines)


def _order(change: Change) -> Tuple[int, float, float]:
    rank = {"changed": 0, "added": 1, "removed": 2, "unchanged": 3}
    return (rank.get(change.status, 4), change.anchor[0], change.anchor[1])


def diff_features(
    before: Sequence[Feature],
    after: Sequence[Feature],
    tolerance: float = DEFAULT_TOLERANCE,
    locate_tolerance: Optional[float] = None,
    before_label: str = "before",
    after_label: str = "after",
    report: Optional[Report] = None,
) -> Diff:
    """Compare two extracted feature sets."""
    report = report or Report()
    if locate_tolerance is None:
        locate_tolerance = pairing_tolerance(before, after)

    pairings, removed, added = pair_features(before, after, locate_tolerance)
    changes: List[Change] = []

    for pairing in pairings:
        changes.append(
            _compare(before[pairing.before_index], after[pairing.after_index], pairing, tolerance)
        )
    for index in removed:
        changes.append(Change("removed", before=before[index], reasons=("gone",)))
    for index in added:
        changes.append(Change("added", after=after[index], reasons=("new",)))

    changes.sort(key=_order)

    for change in changes:
        location = Location(point=change.anchor)
        if change.status == "changed":
            report.add(
                Code.FEATURE_CHANGED,
                Severity.WARNING,
                f"{change.feature.kind} at ({change.anchor[0]:g}, {change.anchor[1]:g}): "
                + "; ".join(change.reasons),
                location,
                **_measurements(change),
            )
        elif change.status == "added":
            report.add(
                Code.FEATURE_ADDED,
                Severity.WARNING,
                f"{change.feature.kind} {change.feature.describe_size()} at "
                f"({change.anchor[0]:g}, {change.anchor[1]:g}) is new in {after_label}",
                location,
            )
        elif change.status == "removed":
            report.add(
                Code.FEATURE_REMOVED,
                Severity.WARNING,
                f"{change.feature.kind} {change.feature.describe_size()} at "
                f"({change.anchor[0]:g}, {change.anchor[1]:g}) is gone in {after_label}",
                location,
            )
        if change.ambiguous:
            report.add(
                Code.REVISION_AMBIGUOUS,
                Severity.WARNING,
                f"the {change.feature.kind} at ({change.anchor[0]:g}, "
                f"{change.anchor[1]:g}) was matched to a feature that another one "
                "sits nearly as close to, so this pairing is a judgement call",
                location,
            )

    report.metrics["features_before"] = len(before)
    report.metrics["features_after"] = len(after)
    for status, count in sorted(
        {c.status: sum(1 for x in changes if x.status == c.status) for c in changes}.items()
    ):
        report.metrics[f"features_{status}"] = count

    return Diff(
        changes=changes,
        report=report,
        before_label=before_label,
        after_label=after_label,
        tolerance=locate_tolerance,
    )


def _measurements(change: Change) -> Dict[str, float]:
    data: Dict[str, float] = {}
    if change.moved:
        data["moved"] = change.moved
    if change.size_delta is not None:
        data["size_delta"] = change.size_delta
    if change.before is not None and change.before.diameter is not None:
        data["diameter_before"] = change.before.diameter
    if change.after is not None and change.after.diameter is not None:
        data["diameter_after"] = change.after.diameter
    return data


def diff_faces(
    before: Sequence[Face2D],
    after: Sequence[Face2D],
    tolerance: float = DEFAULT_TOLERANCE,
    locate_tolerance: Optional[float] = None,
    before_label: str = "before",
    after_label: str = "after",
) -> Diff:
    """Compare two drawings that have already been loaded."""
    return diff_features(
        extract_features(before),
        extract_features(after),
        tolerance=tolerance,
        locate_tolerance=locate_tolerance,
        before_label=before_label,
        after_label=after_label,
    )


def diff_drawings(
    before_path: Union[str, Path],
    after_path: Union[str, Path],
    layer: Optional[str] = None,
    tolerance: float = DEFAULT_TOLERANCE,
    locate_tolerance: Optional[float] = None,
    revision_before: Optional[str] = None,
    revision_after: Optional[str] = None,
) -> Diff:
    """Compare two drawing files feature by feature.

    ``revision_before`` and ``revision_after`` are the revision letters as
    written on the two prints. If they are supplied and identical while the
    geometry is not, that is reported: two files claiming the same revision
    and describing different parts is the failure mode that gets the wrong
    one cut.
    """
    from .loaders import load_faces

    before_label = revision_before or Path(before_path).stem
    after_label = revision_after or Path(after_path).stem

    report = Report(source=f"{before_path} → {after_path}")
    try:
        before_faces, _ = load_faces(before_path, layer=layer)
        after_faces, _ = load_faces(after_path, layer=layer)
    except Exception as exc:
        report.add(Code.FILE_UNREADABLE, Severity.CRITICAL, str(exc))
        return Diff(report=report, before_label=before_label, after_label=after_label)

    diff = diff_features(
        extract_features(before_faces),
        extract_features(after_faces),
        tolerance=tolerance,
        locate_tolerance=locate_tolerance,
        before_label=before_label,
        after_label=after_label,
        report=report,
    )

    if (
        revision_before is not None
        and revision_after is not None
        and revision_before == revision_after
        and not diff.identical
    ):
        report.add(
            Code.REVISION_UNDOCUMENTED,
            Severity.ERROR,
            f"both files are marked revision {revision_after} but they describe "
            f"different geometry ({len(diff.changed)} changed, {len(diff.added)} added, "
            f"{len(diff.removed)} removed)",
        )

    return diff
