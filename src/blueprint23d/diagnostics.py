"""Structured defects: what is wrong, where, how big, and what to do.

A geometry tool that reports ``ValueError: invalid geometry`` has told the
shop foreman nothing. They need to know *which* contour is open, *where*,
by *how much*, and whether it matters for the process they are about to
run. That is the difference between a file someone can fix in thirty
seconds and a file that goes back to the customer with a shrug.

So every check in the package reports through this module rather than
raising, and a defect carries four things:

* a **stable code** that callers can branch on and that never gets
  renumbered, so an integration built against ``OPEN_CONTOUR`` keeps
  working;
* a **location** in drawing coordinates, plus the entity handles and layer
  it came from, so the defect can be found in the original file;
* **measurements** -- the actual numbers, in drawing units, because "gap of
  0.13 mm" and "gap of 13 mm" are completely different conversations;
* a **remedy**, written for whoever has to fix it.

Severity is deliberately separate from code. The same
``FEATURE_BELOW_KERF`` is fatal on a plasma table and irrelevant on a
waterjet, so severity is assigned by the check that raises it, in the
context of the process being run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

Point2 = Tuple[float, float]


class Severity(str, Enum):
    """How much this defect matters. Ordered."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "error": 2, "critical": 3}[self.value]

    def __lt__(self, other: "Severity") -> bool:  # type: ignore[override]
        return self.rank < other.rank


class Code(str, Enum):
    """Stable defect identifiers.

    These are API. Add freely; never repurpose or renumber, because
    somebody's integration is branching on the string.
    """

    # --- file and parsing -------------------------------------------
    FILE_UNREADABLE = "FILE_UNREADABLE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    ENTITY_LIMIT_EXCEEDED = "ENTITY_LIMIT_EXCEEDED"
    UNSUPPORTED_ENTITY = "UNSUPPORTED_ENTITY"
    CONVERTER_UNAVAILABLE = "CONVERTER_UNAVAILABLE"
    CONVERSION_FAILED = "CONVERSION_FAILED"
    CONVERSION_WARNING = "CONVERSION_WARNING"
    CONVERSION_METADATA_LOST = "CONVERSION_METADATA_LOST"
    COORDINATE_OUT_OF_RANGE = "COORDINATE_OUT_OF_RANGE"
    NON_FINITE_COORDINATE = "NON_FINITE_COORDINATE"
    PARSE_TIMEOUT = "PARSE_TIMEOUT"

    # --- units -------------------------------------------------------
    UNITS_UNDECLARED = "UNITS_UNDECLARED"
    UNITS_AMBIGUOUS = "UNITS_AMBIGUOUS"
    UNITS_CONTRADICTED = "UNITS_CONTRADICTED"

    # --- topology ----------------------------------------------------
    OPEN_CONTOUR = "OPEN_CONTOUR"
    SELF_INTERSECTION = "SELF_INTERSECTION"
    DUPLICATE_EDGE = "DUPLICATE_EDGE"
    OVERLAPPING_EDGE = "OVERLAPPING_EDGE"
    ZERO_LENGTH_EDGE = "ZERO_LENGTH_EDGE"
    DEGENERATE_LOOP = "DEGENERATE_LOOP"
    NO_CLOSED_PROFILE = "NO_CLOSED_PROFILE"
    NESTING_AMBIGUOUS = "NESTING_AMBIGUOUS"

    # --- manufacturability -------------------------------------------
    FEATURE_BELOW_KERF = "FEATURE_BELOW_KERF"
    HOLE_TOO_SMALL_FOR_THICKNESS = "HOLE_TOO_SMALL_FOR_THICKNESS"
    BRIDGE_TOO_NARROW = "BRIDGE_TOO_NARROW"
    INTERNAL_CORNER_TOO_SHARP = "INTERNAL_CORNER_TOO_SHARP"
    PART_EXCEEDS_SHEET = "PART_EXCEEDS_SHEET"
    BEND_RELIEF_MISSING = "BEND_RELIEF_MISSING"
    BEND_RADIUS_TOO_SMALL = "BEND_RADIUS_TOO_SMALL"

    # --- drawing consistency -----------------------------------------
    DIMENSION_MISMATCH = "DIMENSION_MISMATCH"
    DIMENSION_UNMATCHED = "DIMENSION_UNMATCHED"
    ANNOTATION_ON_GEOMETRY_LAYER = "ANNOTATION_ON_GEOMETRY_LAYER"

    # --- revision comparison -----------------------------------------
    FEATURE_ADDED = "FEATURE_ADDED"
    FEATURE_REMOVED = "FEATURE_REMOVED"
    FEATURE_CHANGED = "FEATURE_CHANGED"
    REVISION_AMBIGUOUS = "REVISION_AMBIGUOUS"
    REVISION_UNDOCUMENTED = "REVISION_UNDOCUMENTED"

    # --- reconstruction confidence -----------------------------------
    FEATURE_UNRESOLVABLE = "FEATURE_UNRESOLVABLE"
    FIT_RESIDUAL_HIGH = "FIT_RESIDUAL_HIGH"
    APPROXIMATION_USED = "APPROXIMATION_USED"


#: Human-readable remedies, keyed by code. Written for the person holding
#: the file, not for the developer who wrote the check.
REMEDIES: Dict[Code, str] = {
    Code.OPEN_CONTOUR: "Close the gap in your CAD package, or raise the weld tolerance if the gap is smaller than your drawing precision.",
    Code.SELF_INTERSECTION: "Remove the crossing so the outline traces a single closed path; cutters cannot follow a figure-eight.",
    Code.DUPLICATE_EDGE: "Delete the duplicate. Overlapping geometry makes the machine cut the same line twice, burning the edge and wasting time.",
    Code.OVERLAPPING_EDGE: "Trim the overlap so each length of the outline is described exactly once.",
    Code.ZERO_LENGTH_EDGE: "Delete the zero-length entity; it usually comes from a double-clicked vertex.",
    Code.UNITS_UNDECLARED: "Set the drawing units in your CAD package before exporting, or state them when submitting the file.",
    Code.UNITS_AMBIGUOUS: "Confirm the units. Getting this wrong scales the part by 25.4.",
    Code.UNITS_CONTRADICTED: "The header units disagree with the dimensions written on the drawing. Check which is right before cutting.",
    Code.FEATURE_BELOW_KERF: "Enlarge the feature or choose a process with a finer kerf; as drawn the cutter cannot fit.",
    Code.HOLE_TOO_SMALL_FOR_THICKNESS: "Enlarge the hole or drill it as a secondary operation; thermal processes cannot cut a hole much smaller than the material is thick.",
    Code.BRIDGE_TOO_NARROW: "Widen the web between these features, or it will burn through and drop out.",
    Code.INTERNAL_CORNER_TOO_SHARP: "Add a corner radius at least the tool radius, or the corner will come out rounded anyway and out of tolerance.",
    Code.PART_EXCEEDS_SHEET: "Reduce the part or nest it on a larger sheet.",
    Code.DIMENSION_MISMATCH: "The drawing text and the drawn geometry disagree. Establish which is correct before anything is cut to it.",
    Code.FEATURE_UNRESOLVABLE: "This feature could not be measured reliably from the source. Supply a vector drawing, or confirm the dimension by hand.",
    Code.NO_CLOSED_PROFILE: "No closed outline was found. Check the layer, and that the profile actually closes.",
    Code.CONVERTER_UNAVAILABLE: "Install a DWG converter: LibreDWG (dwg2dxf) or the ODA File Converter, and make sure it is on PATH.",
    Code.CONVERSION_FAILED: "The DWG could not be converted. Try re-saving it from your CAD package, or export DXF directly.",
    Code.CONVERSION_METADATA_LOST: "Layer names did not survive DWG conversion, so layer filters will not match. Select geometry another way, or supply DXF directly.",
    Code.FEATURE_ADDED: "This feature is new in the later revision. Confirm it was intended before re-running an existing program.",
    Code.FEATURE_REMOVED: "This feature is gone in the later revision. Any tooling, fixture, or program step for it is now stale.",
    Code.FEATURE_CHANGED: "This feature moved or changed size between revisions. Re-check the program, not just the drawing.",
    Code.REVISION_AMBIGUOUS: "Two features sit close enough that it is not certain which one became which. Confirm by hand before trusting the comparison.",
    Code.REVISION_UNDOCUMENTED: "Geometry changed but the revision note does not say so. Ask the customer which change is authoritative.",
}


@dataclass(frozen=True)
class Location:
    """Where a defect is, in both drawing space and the source file."""

    point: Optional[Point2] = None
    entity_handles: Tuple[str, ...] = ()
    layer: Optional[str] = None
    #: A second point, for defects that span (an overlap, a bridge).
    other_point: Optional[Point2] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.point is not None:
            data["point"] = [self.point[0], self.point[1]]
        if self.other_point is not None:
            data["other_point"] = [self.other_point[0], self.other_point[1]]
        if self.entity_handles:
            data["entity_handles"] = list(self.entity_handles)
        if self.layer is not None:
            data["layer"] = self.layer
        return data

    def describe(self) -> str:
        parts = []
        if self.point is not None:
            parts.append(f"({self.point[0]:.3f}, {self.point[1]:.3f})")
        if self.layer:
            parts.append(f"layer {self.layer}")
        if self.entity_handles:
            parts.append("entity " + ", ".join(f"#{h}" for h in self.entity_handles))
        return " ".join(parts)


@dataclass(frozen=True)
class Defect:
    """One thing wrong with a drawing."""

    code: Code
    severity: Severity
    message: str
    location: Location = field(default_factory=Location)
    #: Actual numbers in drawing units -- gap size, radius, angle.
    measurements: Dict[str, float] = field(default_factory=dict)
    remedy: Optional[str] = None

    def __post_init__(self):
        if self.remedy is None:
            object.__setattr__(self, "remedy", REMEDIES.get(self.code))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
        }
        location = self.location.to_dict()
        if location:
            data["location"] = location
        if self.measurements:
            data["measurements"] = dict(self.measurements)
        if self.remedy:
            data["remedy"] = self.remedy
        return data

    def describe(self) -> str:
        where = self.location.describe()
        head = f"[{self.severity.value.upper()}] {self.code.value}"
        return f"{head}: {self.message}" + (f"  at {where}" if where else "")


@dataclass
class Report:
    """Everything found about one drawing.

    Ordered by severity then by position, so the worst problem is the
    first thing anyone reads.
    """

    defects: List[Defect] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

    def add(
        self,
        code: Code,
        severity: Severity,
        message: str,
        location: Optional[Location] = None,
        **measurements: float,
    ) -> Defect:
        defect = Defect(
            code=code,
            severity=severity,
            message=message,
            location=location or Location(),
            measurements=measurements,
        )
        self.defects.append(defect)
        return defect

    def extend(self, defects: Iterable[Defect]) -> None:
        self.defects.extend(defects)

    # ---- querying ------------------------------------------------------

    def of(self, *codes: Code) -> List[Defect]:
        wanted = set(codes)
        return [d for d in self.defects if d.code in wanted]

    def at_least(self, severity: Severity) -> List[Defect]:
        return [d for d in self.defects if d.severity.rank >= severity.rank]

    @property
    def worst(self) -> Optional[Severity]:
        if not self.defects:
            return None
        return max((d.severity for d in self.defects), key=lambda s: s.rank)

    @property
    def cuttable(self) -> bool:
        """Whether anything blocking was found.

        ``ERROR`` and above mean the file cannot be run as supplied.
        Warnings are judgement calls for the operator.
        """
        return not self.at_least(Severity.ERROR)

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for defect in self.defects:
            out[defect.severity.value] = out.get(defect.severity.value, 0) + 1
        return out

    # ---- output --------------------------------------------------------

    def sorted_defects(self) -> List[Defect]:
        def key(defect: Defect):
            point = defect.location.point or (0.0, 0.0)
            return (-defect.severity.rank, defect.code.value, point[0], point[1])

        return sorted(self.defects, key=key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "cuttable": self.cuttable,
            "worst_severity": self.worst.value if self.worst else None,
            "counts": self.counts(),
            "metrics": dict(self.metrics),
            "defects": [d.to_dict() for d in self.sorted_defects()],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False, default=_fallback)

    def describe(self, limit: Optional[int] = None) -> str:
        lines = []
        verdict = "READY TO CUT" if self.cuttable else "NOT CUTTABLE AS SUPPLIED"
        lines.append(f"{self.source or 'drawing'}: {verdict}")
        if self.defects:
            counts = self.counts()
            lines.append(
                "  " + ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
            )
        shown = self.sorted_defects()
        if limit is not None:
            shown = shown[:limit]
        for defect in shown:
            lines.append("  " + defect.describe())
        remaining = len(self.defects) - len(shown)
        if remaining > 0:
            lines.append(f"  ... and {remaining} more")
        return "\n".join(lines)


def _fallback(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return str(value)


def merge(*reports: Report) -> Report:
    """Combine reports, keeping metrics from all of them."""
    combined = Report(source=next((r.source for r in reports if r.source), None))
    for report in reports:
        combined.defects.extend(report.defects)
        combined.metrics.update(report.metrics)
    return combined
