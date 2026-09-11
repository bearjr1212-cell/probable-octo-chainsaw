"""Working out what a drawing's numbers actually mean.

A DXF is a pile of bare numbers. Whether ``100.0`` means 100 mm or 100
inches is carried, if at all, in a header field that plenty of exporters
never set. Get it wrong and every dimension is off by a factor of 25.4 —
which is not a subtle error but a scrapped sheet, and it is one of the most
common ways a cutting job goes wrong.

Nothing decides this reliably on its own, so four independent lines of
evidence are gathered and weighed, and the answer always arrives with its
reasoning attached rather than as a bare assertion:

1. **The declared header value** (``$INSUNITS``). Authoritative when
   present and trusted accordingly — but it is ``0`` (unitless) far more
   often than anyone would like.

2. **Dimension text against measured geometry.** This is the strong one,
   and as far as I can tell nobody else does it. If a dimension's text
   reads ``4.00`` while the geometry it spans measures ``101.6`` drawing
   units, the ratio is 25.4: the part is *drawn* in millimetres and
   *dimensioned* in inches. That is a fact about the file, not a guess,
   and it is exactly the sort of mixed-unit drawing that causes expensive
   mistakes.

3. **Physical plausibility.** A cut part is usually somewhere between a
   few millimetres and a few metres. A profile measuring 2400 units is a
   plausible sheet in millimetres and an implausible 60-metre part in
   inches.

4. **Round numbers.** People design in round numbers *in their own unit*.
   Coordinates clustering on multiples of 0.5 mm suggest metric; a pile of
   values landing on sixteenths of an inch suggests imperial, and 1/16"
   is 1.5875 mm, which is not a round metric number at all.

None of these is conclusive alone. Together they usually are, and when
they are not, :class:`UnitInference` says so and the caller can ask a
human rather than quietly scaling the part.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .annotations import Dimension
from .diagnostics import Code, Report, Severity


class Unit(str, Enum):
    """Length units a drawing might be in."""

    UNITLESS = "unitless"
    MILLIMETRE = "mm"
    CENTIMETRE = "cm"
    METRE = "m"
    INCH = "in"
    FOOT = "ft"

    @property
    def to_mm(self) -> float:
        return _TO_MM[self]

    @property
    def label(self) -> str:
        return self.value


_TO_MM: Dict[Unit, float] = {
    Unit.UNITLESS: 1.0,
    Unit.MILLIMETRE: 1.0,
    Unit.CENTIMETRE: 10.0,
    Unit.METRE: 1000.0,
    Unit.INCH: 25.4,
    Unit.FOOT: 304.8,
}

#: DXF ``$INSUNITS`` codes. 0 means the exporter declined to say.
INSUNITS: Dict[int, Unit] = {
    0: Unit.UNITLESS,
    1: Unit.INCH,
    2: Unit.FOOT,
    4: Unit.MILLIMETRE,
    5: Unit.CENTIMETRE,
    6: Unit.METRE,
}

#: Units worth considering for a part that is going to be cut.
CANDIDATES: Tuple[Unit, ...] = (Unit.MILLIMETRE, Unit.INCH, Unit.CENTIMETRE, Unit.METRE, Unit.FOOT)

#: Plausible overall size of a fabricated part, in millimetres. Below the
#: first figure it is jewellery; above the second it does not fit a sheet.
PLAUSIBLE_MM = (3.0, 6000.0)
COMFORTABLE_MM = (20.0, 3000.0)

#: Evidence weight at which confidence reaches half its agreement value.
#: Tuned so a declared header alone is decisive, a size guess alone is not.
_EVIDENCE_HALF_WEIGHT = 1.5


@dataclass(frozen=True)
class Evidence:
    """One reason to believe a particular unit."""

    source: str
    unit: Optional[Unit]
    weight: float
    detail: str

    def describe(self) -> str:
        name = self.unit.label if self.unit else "inconclusive"
        return f"{self.source}: {name} (weight {self.weight:.2f}) — {self.detail}"


@dataclass
class UnitInference:
    """What the units are, how sure we are, and why."""

    unit: Unit
    confidence: float
    evidence: List[Evidence] = field(default_factory=list)
    scores: Dict[Unit, float] = field(default_factory=dict)
    #: Set when the drawing is dimensioned in a different unit than drawn.
    annotation_unit: Optional[Unit] = None

    @property
    def scale_to_mm(self) -> float:
        return self.unit.to_mm

    @property
    def decided(self) -> bool:
        return self.confidence >= 0.6

    @property
    def contradicted(self) -> bool:
        return self.annotation_unit is not None and self.annotation_unit != self.unit

    def runner_up(self) -> Optional[Tuple[Unit, float]]:
        ranked = sorted(self.scores.items(), key=lambda kv: -kv[1])
        return ranked[1] if len(ranked) > 1 else None

    def describe(self) -> str:
        lines = [f"units: {self.unit.label} (confidence {self.confidence:.0%})"]
        if self.contradicted and self.annotation_unit:
            lines.append(
                f"  drawn in {self.unit.label} but dimensioned in {self.annotation_unit.label}"
            )
        for item in self.evidence:
            lines.append("  " + item.describe())
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Individual lines of evidence
# --------------------------------------------------------------------------


def _header_evidence(insunits: Optional[int]) -> Optional[Evidence]:
    if insunits is None:
        return Evidence("header", None, 0.0, "$INSUNITS not present")
    unit = INSUNITS.get(int(insunits))
    if unit is None:
        return Evidence("header", None, 0.0, f"$INSUNITS={insunits} is not a length unit")
    if unit is Unit.UNITLESS:
        return Evidence("header", None, 0.0, "$INSUNITS=0, the exporter declared no units")
    return Evidence("header", unit, 3.0, f"$INSUNITS={insunits}")


def _dimension_evidence(dimensions: Sequence[Dimension]) -> List[Evidence]:
    """Compare what dimensions *say* against what they *span*.

    A ratio near 25.4 or 1/25.4 between the two is not noise: it is a
    drawing whose text and geometry are in different units.
    """
    usable = [
        d
        for d in dimensions
        if d.stated_value is not None
        and d.measurement > 1e-9
        and d.stated_value > 1e-9
        and not d.is_radial  # radial text may be a diameter against a radius span
    ]
    if not usable:
        return [Evidence("dimensions", None, 0.0, "no checkable dimensions")]

    ratios = [d.stated_value / d.measurement for d in usable]  # type: ignore[operator]
    agreeing = sum(1 for r in ratios if abs(r - 1.0) < 0.01)
    mm_text_on_inch_geometry = sum(1 for r in ratios if abs(r - 25.4) < 0.3)
    inch_text_on_mm_geometry = sum(1 for r in ratios if abs(r - 1.0 / 25.4) < 0.001)

    total = len(ratios)
    evidence: List[Evidence] = []

    if agreeing == total:
        evidence.append(
            Evidence(
                "dimensions",
                None,
                0.0,
                f"all {total} dimensions agree with the geometry, so text and drawing "
                "share a unit (which one is still open)",
            )
        )
    if inch_text_on_mm_geometry >= max(1, total // 2):
        evidence.append(
            Evidence(
                "dimension/geometry ratio",
                Unit.MILLIMETRE,
                4.0,
                f"{inch_text_on_mm_geometry} of {total} dimensions read 25.4x smaller than "
                "they measure: drawn in mm, dimensioned in inches",
            )
        )
    if mm_text_on_inch_geometry >= max(1, total // 2):
        evidence.append(
            Evidence(
                "dimension/geometry ratio",
                Unit.INCH,
                4.0,
                f"{mm_text_on_inch_geometry} of {total} dimensions read 25.4x larger than "
                "they measure: drawn in inches, dimensioned in mm",
            )
        )
    return evidence


def _plausibility_evidence(extent: float) -> Evidence:
    """Score each candidate by whether the resulting part could exist."""
    if extent <= 0.0:
        return Evidence("size", None, 0.0, "no extent to judge")

    best: Optional[Unit] = None
    best_score = 0.0
    for unit in CANDIDATES:
        millimetres = extent * unit.to_mm
        if COMFORTABLE_MM[0] <= millimetres <= COMFORTABLE_MM[1]:
            score = 1.0
        elif PLAUSIBLE_MM[0] <= millimetres <= PLAUSIBLE_MM[1]:
            score = 0.5
        else:
            score = 0.0
        if score > best_score:
            best, best_score = unit, score

    if best is None:
        return Evidence(
            "size", None, 0.0, f"an extent of {extent:g} units is implausible in every unit"
        )
    return Evidence(
        "size",
        best,
        1.2 * best_score,
        f"{extent:g} units is {extent * best.to_mm:.1f} mm in {best.label}, a plausible part size",
    )


def _roundness_evidence(values: Sequence[float]) -> Evidence:
    """People design in round numbers — in whichever unit they think in."""
    samples = [abs(v) for v in values if math.isfinite(v) and abs(v) > 1e-9]
    if len(samples) < 6:
        return Evidence("round numbers", None, 0.0, "too few coordinates to judge")

    def hits(step: float) -> float:
        return sum(1 for v in samples if abs(v / step - round(v / step)) < 1e-6) / len(samples)

    metric = max(hits(0.5), hits(1.0))
    # Sixteenths of an inch, expressed in whatever the drawing unit is if
    # that unit were inches.
    imperial = max(hits(1.0 / 16.0), hits(1.0 / 8.0), hits(0.25))

    if metric >= 0.8 and metric > imperial + 0.2:
        return Evidence("round numbers", Unit.MILLIMETRE, 1.0, f"{metric:.0%} of coordinates land on 0.5 steps")
    if imperial >= 0.8 and imperial > metric + 0.2:
        return Evidence("round numbers", Unit.INCH, 1.0, f"{imperial:.0%} of coordinates land on sixteenths")
    return Evidence("round numbers", None, 0.0, "no clear preference for metric or imperial steps")


# --------------------------------------------------------------------------
# Combining
# --------------------------------------------------------------------------


def infer(
    insunits: Optional[int] = None,
    extent: float = 0.0,
    coordinates: Sequence[float] = (),
    dimensions: Sequence[Dimension] = (),
) -> UnitInference:
    """Weigh every line of evidence and decide."""
    evidence: List[Evidence] = []

    header = _header_evidence(insunits)
    if header is not None:
        evidence.append(header)
    evidence.extend(_dimension_evidence(dimensions))
    evidence.append(_plausibility_evidence(extent))
    evidence.append(_roundness_evidence(coordinates))

    scores: Dict[Unit, float] = {unit: 0.0 for unit in CANDIDATES}
    for item in evidence:
        if item.unit is not None and item.unit in scores:
            scores[item.unit] += item.weight

    total = sum(scores.values())
    if total <= 0.0:
        # Nothing to go on. Millimetres is the least-bad default because it
        # is far more common in fabrication, but confidence says otherwise.
        return UnitInference(Unit.MILLIMETRE, 0.0, evidence, scores)

    unit = max(scores.items(), key=lambda kv: kv[1])[0]

    # Confidence is agreement *and* quantity. Share-of-total alone reports
    # 100% when a single weak signal is the only voter, which is exactly
    # the case where a human should be asked instead. Saturating on the
    # total evidence keeps a lone plausibility check below the decision
    # threshold while a declared header or a dimension ratio clears it.
    agreement = scores[unit] / total
    saturation = total / (total + _EVIDENCE_HALF_WEIGHT)
    confidence = agreement * saturation

    # A drawing dimensioned in a different unit than it is drawn in is the
    # finding, not a tie to be broken silently.
    annotation_unit: Optional[Unit] = None
    for item in evidence:
        if item.source == "dimension/geometry ratio" and item.unit is not None:
            annotation_unit = Unit.INCH if item.unit is Unit.MILLIMETRE else Unit.MILLIMETRE

    return UnitInference(unit, confidence, evidence, scores, annotation_unit)


def report_units(inference: UnitInference, report: Optional[Report] = None) -> Report:
    """Turn an inference into diagnostics a caller can act on."""
    report = report or Report()

    if inference.confidence <= 0.0:
        report.add(
            Code.UNITS_UNDECLARED,
            Severity.ERROR,
            "the drawing declares no units and nothing else settles it; "
            f"assuming {inference.unit.label}",
        )
        return report

    if not inference.decided:
        runner_up = inference.runner_up()
        alternative = f", next best {runner_up[0].label}" if runner_up else ""
        report.add(
            Code.UNITS_AMBIGUOUS,
            Severity.WARNING,
            f"units inferred as {inference.unit.label} but only at "
            f"{inference.confidence:.0%} confidence{alternative}",
            confidence=inference.confidence,
        )

    if inference.contradicted and inference.annotation_unit:
        report.add(
            Code.UNITS_CONTRADICTED,
            Severity.ERROR,
            f"the part is drawn in {inference.unit.label} but dimensioned in "
            f"{inference.annotation_unit.label}; one of the two is wrong",
        )

    return report


def infer_from_dxf(path: Union[str, Path]) -> UnitInference:
    """Read a DXF and infer its units from everything it contains."""
    import ezdxf

    from .annotations import extract_dimensions

    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()

    try:
        insunits = doc.header.get("$INSUNITS")
    except Exception:
        insunits = None

    coordinates: List[float] = []
    minx = miny = math.inf
    maxx = maxy = -math.inf

    for entity in msp:
        for attribute in ("start", "end", "center", "insert"):
            if entity.dxf.hasattr(attribute):
                try:
                    point = entity.dxf.get(attribute)
                    x, y = float(point[0]), float(point[1])
                except Exception:
                    continue
                coordinates.extend((x, y))
                minx, maxx = min(minx, x), max(maxx, x)
                miny, maxy = min(miny, y), max(maxy, y)
        if entity.dxftype() == "LWPOLYLINE":
            try:
                for x, y, *_ in entity.get_points("xyb"):
                    coordinates.extend((float(x), float(y)))
                    minx, maxx = min(minx, float(x)), max(maxx, float(x))
                    miny, maxy = min(miny, float(y)), max(maxy, float(y))
            except Exception:
                pass
        if entity.dxf.hasattr("radius"):
            try:
                coordinates.append(float(entity.dxf.radius))
            except Exception:
                pass

    extent = 0.0
    if math.isfinite(minx) and math.isfinite(maxx):
        extent = max(maxx - minx, maxy - miny)

    return infer(
        insunits=insunits,
        extent=extent,
        coordinates=coordinates,
        dimensions=extract_dimensions(msp),
    )


def convert_length(value: float, source: Unit, target: Unit = Unit.MILLIMETRE) -> float:
    return value * source.to_mm / target.to_mm
