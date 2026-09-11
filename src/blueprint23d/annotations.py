"""Reading what the drawing *says*, as opposed to what it draws.

Every other module in this package reads geometry. This one reads the
dimensions, and the distinction matters more than it sounds.

Inside a live CAD model a dimension is *associative*: it is derived from
the geometry, so the two cannot disagree. But a DXF that arrives from a
customer has usually lost that association — it may have been exported
from a different system, edited by hand, scaled, or had its text
overridden to say something the geometry never said. At that point the
drawing holds two independent claims about the part, and nothing checks
them against each other.

That is what :mod:`audit` does with the output of this module, and it is
the single highest-value check in the package, because a drawing that
contradicts itself is how a part gets cut to the wrong size while every
individual step looks correct.

Three things are extracted per dimension:

* the **measurement** — what the dimension's own definition points span;
* the **displayed text** — what a human reading the drawing will believe,
  which is usually ``<>`` meaning "show the measurement" but may be an
  override that says something else entirely;
* the **reference points** — where on the part it was measuring, so the
  claim can be checked against real geometry rather than against the
  dimension's own stored copy of it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

Point2 = Tuple[float, float]

#: DXF dimension type codes, masked with 0x07.
LINEAR = 0
ALIGNED = 1
ANGULAR = 2
DIAMETER = 3
RADIUS = 4
ANGULAR_3P = 5
ORDINATE = 6

KIND_NAMES = {
    LINEAR: "linear",
    ALIGNED: "aligned",
    ANGULAR: "angular",
    DIAMETER: "diameter",
    RADIUS: "radius",
    ANGULAR_3P: "angular",
    ORDINATE: "ordinate",
}

#: Text that means "display the measured value" rather than an override.
_AUTOMATIC_TEXT = {"", "<>"}

#: Pull a number out of dimension text that may carry prefixes, suffixes
#: and tolerances: "Ø6.35", "25.4 REF", "100±0.1", "R3.0", "2X Ø5".
_NUMBER = re.compile(r"[-+]?\d*\.?\d+")


@dataclass(frozen=True)
class Dimension:
    """One dimension annotation, and the claim it makes."""

    kind: int
    measurement: float
    #: Raw text as stored; ``<>`` means the value is shown automatically.
    raw_text: str
    #: The number a human reads off the drawing, if one can be determined.
    stated_value: Optional[float]
    #: Points the dimension was measuring between, in drawing coordinates.
    reference_points: Tuple[Point2, ...] = ()
    handle: Optional[str] = None
    layer: Optional[str] = None
    text_position: Optional[Point2] = None

    @property
    def kind_name(self) -> str:
        return KIND_NAMES.get(self.kind, "unknown")

    @property
    def is_overridden(self) -> bool:
        """True when the drawing shows text that is not the measured value."""
        return self.raw_text.strip() not in _AUTOMATIC_TEXT

    @property
    def is_radial(self) -> bool:
        return self.kind in (DIAMETER, RADIUS)

    @property
    def nominal_radius(self) -> Optional[float]:
        """The radius this dimension asserts, for radial dimensions."""
        value = self.stated_value if self.stated_value is not None else self.measurement
        if self.kind == RADIUS:
            return value
        if self.kind == DIAMETER:
            return value / 2.0
        return None

    @property
    def anchor(self) -> Optional[Point2]:
        """A representative point, for reporting a location."""
        if self.reference_points:
            return self.reference_points[0]
        return self.text_position

    def describe(self) -> str:
        shown = f"{self.stated_value:g}" if self.stated_value is not None else self.raw_text or "?"
        marker = " (overridden)" if self.is_overridden else ""
        return f"{self.kind_name} {shown}{marker}, measures {self.measurement:g}"


def parse_stated_value(raw_text: str, measurement: float) -> Optional[float]:
    """What number a person reads off this dimension.

    Automatic text shows the measurement. An override may still contain
    the value inside decoration (``Ø6.35``, ``25.4 REF``, ``100±0.1``), so
    the first number is taken; text with no number at all (``TYP``,
    ``SEE NOTE``) states nothing checkable and returns ``None``.
    """
    text = raw_text.strip()
    if text in _AUTOMATIC_TEXT:
        return measurement

    # A leading "2X" or "4X" is a count, not the dimension. Drop it.
    text = re.sub(r"^\s*\d+\s*[xX]\s*", "", text)
    match = _NUMBER.search(text)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _point2(value) -> Point2:
    return (float(value[0]), float(value[1]))


def _reference_points(entity, kind: int) -> Tuple[Point2, ...]:
    """The points on the part that the dimension is measuring."""
    dxf = entity.dxf
    points: List[Point2] = []

    def take(name: str) -> None:
        if dxf.hasattr(name):
            points.append(_point2(dxf.get(name)))

    if kind in (LINEAR, ALIGNED):
        # defpoint2 and defpoint3 are the two extension-line origins.
        take("defpoint2")
        take("defpoint3")
    elif kind in (DIAMETER, RADIUS):
        # defpoint sits on the arc; defpoint4 is the centre (radius) or the
        # far side (diameter).
        take("defpoint")
        take("defpoint4")
    else:
        for name in ("defpoint", "defpoint2", "defpoint3", "defpoint4"):
            take(name)
    return tuple(points)


def extract_dimensions(msp) -> List[Dimension]:
    """Read every DIMENSION entity in a modelspace."""
    dimensions: List[Dimension] = []
    for entity in msp:
        if entity.dxftype() != "DIMENSION":
            continue
        try:
            kind = int(entity.dimtype) & 0x07
            measurement = float(entity.get_measurement())
        except Exception:
            # Angular dimensions return a tuple from some writers, and
            # broken dimensions raise; neither should cost us the rest.
            continue
        if not math.isfinite(measurement):
            continue

        raw_text = ""
        try:
            raw_text = entity.dxf.get("text", "") or ""
        except Exception:
            raw_text = ""

        text_position = None
        try:
            if entity.dxf.hasattr("text_midpoint"):
                text_position = _point2(entity.dxf.text_midpoint)
        except Exception:
            text_position = None

        dimensions.append(
            Dimension(
                kind=kind,
                measurement=measurement,
                raw_text=raw_text,
                stated_value=parse_stated_value(raw_text, measurement),
                reference_points=_reference_points(entity, kind),
                handle=getattr(entity.dxf, "handle", None),
                layer=getattr(entity.dxf, "layer", None),
                text_position=text_position,
            )
        )
    return dimensions


def load_dimensions(path) -> List[Dimension]:
    """Read the dimensions from a DXF file."""
    import ezdxf

    return extract_dimensions(ezdxf.readfile(str(path)).modelspace())
