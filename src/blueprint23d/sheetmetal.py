"""Bends, flat patterns, and the reason a folded part comes out the wrong size.

A folded bracket is drawn by its finished dimensions — 50 up, 30 across,
50 down — and cut from a flat blank that is *shorter* than 130, because
bending consumes material. How much shorter is not a detail. On a 2 mm
part with two 90° bends it is about 7 mm, which is far outside any
tolerance anybody quotes, and a shop that cuts 130 has scrapped the sheet.

The number that decides it is the **K-factor**: where through the
thickness the neutral axis sits, as a fraction of the thickness. Material
outside it stretches, material inside it compresses, and the neutral line
alone keeps its length. K is not a constant — it moves outward as the bend
gets gentler, because a generous radius strains the section less.

Rather than the usual three-row lookup table (0.33 tight, 0.41 medium,
0.45 generous), this uses the **DIN 6935** correction, which is continuous
and is what the table is a rounding of:

.. code::

    k = 0.65 + 0.5 · log₁₀(R / T),   clamped to [0.65, 1.0]
    K = k / 2                         so K runs from 0.325 to 0.5

Then, with the bend angle θ (the angle turned *through*, so a right angle
is 90°):

.. code::

    bend allowance  BA   = θ · (R + K·T)          arc of the neutral line
    outside setback OSSB = tan(θ/2) · (R + T)     apex to tangent point
    bend deduction  BD   = 2·OSSB − BA

and the flat blank is the sum of the outside flange dimensions less one
deduction per bend. That is the whole calculation, and it is exact given
K; everything uncertain is in K, which is why K is reported rather than
buried.

The rest of the module is the four rules that stop a press brake:

* a **bend radius below the material's minimum** — the outside of the bend
  is in tension and it cracks;
* a **missing relief** where a bend runs out at an edge — the material
  tears at the end of the bend line;
* a **flange too short to hold** — below about four thicknesses the part
  drops into the die;
* a **hole too close to the bend** — it draws into an oval.

Every threshold is a multiple of thickness and every one is an argument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .brep import Face2D, Loop2D
from .curves import Arc2D, Line2D
from .diagnostics import Code, Location, Report, Severity

Point2 = Tuple[float, float]


# --------------------------------------------------------------------------
# Material
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Material:
    """What the sheet is, in the two ways bending cares about."""

    name: str
    #: Smallest inside bend radius, as a multiple of thickness, before the
    #: outside of the bend starts to crack.
    min_bend_ratio: float
    #: Set to pin K rather than deriving it from the geometry. Shops that
    #: have measured their own tooling should.
    k_override: Optional[float] = None
    note: str = ""

    def but(self, **overrides: Any) -> "Material":
        return replace(self, **overrides)


#: Minimum-radius ratios are the conventional cold-forming figures for
#: annealed or soft-temper stock, bent across the grain. Bending *with* the
#: grain needs roughly twice the radius, and no drawing says which way the
#: grain runs -- so these are a floor, not a guarantee.
MATERIALS: Dict[str, Material] = {
    "mild_steel": Material("mild steel", 1.0, note="forgiving; the default for fabrication"),
    "stainless_304": Material(
        "304 stainless", 1.5, note="springs back noticeably; overbend or expect to re-strike"
    ),
    "aluminium_5052": Material("5052 aluminium", 1.0, note="the formable aluminium"),
    "aluminium_6061_t6": Material(
        "6061-T6 aluminium",
        3.0,
        note="cracks at a tight radius; anneal, or use 5052 if it must fold",
    ),
    "copper": Material("copper", 0.5, note="very formable"),
    "brass": Material("brass", 1.0),
}


def material(name: str) -> Material:
    try:
        return MATERIALS[name]
    except KeyError:
        raise ValueError(
            f"unknown material {name!r}; available: {', '.join(sorted(MATERIALS))}"
        ) from None


# --------------------------------------------------------------------------
# The bend calculation
# --------------------------------------------------------------------------


def k_factor(radius: float, thickness: float, mat: Optional[Material] = None) -> float:
    """Where the neutral axis sits, as a fraction of the thickness.

    DIN 6935's continuous correction rather than a lookup table. K rises
    with the radius because a gentler bend strains the section less and
    the neutral line moves outward; it is bounded below by 0.325 (the
    tightest practical bend) and above by 0.5 (no strain at all, the
    geometric middle).
    """
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")
    if mat is not None and mat.k_override is not None:
        return mat.k_override
    if radius <= 0.0:
        return 0.325
    k = 0.65 + 0.5 * math.log10(radius / thickness)
    return min(1.0, max(0.65, k)) / 2.0


def bend_allowance(angle: float, radius: float, thickness: float, k: float) -> float:
    """Arc length of the neutral line through the bend. ``angle`` in radians."""
    return angle * (radius + k * thickness)


def outside_setback(angle: float, radius: float, thickness: float) -> float:
    """Apex of the outside corner to the tangent point, along a flange."""
    return math.tan(angle / 2.0) * (radius + thickness)


def bend_deduction(angle: float, radius: float, thickness: float, k: float) -> float:
    """How much shorter the blank is than the sum of the outside flanges."""
    return 2.0 * outside_setback(angle, radius, thickness) - bend_allowance(
        angle, radius, thickness, k
    )


@dataclass(frozen=True)
class Bend:
    """One fold.

    ``angle`` is the angle turned *through*, in radians, so a right-angle
    bend is ``math.pi / 2`` and not ``math.pi``. ``direction`` is ``+1``
    for up and ``-1`` for down; it changes nothing about the length and
    everything about whether the part is the one that was drawn.
    """

    angle: float
    radius: float
    direction: int = 1
    #: Where the bend runs, in the flat pattern. Optional: the length
    #: calculation does not need it, the manufacturability rules do.
    line: Optional[Tuple[Point2, Point2]] = None

    @property
    def degrees(self) -> float:
        return math.degrees(self.angle)

    def k(self, thickness: float, mat: Optional[Material] = None) -> float:
        return k_factor(self.radius, thickness, mat)

    def allowance(self, thickness: float, mat: Optional[Material] = None) -> float:
        return bend_allowance(self.angle, self.radius, thickness, self.k(thickness, mat))

    def deduction(self, thickness: float, mat: Optional[Material] = None) -> float:
        return bend_deduction(self.angle, self.radius, thickness, self.k(thickness, mat))

    def describe(self, thickness: float, mat: Optional[Material] = None) -> str:
        way = "up" if self.direction >= 0 else "down"
        return (
            f"{self.degrees:g}° {way} at R{self.radius:g}: "
            f"K={self.k(thickness, mat):.3f}, "
            f"allowance {self.allowance(thickness, mat):.4g}, "
            f"deduction {self.deduction(thickness, mat):.4g}"
        )


@dataclass(frozen=True)
class FlatPattern:
    """The blank, and the arithmetic that got there."""

    flat_length: float
    folded_length: float
    thickness: float
    material: Material
    flanges: Tuple[float, ...]
    deductions: Tuple[float, ...]
    k_factors: Tuple[float, ...]

    @property
    def shortfall(self) -> float:
        """How much shorter the blank is than the finished outside length.

        The number that scraps a sheet when nobody subtracts it.
        """
        return self.folded_length - self.flat_length

    def describe(self) -> str:
        lines = [
            f"{self.material.name} at {self.thickness:g} thick",
            f"  flanges      {' + '.join(f'{f:g}' for f in self.flanges)} "
            f"= {self.folded_length:g} outside",
        ]
        for index, (deduction, k) in enumerate(zip(self.deductions, self.k_factors), 1):
            lines.append(f"  bend {index}       −{deduction:.4g}  (K={k:.3f})")
        lines.append(f"  blank        {self.flat_length:.4g}")
        lines.append(f"  shortfall    {self.shortfall:.4g}  cut this and it fits")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flat_length": self.flat_length,
            "folded_length": self.folded_length,
            "shortfall": self.shortfall,
            "thickness": self.thickness,
            "material": self.material.name,
            "flanges": list(self.flanges),
            "deductions": list(self.deductions),
            "k_factors": list(self.k_factors),
        }


def flat_pattern(
    flanges: Sequence[float],
    bends: Sequence[Bend],
    thickness: float,
    mat: Optional[Material] = None,
) -> FlatPattern:
    """Blank length for a part described by its finished outside dimensions.

    There is one more flange than there are bends, which is checked rather
    than assumed: a mismatch here is the sort of thing that produces a
    confident wrong number.
    """
    if len(flanges) != len(bends) + 1:
        raise ValueError(
            f"{len(flanges)} flanges need {len(flanges) - 1} bends, got {len(bends)}"
        )
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")

    mat = mat or MATERIALS["mild_steel"]
    deductions = tuple(bend.deduction(thickness, mat) for bend in bends)
    k_factors = tuple(bend.k(thickness, mat) for bend in bends)
    folded = float(sum(flanges))

    return FlatPattern(
        flat_length=folded - sum(deductions),
        folded_length=folded,
        thickness=thickness,
        material=mat,
        flanges=tuple(float(f) for f in flanges),
        deductions=deductions,
        k_factors=k_factors,
    )


# --------------------------------------------------------------------------
# Geometry around a bend line
# --------------------------------------------------------------------------


def _point_to_segment(p: Point2, a: Point2, b: Point2) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length_squared = dx * dx + dy * dy
    if length_squared <= 0.0:
        return math.dist(p, a)
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / length_squared
    t = max(0.0, min(1.0, t))
    return math.dist(p, (ax + t * dx, ay + t * dy))


def _signed_offset(p: Point2, a: Point2, b: Point2) -> Tuple[float, float]:
    """Perpendicular offset of ``p`` from line ``ab``, and its projection.

    The projection is returned as a fraction along the segment, so a caller
    can ignore boundary that sits off the end of the bend.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length <= 0.0:
        return 0.0, 0.0
    ux, uy = dx / length, dy / length
    px, py = p[0] - a[0], p[1] - a[1]
    along = (px * ux + py * uy) / length
    across = px * -uy + py * ux
    return across, along


def flange_widths(
    face: Face2D, bend: Bend, tolerance: float = 0.05
) -> Tuple[float, float]:
    """How much material lies each side of a bend line.

    Measured perpendicular to the bend, over the stretch of boundary that
    actually faces it, which is the width the press brake has to grip.
    Returns ``(0.0, 0.0)`` if the bend has no line.
    """
    if bend.line is None:
        return (0.0, 0.0)
    a, b = bend.line
    left = math.inf
    right = math.inf
    for point in face.outer.tessellate(tolerance):
        across, along = _signed_offset(point, a, b)
        if not (-0.05 <= along <= 1.05):
            continue
        if across > 0.0:
            left = min(left, across)
        elif across < 0.0:
            right = min(right, -across)
    return (0.0 if not math.isfinite(left) else left,
            0.0 if not math.isfinite(right) else right)


def has_relief(
    face: Face2D, endpoint: Point2, window: float, thickness: float, tolerance: float = 0.05
) -> bool:
    """Whether the outer boundary near a bend's end is notched.

    A relief is a notch, and a notch is a departure from a straight edge.
    So: take the boundary within ``window`` of where the bend runs out and
    ask whether it is straight. If every sample sits within half a
    thickness of the chord through the first and last of them, the edge
    runs straight past and there is no relief there.

    A heuristic, and an honest one: it cannot tell a relief from any other
    notch, so it errs toward believing a relief exists rather than
    reporting one that is there as missing.
    """
    points = [
        p for p in face.outer.tessellate(tolerance) if math.dist(p, endpoint) <= window
    ]
    if len(points) < 3:
        return False

    first, last = points[0], points[-1]
    if math.dist(first, last) < 1e-12:
        return True
    deviation = max(_point_to_segment(p, first, last) for p in points)
    return deviation > 0.5 * thickness


# --------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------


def check_sheet_metal(
    face: Face2D,
    bends: Sequence[Bend],
    thickness: float,
    mat: Optional[Material] = None,
    min_flange_ratio: float = 4.0,
    hole_clearance_ratio: float = 2.5,
    tolerance: float = 0.05,
    report: Optional[Report] = None,
) -> Report:
    """Check a flat pattern and its bend lines against a press brake."""
    report = report or Report()
    mat = mat or MATERIALS["mild_steel"]

    minimum_radius = mat.min_bend_ratio * thickness
    minimum_flange = min_flange_ratio * thickness

    for index, bend in enumerate(bends, 1):
        where = _bend_midpoint(bend)

        if bend.radius < minimum_radius - 1e-12:
            report.add(
                Code.BEND_RADIUS_TOO_SMALL,
                Severity.ERROR if bend.radius < 0.6 * minimum_radius else Severity.WARNING,
                f"bend {index} is drawn at R{bend.radius:g} and {mat.name} at "
                f"{thickness:g} thick wants at least R{minimum_radius:g}; the "
                "outside of the bend is in tension and will crack",
                Location(point=where),
                radius=bend.radius,
                minimum=minimum_radius,
            )

        if bend.line is None:
            continue

        left, right = flange_widths(face, bend, tolerance)
        for side, width in (("one side", left), ("the other", right)):
            if 0.0 < width < minimum_flange:
                report.add(
                    Code.FLANGE_TOO_SHORT,
                    Severity.ERROR if width < 0.6 * minimum_flange else Severity.WARNING,
                    f"bend {index} leaves only {width:.4g} on {side}, against "
                    f"{minimum_flange:g} ({min_flange_ratio:g}×t) to hold; the part "
                    "will drop into the die",
                    Location(point=where),
                    width=width,
                    minimum=minimum_flange,
                )

        window = 2.0 * (bend.radius + thickness)
        for endpoint in bend.line:
            if _on_boundary(face, endpoint, thickness, tolerance) and not has_relief(
                face, endpoint, window, thickness, tolerance
            ):
                report.add(
                    Code.BEND_RELIEF_MISSING,
                    Severity.WARNING,
                    f"bend {index} runs out at a straight edge with no relief; "
                    f"notch it at least {thickness:g} wide and "
                    f"{bend.radius + thickness:g} deep or it will tear here",
                    Location(point=endpoint),
                    width=thickness,
                    depth=bend.radius + thickness,
                )

        clearance = hole_clearance_ratio * thickness + bend.radius
        for loop in face.inners:
            distance, centre, size = _hole_clearance(loop, bend.line, tolerance)
            if distance < clearance:
                report.add(
                    Code.HOLE_NEAR_BEND,
                    Severity.ERROR if distance < 0.6 * clearance else Severity.WARNING,
                    f"a hole sits {distance:.4g} from bend {index}, against "
                    f"{clearance:.4g} ({hole_clearance_ratio:g}×t + R); it will "
                    "draw into an oval",
                    Location(point=centre),
                    distance=distance,
                    minimum=clearance,
                    hole=size,
                )

    report.metrics["bends"] = len(bends)
    report.metrics["thickness"] = thickness
    report.metrics["material"] = mat.name
    report.metrics["minimum_bend_radius"] = minimum_radius
    return report


def _bend_midpoint(bend: Bend) -> Optional[Point2]:
    if bend.line is None:
        return None
    (x0, y0), (x1, y1) = bend.line
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _on_boundary(face: Face2D, point: Point2, thickness: float, tolerance: float) -> bool:
    """Whether a bend line's end reaches the edge of the blank."""
    points = face.outer.tessellate(tolerance)
    n = len(points)
    best = min(
        _point_to_segment(point, points[i], points[(i + 1) % n]) for i in range(n)
    )
    return best <= max(thickness, 10.0 * tolerance)


def _hole_clearance(
    loop: Loop2D, line: Tuple[Point2, Point2], tolerance: float
) -> Tuple[float, Point2, float]:
    """Gap between a hole's edge and a bend line, with the hole's centre."""
    a, b = line
    x0, y0, x1, y1 = loop.bounds()
    centre = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    size = min(x1 - x0, y1 - y0)
    best = min(_point_to_segment(p, a, b) for p in loop.tessellate(tolerance))
    return best, centre, size


# --------------------------------------------------------------------------
# Reading bend lines out of a drawing
# --------------------------------------------------------------------------

#: Bend lines have no standard representation in DXF, so the convention is
#: the layer name: ``BEND``, or ``BEND UP 90 R2`` / ``BEND_DOWN_45``.
#: Anything not stated falls back to the arguments of :func:`load_bends`.
_DEFAULT_BEND_LAYER = "BEND"


def parse_bend_layer(
    name: str, default_angle: float, default_radius: float
) -> Tuple[float, float, int]:
    """Pull angle, radius and direction out of a layer name.

    Returns ``(angle_radians, radius, direction)``, falling back to the
    defaults for whatever the name does not say.
    """
    import re

    upper = name.upper()
    direction = -1 if "DOWN" in upper else 1

    angle = default_angle
    radius = default_radius

    radius_match = re.search(r"R\s*([0-9]*\.?[0-9]+)", upper)
    if radius_match:
        radius = float(radius_match.group(1))
        upper = upper[: radius_match.start()] + upper[radius_match.end() :]

    angle_match = re.search(r"(?<![A-Z0-9.])([0-9]*\.?[0-9]+)", upper)
    if angle_match:
        angle = math.radians(float(angle_match.group(1)))

    return angle, radius, direction


def load_bends(
    path,
    layer: str = _DEFAULT_BEND_LAYER,
    default_angle: float = math.pi / 2,
    default_radius: float = 1.0,
) -> Tuple[List[Bend], Report]:
    """Read bend lines from a DXF, by layer-name convention.

    Every ``LINE`` on a layer whose name starts with ``layer`` becomes a
    bend; the rest of the layer name supplies the angle, radius and
    direction where it says so.
    """
    import ezdxf

    report = Report(source=str(path))
    bends: List[Bend] = []
    try:
        doc = ezdxf.readfile(str(path))
    except Exception as exc:
        report.add(Code.FILE_UNREADABLE, Severity.CRITICAL, str(exc))
        return bends, report

    prefix = layer.upper()
    for entity in doc.modelspace():
        if entity.dxftype() != "LINE":
            continue
        name = str(entity.dxf.layer)
        if not name.upper().startswith(prefix):
            continue
        angle, radius, direction = parse_bend_layer(name, default_angle, default_radius)
        start = (float(entity.dxf.start.x), float(entity.dxf.start.y))
        end = (float(entity.dxf.end.x), float(entity.dxf.end.y))
        if math.dist(start, end) <= 0.0:
            report.add(
                Code.BEND_LINE_INVALID,
                Severity.WARNING,
                f"a zero-length entity on layer {name} is not a bend line",
                Location(point=start, layer=name),
            )
            continue
        bends.append(Bend(angle=angle, radius=radius, direction=direction, line=(start, end)))

    if not bends:
        report.add(
            Code.BEND_LINE_INVALID,
            Severity.INFO,
            f"no bend lines found on a layer starting with {layer!r}; "
            "this was read as a flat part",
        )
    report.metrics["bends"] = len(bends)
    return bends, report


def check_drawing(
    path,
    thickness: float,
    material_name: str = "mild_steel",
    layer: Optional[str] = None,
    bend_layer: str = _DEFAULT_BEND_LAYER,
    **kwargs: Any,
) -> Tuple[List[Bend], Report]:
    """Read a flat pattern with bend lines and check it end to end."""
    from .diagnostics import merge
    from .loaders import load_faces

    mat = material(material_name)
    report = Report(source=str(path))

    try:
        faces, _ = load_faces(path, layer=layer)
    except Exception as exc:
        report.add(Code.FILE_UNREADABLE, Severity.CRITICAL, str(exc))
        return [], report

    if not faces:
        report.add(Code.NO_CLOSED_PROFILE, Severity.CRITICAL, "no flat pattern to check")
        return [], report

    bends, bend_report = load_bends(path, layer=bend_layer)
    face = max(faces, key=lambda f: f.area())
    combined = merge(report, bend_report)
    check_sheet_metal(face, bends, thickness, mat, report=combined, **kwargs)
    combined.source = str(path)
    return bends, combined
