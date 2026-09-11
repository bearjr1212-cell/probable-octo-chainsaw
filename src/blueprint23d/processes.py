"""Cutting processes: what they can make, and what it costs to make it.

A drawing is not manufacturable in the abstract. It is manufacturable on a
*machine*, and the same file is fine on one and impossible on the next. A
2 mm hole through 6 mm plate is routine on a waterjet, marginal on a fibre
laser, and cannot be pierced at all on a plasma table, because the arc is
wider than the hole. Nothing in the DXF knows that.

So every manufacturability rule here takes a :class:`Process` and a
thickness, and the same geometry is checked against a different standard
for each. The rules are the ones that actually stop jobs:

* a feature narrower than the **kerf** -- the cutter physically does not
  fit, and the slot comes out as a single melted line;
* a hole small relative to the **thickness** -- thermal processes cannot
  pierce a hole much smaller than the plate is thick without the molten
  edge closing it back up;
* a **web** between two cuts too narrow to survive -- it burns through and
  the part drops into the slats;
* an **internal corner** sharper than the tool -- a router leaves the tool
  radius there whatever the drawing says, so the corner is out of
  tolerance before the first chip;
* a part that **does not fit the sheet**.

Every threshold is a ratio of the thickness rather than an absolute, which
is how shop rules of thumb are actually stated, and every one of them is
an argument with a default rather than a constant in the code.

**On the numbers.** The built-in profiles are industry rules of thumb, not
measurements of anyone's machine. A shop's own feed rates, kerf and
minimum web are better than these in every case, which is why
:class:`Process` is a frozen dataclass with :meth:`Process.but` -- override
what you know and keep the rest. The feed model is a power law
``feed = C / tᵇ`` fitted to published cutting charts; it reproduces them to
within about 15% over the usual range, and it is reported as an
*estimate*, never as a certified time. The cut length it multiplies, by
contrast, is exact: arcs have closed-form length and nothing here
tessellates them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .brep import Face2D, Loop2D
from .curves import Arc2D, Curve2D, Line2D
from .diagnostics import Code, Location, Report, Severity

Point2 = Tuple[float, float]

SECONDS_PER_MINUTE = 60.0


# --------------------------------------------------------------------------
# How fast the machine cuts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FeedModel:
    """Cutting speed as a function of material thickness, in mm/min."""

    def rate(self, thickness: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class PowerLawFeed(FeedModel):
    """``feed = coefficient / thickness ** exponent``.

    Thermal cutting charts are very close to straight lines on log-log
    axes, because the rate is set by how fast energy can be delivered
    through the section. Two published points fix the whole curve, which is
    why this is two numbers rather than a table nobody will maintain.
    """

    coefficient: float
    exponent: float

    def rate(self, thickness: float) -> float:
        if thickness <= 0.0:
            raise ValueError("thickness must be positive")
        return self.coefficient / (thickness**self.exponent)

    def describe(self) -> str:
        return f"{self.coefficient:g} / t^{self.exponent:g} mm/min"


@dataclass(frozen=True)
class AreaRateFeed(FeedModel):
    """Wire EDM: the machine removes a fixed cut *area* per minute.

    Speed along the path is therefore inversely proportional to thickness,
    exactly, rather than by a fitted exponent.
    """

    area_rate: float  # mm² of cut face per minute

    def rate(self, thickness: float) -> float:
        if thickness <= 0.0:
            raise ValueError("thickness must be positive")
        return self.area_rate / thickness

    def describe(self) -> str:
        return f"{self.area_rate:g} mm²/min of cut face"


@dataclass(frozen=True)
class PassFeed(FeedModel):
    """Milling: constant feed, but the path is walked once per depth pass."""

    feed: float
    pass_depth: float

    def passes(self, thickness: float) -> int:
        return max(1, math.ceil(thickness / self.pass_depth))

    def rate(self, thickness: float) -> float:
        # Effective rate along the profile once the passes are accounted for.
        return self.feed / self.passes(thickness)

    def describe(self) -> str:
        return f"{self.feed:g} mm/min in {self.pass_depth:g} mm passes"


# --------------------------------------------------------------------------
# The process
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Process:
    """One cutting process on one machine.

    Every field is a default to be overridden with shop data. See
    :meth:`but`.
    """

    name: str
    feed: FeedModel
    #: Width of material the cut consumes. A feature narrower than this
    #: cannot be produced at all.
    kerf: float
    #: Smallest hole diameter as a multiple of thickness.
    min_hole_ratio: float
    #: Narrowest surviving web between two cuts, as a multiple of thickness.
    min_bridge_ratio: float
    #: Radius the process leaves in an internal corner. Zero for a thermal
    #: beam, the cutter radius for a router.
    tool_radius: float
    #: Time to punch through before the cut can start.
    pierce_seconds: float
    #: Positioning speed between cuts.
    rapid_rate: float
    #: Machine plus operator, per hour, in whatever currency you quote in.
    machine_rate: float
    #: Load, unload, program call-up.
    setup_seconds: float
    #: Usable sheet, as (length, width).
    sheet: Tuple[float, float]
    max_thickness: float
    #: Profile tolerance the process typically holds.
    positional_tolerance: float
    edge_note: str = ""

    def but(self, **overrides: Any) -> "Process":
        """A copy with some fields replaced -- your kerf, their feed rates."""
        return replace(self, **overrides)

    # ---- derived thresholds -------------------------------------------

    def min_hole_diameter(self, thickness: float) -> float:
        return max(self.min_hole_ratio * thickness, self.kerf * 2.0)

    def min_bridge(self, thickness: float) -> float:
        return max(self.min_bridge_ratio * thickness, self.kerf)

    def min_feature(self, thickness: float) -> float:
        """Narrowest slot or web the process can make at all."""
        return self.kerf

    def cut_seconds(self, length: float, thickness: float) -> float:
        return length / self.feed.rate(thickness) * SECONDS_PER_MINUTE

    def describe(self) -> str:
        return (
            f"{self.name}: kerf {self.kerf:g}, feed {self.feed.describe()}, "
            f"min hole {self.min_hole_ratio:g}×t, min web {self.min_bridge_ratio:g}×t, "
            f"up to {self.max_thickness:g} thick"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kerf": self.kerf,
            "feed": self.feed.describe(),
            "min_hole_ratio": self.min_hole_ratio,
            "min_bridge_ratio": self.min_bridge_ratio,
            "tool_radius": self.tool_radius,
            "pierce_seconds": self.pierce_seconds,
            "machine_rate": self.machine_rate,
            "sheet": list(self.sheet),
            "max_thickness": self.max_thickness,
            "positional_tolerance": self.positional_tolerance,
        }


#: Built-in profiles. Mild steel, metric, and deliberately conservative.
#: These are starting points -- see the module docstring.
PROCESSES: Dict[str, Process] = {
    "fiber_laser": Process(
        name="fibre laser (3 kW, mild steel)",
        feed=PowerLawFeed(coefficient=8000.0, exponent=0.86),
        kerf=0.2,
        min_hole_ratio=1.0,
        min_bridge_ratio=1.0,
        tool_radius=0.0,
        pierce_seconds=0.6,
        rapid_rate=60000.0,
        machine_rate=90.0,
        setup_seconds=120.0,
        sheet=(3000.0, 1500.0),
        max_thickness=20.0,
        positional_tolerance=0.1,
        edge_note="square edge, light heat-affected zone, no secondary work on most jobs",
    ),
    "co2_laser": Process(
        name="CO₂ laser (4 kW, mild steel)",
        feed=PowerLawFeed(coefficient=6000.0, exponent=0.80),
        kerf=0.25,
        min_hole_ratio=1.0,
        min_bridge_ratio=1.0,
        tool_radius=0.0,
        pierce_seconds=1.0,
        rapid_rate=40000.0,
        machine_rate=85.0,
        setup_seconds=150.0,
        sheet=(3000.0, 1500.0),
        max_thickness=25.0,
        positional_tolerance=0.15,
        edge_note="square edge; slower than fibre on thin sheet, comparable on plate",
    ),
    "plasma": Process(
        name="plasma (100 A, mild steel)",
        feed=PowerLawFeed(coefficient=21000.0, exponent=1.10),
        kerf=2.5,
        min_hole_ratio=1.5,
        min_bridge_ratio=1.5,
        tool_radius=0.0,
        pierce_seconds=1.5,
        rapid_rate=25000.0,
        machine_rate=55.0,
        setup_seconds=180.0,
        sheet=(3000.0, 1500.0),
        max_thickness=25.0,
        positional_tolerance=0.8,
        edge_note="tapered edge and a real heat-affected zone; holes below 1.5×t close up on pierce",
    ),
    "waterjet": Process(
        name="abrasive waterjet",
        feed=PowerLawFeed(coefficient=2150.0, exponent=1.20),
        kerf=1.0,
        min_hole_ratio=0.5,
        min_bridge_ratio=0.6,
        tool_radius=0.0,
        pierce_seconds=8.0,
        rapid_rate=15000.0,
        machine_rate=110.0,
        setup_seconds=180.0,
        sheet=(3000.0, 1500.0),
        max_thickness=150.0,
        positional_tolerance=0.15,
        edge_note="no heat at all, so no distortion and no hardened edge; slow and abrasive is billed",
    ),
    "router": Process(
        name="CNC router (6 mm cutter, aluminium)",
        feed=PassFeed(feed=2500.0, pass_depth=3.0),
        kerf=6.0,
        min_hole_ratio=0.0,
        min_bridge_ratio=1.0,
        tool_radius=3.0,
        pierce_seconds=4.0,
        rapid_rate=20000.0,
        machine_rate=70.0,
        setup_seconds=300.0,
        sheet=(2440.0, 1220.0),
        max_thickness=25.0,
        positional_tolerance=0.1,
        edge_note="every internal corner comes out at the cutter radius, whatever the drawing shows",
    ),
    "wire_edm": Process(
        name="wire EDM (0.25 mm wire)",
        feed=AreaRateFeed(area_rate=200.0),
        kerf=0.3,
        min_hole_ratio=0.0,
        min_bridge_ratio=0.3,
        tool_radius=0.15,
        pierce_seconds=180.0,  # threading the wire, and a start hole must exist
        rapid_rate=3000.0,
        machine_rate=140.0,
        setup_seconds=900.0,
        sheet=(400.0, 300.0),
        max_thickness=300.0,
        positional_tolerance=0.005,
        edge_note="the tightest tolerance on this list and the slowest; every internal cut needs a start hole",
    ),
}


def process(name: str) -> Process:
    """Look up a built-in profile by key."""
    try:
        return PROCESSES[name]
    except KeyError:
        raise ValueError(
            f"unknown process {name!r}; available: {', '.join(sorted(PROCESSES))}"
        ) from None


# --------------------------------------------------------------------------
# Geometry the rules need
# --------------------------------------------------------------------------


def _samples(loop: Loop2D, tolerance: float) -> List[Point2]:
    return loop.tessellate(tolerance)


def _closest_on_segment(p: Point2, a: Point2, b: Point2) -> Tuple[float, Point2, float]:
    """Distance from ``p`` to segment ``ab``, the foot of it, and its parameter.

    Point-to-*segment* rather than point-to-point, because a straight edge
    tessellates to its two endpoints: a hole sitting 0.5 mm off the middle
    of a long edge is 0.5 mm from the edge and fifty from either corner,
    and a point-to-point measure would report the fifty.
    """
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length_squared = dx * dx + dy * dy
    if length_squared <= 0.0:
        return math.dist(p, a), a, 0.0
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / length_squared
    t = max(0.0, min(1.0, t))
    foot = (ax + t * dx, ay + t * dy)
    return math.dist(p, foot), foot, t


def _edges(points: Sequence[Point2]):
    n = len(points)
    for i in range(n):
        yield points[i], points[(i + 1) % n]


#: A distance across a feature counts as a *width* only when the boundary
#: has to travel at least this many times further to get from one side to
#: the other. Without it, every tessellation step reads as a narrow gap.
#: At 1.5 a circle's diameter qualifies (half the circumference is πr
#: against 1.5 × 2r) and a tessellation chord does not.
_WIDTH_DETOUR = 1.5


def narrowest_width(loop: Loop2D, tolerance: float = 0.01) -> Tuple[float, Point2]:
    """Narrowest across-the-feature distance, and where it is.

    The quantity that decides whether a cutter fits down a slot. Two
    independent measures, because either alone misses real cases:

    * the smaller side of the bounding box, which is exact and catches any
      compact feature -- a 0.15 mm square hole, a thin slot;
    * the shortest straight line between two parts of the boundary that are
      far apart *along* it, which catches the narrow leg of an L-shaped
      cutout that the bounding box says nothing about.

    The bounding box is exact. The second measure runs on the
    tessellation, so the proven chordal deviation is subtracted from it,
    which makes the returned value a **lower bound** on the true width.
    That is the safe direction for a manufacturability check: it can cause
    an extra flag on a marginal feature, never a missed one on a feature
    the cutter genuinely will not fit.
    """
    x0, y0, x1, y1 = loop.bounds()
    best = min(x1 - x0, y1 - y0)
    where = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    points = _samples(loop, tolerance)
    n = len(points)
    if n < 4:
        return best, where

    # Arc length along the boundary to each sample, and the total.
    along = [0.0]
    for a, b in _edges(points):
        along.append(along[-1] + math.dist(a, b))
    perimeter = along[-1]

    for i, p in enumerate(points):
        for j in range(n):
            if j == i or (j + 1) % n == i:
                continue  # the edges p sits on are not across from it
            distance, foot, t = _closest_on_segment(p, points[j], points[(j + 1) % n])
            if distance >= best or distance <= 0.0:
                continue
            # Distance *along the boundary* to the foot itself, not to the
            # segment's start: a foot at the far end of an edge is that
            # much further round, and using the start under-counts it.
            foot_along = along[j] + t * (along[j + 1] - along[j])
            gap = abs(foot_along - along[i])
            detour = min(gap, perimeter - gap)
            if detour > _WIDTH_DETOUR * distance:
                best = distance
                where = ((p[0] + foot[0]) / 2.0, (p[1] + foot[1]) / 2.0)

    if best < min(x1 - x0, y1 - y0):
        best = max(0.0, best - 2.0 * loop.deviation_bound(tolerance))
    return best, where


def loop_clearance(a: Loop2D, b: Loop2D, tolerance: float = 0.01) -> Tuple[float, Point2]:
    """Closest approach between two loops, and the midpoint of that gap.

    This is the width of the web the process has to leave standing.
    Measured point-to-segment in both directions, so a long straight edge
    is treated as an edge and not as its two endpoints.

    Both tessellations are inscribed, so the polyline gap can overstate
    the true one. Each loop's proven deviation bound is subtracted, making
    the result a **lower bound** on the real web -- the direction that
    cannot hide a web too narrow to survive.
    """
    pa = _samples(a, tolerance)
    pb = _samples(b, tolerance)
    best = math.inf
    where = (0.0, 0.0)

    for points, others in ((pa, pb), (pb, pa)):
        for p in points:
            for start, end in _edges(others):
                distance, foot, _ = _closest_on_segment(p, start, end)
                if distance < best:
                    best = distance
                    where = ((p[0] + foot[0]) / 2.0, (p[1] + foot[1]) / 2.0)

    slack = a.deviation_bound(tolerance) + b.deviation_bound(tolerance)
    return max(0.0, best - slack), where


def _tangent_angle(curve: Curve2D, t: float) -> float:
    dx, dy = curve.tangent(t)
    return math.atan2(dy, dx)


def corner_angles(loop: Loop2D) -> List[Tuple[Point2, float, float]]:
    """Interior angle at every join, with the radius blending it.

    Returns ``(point, interior_angle, radius)``. The radius is the arc's
    radius when one of the two curves meeting there is an arc, and zero at
    a true corner between two straight edges -- which is the case a router
    cannot make.
    """
    corners: List[Tuple[Point2, float, float]] = []
    curves = loop.oriented(True).curves
    count = len(curves)
    if count < 2:
        return corners

    for index in range(count):
        incoming = curves[index - 1]
        outgoing = curves[index]
        point = outgoing.start

        turn = _tangent_angle(outgoing, 0.0) - _tangent_angle(incoming, 1.0)
        # Normalise to (-pi, pi]: the signed turn taken at the corner.
        turn = (turn + math.pi) % (2.0 * math.pi) - math.pi
        # Counter-clockwise loop: a left turn is convex, a right turn cuts
        # into the material. Interior angle is pi minus the turn.
        interior = math.pi - turn

        radius = 0.0
        for curve in (incoming, outgoing):
            if isinstance(curve, Arc2D):
                radius = max(radius, curve.radius)
        corners.append((point, interior, radius))
    return corners


# --------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------


def check_manufacturability(
    face: Face2D,
    proc: Process,
    thickness: float,
    tolerance: float = 0.01,
    report: Optional[Report] = None,
) -> Report:
    """Check one part against one process at one thickness."""
    report = report or Report()

    if thickness > proc.max_thickness:
        report.add(
            Code.THICKNESS_EXCEEDS_PROCESS,
            Severity.ERROR,
            f"{thickness:g} thick is beyond what {proc.name} will cut "
            f"({proc.max_thickness:g} max)",
            thickness=thickness,
            limit=proc.max_thickness,
        )

    _check_sheet(face, proc, report)
    _check_holes(face, proc, thickness, tolerance, report)
    _check_webs(face, proc, thickness, tolerance, report)
    _check_corners(face, proc, report)

    report.metrics["process"] = proc.name
    report.metrics["thickness"] = thickness
    report.metrics["min_hole_diameter"] = proc.min_hole_diameter(thickness)
    report.metrics["min_web"] = proc.min_bridge(thickness)
    return report


def _check_sheet(face: Face2D, proc: Process, report: Report) -> None:
    x0, y0, x1, y1 = face.bounds()
    width, height = x1 - x0, y1 - y0
    long_side, short_side = max(width, height), min(width, height)
    sheet_long, sheet_short = max(proc.sheet), min(proc.sheet)
    if long_side > sheet_long or short_side > sheet_short:
        report.add(
            Code.PART_EXCEEDS_SHEET,
            Severity.ERROR,
            f"the part is {width:g} × {height:g} and will not fit a "
            f"{proc.sheet[0]:g} × {proc.sheet[1]:g} sheet even turned",
            Location(point=((x0 + x1) / 2.0, (y0 + y1) / 2.0)),
            width=width,
            height=height,
        )


def _check_holes(
    face: Face2D, proc: Process, thickness: float, tolerance: float, report: Report
) -> None:
    floor = proc.min_hole_diameter(thickness)
    for loop in face.inners:
        circle = _as_circle(loop)
        if circle is not None:
            diameter = 2.0 * circle.radius
            centre = circle.center
            width = diameter
        else:
            width, centre = narrowest_width(loop, tolerance)
            diameter = width

        if width < proc.kerf:
            report.add(
                Code.FEATURE_BELOW_KERF,
                Severity.ERROR,
                f"this opening is {width:.4g} across and the kerf is "
                f"{proc.kerf:g}; the cutter does not fit and it will come out "
                "as a single melted line",
                Location(point=centre),
                width=width,
                kerf=proc.kerf,
            )
            continue

        if circle is not None and diameter < floor:
            severity = Severity.ERROR if diameter < 0.7 * floor else Severity.WARNING
            report.add(
                Code.HOLE_TOO_SMALL_FOR_THICKNESS,
                severity,
                f"Ø{diameter:.4g} through {thickness:g} thick is below the "
                f"{proc.min_hole_ratio:g}×t rule for {proc.name} "
                f"(Ø{floor:.4g} minimum); drill it as a second operation",
                Location(point=centre),
                diameter=diameter,
                minimum=floor,
                thickness=thickness,
            )


def _check_webs(
    face: Face2D, proc: Process, thickness: float, tolerance: float, report: Report
) -> None:
    minimum = proc.min_bridge(thickness)
    loops = list(face.loops())
    for i, a in enumerate(loops):
        for b in loops[i + 1 :]:
            gap, where = loop_clearance(a, b, tolerance)
            if gap < minimum:
                report.add(
                    Code.BRIDGE_TOO_NARROW,
                    Severity.ERROR if gap < 0.6 * minimum else Severity.WARNING,
                    f"a web of {gap:.4g} or a shade more between two cuts, "
                    f"against a {minimum:.4g} minimum for {proc.name} at "
                    f"{thickness:g} thick; it will burn through and the part "
                    "will drop",
                    Location(point=where),
                    gap=gap,
                    minimum=minimum,
                )


def _check_corners(face: Face2D, proc: Process, report: Report) -> None:
    if proc.tool_radius <= 0.0:
        return
    for loop_index, loop in enumerate(face.loops()):
        inner = loop_index > 0
        for point, interior, radius in corner_angles(loop):
            # corner_angles orients every loop counter-clockwise, so on the
            # outer profile the material lies inside and a corner the tool
            # must reach into is reflex. A hole re-oriented the same way has
            # its material outside, which flips the test.
            reaches_in = interior > math.pi if not inner else interior < math.pi
            if not reaches_in:
                continue
            if radius >= proc.tool_radius:
                continue
            report.add(
                Code.INTERNAL_CORNER_TOO_SHARP,
                Severity.WARNING,
                f"an internal corner with a {radius:g} radius, cut with a "
                f"{proc.tool_radius * 2:g} mm cutter: it will come out at "
                f"R{proc.tool_radius:g} however it is drawn",
                Location(point=point),
                radius=radius,
                tool_radius=proc.tool_radius,
            )


def _as_circle(loop: Loop2D) -> Optional[Arc2D]:
    if len(loop.curves) != 1:
        return None
    curve = loop.curves[0]
    if isinstance(curve, Arc2D) and abs(abs(curve.sweep) - 2.0 * math.pi) < 1e-9:
        return curve
    return None


# --------------------------------------------------------------------------
# What it costs
# --------------------------------------------------------------------------


def _nearest_neighbour_travel(starts: Sequence[Point2]) -> float:
    """Rapid distance visiting every pierce point, greedily from the origin.

    Not the optimal tour -- that is the travelling salesman -- but a
    consistent one, which is what a comparison between two quotes needs.
    """
    if not starts:
        return 0.0
    remaining = list(starts)
    here = (0.0, 0.0)
    total = 0.0
    while remaining:
        index = min(range(len(remaining)), key=lambda i: math.dist(here, remaining[i]))
        total += math.dist(here, remaining[index])
        here = remaining.pop(index)
    return total


@dataclass(frozen=True)
class Quote:
    """What one part costs on one machine.

    ``cut_length`` and ``pierces`` are exact -- arcs have closed-form
    length and a pierce is a loop. Everything downstream of a feed rate is
    an estimate, and is named as one.
    """

    process: Process
    thickness: float
    cut_length: float
    pierces: int
    rapid_length: float
    cut_seconds: float
    pierce_seconds: float
    rapid_seconds: float
    setup_seconds: float
    part_area: float
    blank_area: float
    #: Parts per sheet, from :mod:`blueprint23d.nesting`. Zero when nesting
    #: was not asked for, in which case material is not costed at all
    #: rather than guessed.
    per_sheet: int = 0
    sheet_cost: float = 0.0

    @property
    def cycle_seconds(self) -> float:
        """Machine time for one part, excluding setup."""
        return self.cut_seconds + self.pierce_seconds + self.rapid_seconds

    @property
    def total_seconds(self) -> float:
        return self.cycle_seconds + self.setup_seconds

    @property
    def machine_cost(self) -> float:
        return self.total_seconds / 3600.0 * self.process.machine_rate

    @property
    def utilisation(self) -> float:
        """Part area over the blank it has to come out of."""
        return self.part_area / self.blank_area if self.blank_area else 0.0

    @property
    def material_cost(self) -> float:
        """Sheet price divided by the sheet's yield.

        The offcut is bought too, so a part that nests badly carries the
        scrap it creates. Zero when no nesting estimate was supplied.
        """
        if self.per_sheet <= 0 or not self.sheet_cost:
            return 0.0
        return self.sheet_cost / self.per_sheet

    @property
    def total_cost(self) -> float:
        return self.machine_cost + self.material_cost

    def batch_seconds(self, quantity: int) -> float:
        """Setup once, cycle ``quantity`` times -- which is the whole
        reason a quote of one is never a tenth of a quote of ten."""
        return self.setup_seconds + quantity * self.cycle_seconds

    def batch_cost(self, quantity: int) -> float:
        return self.batch_seconds(quantity) / 3600.0 * self.process.machine_rate

    def unit_cost(self, quantity: int) -> float:
        """Machine time per part at this batch size, plus material."""
        return self.batch_cost(quantity) / max(1, quantity) + self.material_cost

    def describe(self) -> str:
        return "\n".join(
            [
                f"{self.process.name} @ {self.thickness:g} thick",
                f"  cut length   {self.cut_length:10.2f}  (exact)",
                f"  pierces      {self.pierces:10d}",
                f"  rapid        {self.rapid_length:10.2f}",
                f"  cutting      {self.cut_seconds:10.1f} s  "
                f"at {self.process.feed.rate(self.thickness):.0f} mm/min (estimate)",
                f"  piercing     {self.pierce_seconds:10.1f} s",
                f"  positioning  {self.rapid_seconds:10.1f} s",
                f"  setup        {self.setup_seconds:10.1f} s",
                f"  cycle        {self.cycle_seconds:10.1f} s per part",
                f"  utilisation  {self.utilisation * 100:9.1f} %  of the blank",
                f"  cost         {self.machine_cost:10.2f}  machine, for one",
            ]
            + (
                [
                    f"  material     {self.material_cost:10.2f}  "
                    f"({self.per_sheet} per sheet at {self.sheet_cost:g})"
                ]
                if self.material_cost
                else []
            )
            + [f"  each at 100  {self.unit_cost(100):10.2f}"]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process": self.process.to_dict(),
            "thickness": self.thickness,
            "cut_length": self.cut_length,
            "pierces": self.pierces,
            "rapid_length": self.rapid_length,
            "seconds": {
                "cutting": self.cut_seconds,
                "piercing": self.pierce_seconds,
                "positioning": self.rapid_seconds,
                "setup": self.setup_seconds,
                "cycle": self.cycle_seconds,
                "total": self.total_seconds,
            },
            "part_area": self.part_area,
            "blank_area": self.blank_area,
            "utilisation": self.utilisation,
            "per_sheet": self.per_sheet,
            "cost": {
                "machine": self.machine_cost,
                "material": self.material_cost,
                "one": self.total_cost,
                "each_at_10": self.unit_cost(10),
                "each_at_100": self.unit_cost(100),
            },
        }


def quote(
    face: Face2D,
    proc: Process,
    thickness: float,
    sheet_cost: float = 0.0,
    per_sheet: Optional[int] = None,
) -> Quote:
    """Cost one part on one machine.

    Supply ``sheet_cost`` to have material costed as well; the sheet's
    yield is estimated by :mod:`blueprint23d.nesting` unless ``per_sheet``
    is given. Without a sheet price nothing is guessed -- the material line
    is simply absent.
    """
    loops = list(face.loops())
    cut_length = sum(loop.length() for loop in loops)
    starts = [loop.curves[0].start for loop in loops if loop.curves]
    rapid_length = _nearest_neighbour_travel(starts)

    x0, y0, x1, y1 = face.bounds()
    blank = (x1 - x0) * (y1 - y0)

    if per_sheet is None and sheet_cost:
        from .nesting import estimate_for

        per_sheet = estimate_for(face, proc).per_sheet

    return Quote(
        process=proc,
        thickness=thickness,
        cut_length=cut_length,
        pierces=len(starts),
        rapid_length=rapid_length,
        cut_seconds=proc.cut_seconds(cut_length, thickness),
        pierce_seconds=len(starts) * proc.pierce_seconds,
        rapid_seconds=rapid_length / proc.rapid_rate * SECONDS_PER_MINUTE,
        setup_seconds=proc.setup_seconds,
        part_area=face.area(),
        blank_area=blank,
        per_sheet=per_sheet or 0,
        sheet_cost=sheet_cost,
    )


def compare(face: Face2D, thickness: float, names: Sequence[str] = ()) -> List[Tuple[Quote, Report]]:
    """Quote and check the same part on several processes at once.

    The point of the exercise: the cheapest machine that can actually make
    the part is frequently not the cheapest machine.
    """
    keys = list(names) if names else sorted(PROCESSES)
    results = []
    for key in keys:
        proc = process(key)
        results.append((quote(face, proc, thickness), check_manufacturability(face, proc, thickness)))
    results.sort(key=lambda pair: (not pair[1].cuttable, pair[0].machine_cost))
    return results


def describe_comparison(results: Sequence[Tuple[Quote, Report]], quantity: int = 1) -> str:
    lines = [
        f"{'process':<34}{'cycle':>10}{'each':>10}  verdict",
    ]
    for quote_, report in results:
        blockers = report.at_least(Severity.ERROR)
        verdict = "OK" if not blockers else f"{len(blockers)} blocker(s): " + blockers[0].code.value
        lines.append(
            f"{quote_.process.name:<34}{quote_.cycle_seconds:9.1f}s"
            f"{quote_.unit_cost(quantity):10.2f}  {verdict}"
        )
    return "\n".join(lines)


def quote_drawing(
    path: Union[str, Path],
    process_name: str,
    thickness: float,
    layer: Optional[str] = None,
    sheet_cost: float = 0.0,
) -> Tuple[Optional[Quote], Report]:
    """Read a drawing, check it, and cost it."""
    from .loaders import load_faces

    proc = process(process_name)
    report = Report(source=str(path))
    try:
        faces, _ = load_faces(path, layer=layer)
    except Exception as exc:
        report.add(Code.FILE_UNREADABLE, Severity.CRITICAL, str(exc))
        return None, report

    if not faces:
        report.add(Code.NO_CLOSED_PROFILE, Severity.CRITICAL, "no closed profile to cut")
        return None, report

    face = max(faces, key=lambda f: f.area())
    check_manufacturability(face, proc, thickness, report=report)
    return quote(face, proc, thickness, sheet_cost=sheet_cost), report
