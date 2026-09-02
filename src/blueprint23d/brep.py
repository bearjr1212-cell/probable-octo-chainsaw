"""Boundary representation: exact topology over the exact curves.

A mesh records *where the surface went*; a B-rep records *what the surface
is*. That difference is the whole point of this layer. When a profile with
a circular hole is extruded here, the hole's wall becomes a
:class:`CylindricalSurface` carrying the exact nominal radius -- so the
STEP file says "12mm bore", a CAM system generates a boring cycle for it,
and a metrology report can compare against a real cylinder. Tessellate
first and all of that is gone: the file says "3,000 triangles that happen
to lie near a cylinder", and the nominal diameter is unrecoverable.

Two properties are checked rather than assumed:

**Areas are closed-form.** Green's theorem turns the area of a loop into
:math:`\\oint (x\\,dy - y\\,dx)/2`, and that integral has an exact
elementary antiderivative for lines, circular arcs, and elliptical arcs.
A disc of radius r therefore reports area exactly ``pi*r**2``, not the
area of a polygon inscribed in it. Only free-form spline edges fall back
to quadrature.

**Topology is validated by the generalised Euler-Poincare formula**,

.. math:: V - E + 2F - L - 2S + 2G = 0

with V vertices, E edges, F faces, L loops, S shells and G genus. The
familiar :math:`V - E + F = 2` is the special case where every face is a
disc; it fails for exactly the parts this tool builds, since a plate with
a through-hole has an annular cap face and genus 1. Checking the general
form catches malformed solids that a triangle-soup pipeline would happily
write out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from .curves import (
    Arc2D,
    Curve2D,
    DEFAULT_CHORD_TOLERANCE,
    EllipseArc2D,
    Line2D,
    _GL20_NODES,
    _GL20_WEIGHTS,
)
from .exact import point_in_polygon

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]

_TAU = 2.0 * math.pi

#: Tolerance for deciding that one curve's end meets the next curve's start.
DEFAULT_JOIN_TOLERANCE = 1e-7


# --------------------------------------------------------------------------
# Exact area integrands (Green's theorem)
# --------------------------------------------------------------------------


def _line_area_term(curve: Line2D) -> float:
    (x0, y0), (x1, y1) = curve.p0, curve.p1
    return 0.5 * (x0 * y1 - x1 * y0)


def _arc_area_term(curve: Arc2D) -> float:
    """Exact contribution of a circular arc to the enclosed area.

    With :math:`x = c_x + r\\cos\\theta`, :math:`y = c_y + r\\sin\\theta`,
    the integrand collapses to
    :math:`c_x r\\cos\\theta + c_y r\\sin\\theta + r^2`, whose integral is
    elementary. A full circle therefore contributes exactly
    :math:`\\pi r^2`.
    """
    cx, cy = curve.center
    r = curve.radius
    sweep = curve.sweep

    if abs(abs(sweep) - _TAU) < 1e-12:
        # Over a closed turn the two center-dependent terms integrate to
        # zero analytically. Evaluating them numerically instead would fold
        # in sin(2*pi) = -2.4e-16 scaled by the center offset, so a disc
        # drawn far from the origin would come out a few ULP wrong purely
        # because of where it sits. Return the closed form directly.
        return math.copysign(math.pi * r * r, sweep)

    t0 = curve.start_angle
    t1 = t0 + sweep
    return 0.5 * (cx * r * (math.sin(t1) - math.sin(t0)) - cy * r * (math.cos(t1) - math.cos(t0)) + r * r * sweep)


def _ellipse_area_term(curve: EllipseArc2D) -> float:
    """Exact contribution of an elliptical arc.

    The cross terms cancel and the :math:`\\cos^2 + \\sin^2` terms combine
    into the constant cross product of the two semi-axis vectors, so a
    full ellipse contributes exactly :math:`\\pi a b`.
    """
    cx, cy = curve.center
    mx, my = curve.major
    nx, ny = -my * curve.ratio, mx * curve.ratio
    cross = mx * ny - my * nx
    sweep = curve.sweep

    if abs(abs(sweep) - _TAU) < 1e-12:
        # Same closed-loop argument as the circular case: the center terms
        # cancel over a full turn, so use the closed form rather than
        # numerically cancelling them.
        return math.copysign(math.pi * abs(cross), sweep)

    t0 = curve.start_param
    t1 = t0 + sweep
    return 0.5 * (
        -(math.cos(t1) - math.cos(t0)) * (cy * mx - cx * my)
        + (math.sin(t1) - math.sin(t0)) * (cx * ny - cy * nx)
        + cross * sweep
    )


def _quadrature_area_term(curve: Curve2D, subdivisions: int = 24) -> float:
    """Fallback for free-form curves: Gauss-Legendre on ``(x y' - y x')/2``."""
    total = 0.0
    step = 1.0 / subdivisions
    for s in range(subdivisions):
        a = s * step
        half = 0.5 * step
        mid = a + half
        acc = 0.0
        for node, weight in zip(_GL20_NODES, _GL20_WEIGHTS):
            t = mid + half * node
            x, y = curve.point(t)
            dx, dy = curve.derivative(t)
            acc += weight * (x * dy - y * dx)
        total += acc * half
    return 0.5 * total


def curve_area_term(curve: Curve2D) -> float:
    """Signed area contribution of one curve, exact wherever a closed form exists."""
    if isinstance(curve, Line2D):
        return _line_area_term(curve)
    if isinstance(curve, Arc2D):
        return _arc_area_term(curve)
    if isinstance(curve, EllipseArc2D):
        return _ellipse_area_term(curve)
    return _quadrature_area_term(curve)


def has_exact_area(curve: Curve2D) -> bool:
    return isinstance(curve, (Line2D, Arc2D, EllipseArc2D))


# --------------------------------------------------------------------------
# 2D topology
# --------------------------------------------------------------------------


class Loop2D:
    """A closed chain of curves bounding a region.

    The chain is checked for continuity on construction: each curve must
    start where the previous one ended, and the last must close back onto
    the first. A loop that does not close is not a boundary, and letting
    one through here produces failures much further downstream where the
    cause is unrecognisable.
    """

    __slots__ = ("curves", "_tolerance")

    def __init__(self, curves: Sequence[Curve2D], tolerance: float = DEFAULT_JOIN_TOLERANCE):
        if not curves:
            raise ValueError("a loop needs at least one curve")
        self.curves: Tuple[Curve2D, ...] = tuple(curves)
        self._tolerance = tolerance
        self._validate()

    def _validate(self) -> None:
        n = len(self.curves)
        for i in range(n):
            end = self.curves[i].end
            start = self.curves[(i + 1) % n].start
            gap = math.hypot(end[0] - start[0], end[1] - start[1])
            if gap > self._tolerance:
                raise ValueError(
                    f"loop is not continuous: curve {i} ends at {end} but curve "
                    f"{(i + 1) % n} starts at {start} (gap {gap:.3e} > {self._tolerance:.3e})"
                )

    def __len__(self) -> int:
        return len(self.curves)

    def __iter__(self):
        return iter(self.curves)

    def __repr__(self) -> str:
        kinds = ", ".join(type(c).__name__ for c in self.curves)
        return f"Loop2D({len(self.curves)} curves: {kinds})"

    def signed_area(self) -> float:
        """Signed area by Green's theorem. Positive means counter-clockwise."""
        return sum(curve_area_term(curve) for curve in self.curves)

    def area(self) -> float:
        return abs(self.signed_area())

    def is_area_exact(self) -> bool:
        """True when no edge needed quadrature."""
        return all(has_exact_area(curve) for curve in self.curves)

    def orientation(self) -> int:
        area = self.signed_area()
        return (area > 0.0) - (area < 0.0)

    def reverse(self) -> "Loop2D":
        return Loop2D(tuple(reversed([c.reverse() for c in self.curves])), self._tolerance)

    def oriented(self, counter_clockwise: bool) -> "Loop2D":
        want = 1 if counter_clockwise else -1
        return self if self.orientation() == want else self.reverse()

    def bounds(self) -> Tuple[float, float, float, float]:
        boxes = [curve.bounds() for curve in self.curves]
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )

    def length(self) -> float:
        return sum(curve.length() for curve in self.curves)

    def tessellate(self, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> List[Point2]:
        """Closed polyline approximating the loop, first point not repeated."""
        points: List[Point2] = []
        for curve in self.curves:
            segment = curve.tessellate(tolerance)
            points.extend(segment[:-1])  # next curve contributes the joint
        return points

    def contains_point(self, point: Point2, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> int:
        """``+1`` inside, ``-1`` outside, ``0`` on the tessellated boundary.

        Decided on the tessellation with exact predicates, so the answer is
        exactly right for the polyline; it can differ from the true curved
        region only within ``tolerance`` of the boundary.
        """
        return point_in_polygon(point, self.tessellate(tolerance))

    def deviation_bound(self, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> float:
        return max((c.deviation_certificate(tolerance).bound for c in self.curves), default=0.0)


@dataclass(frozen=True)
class Face2D:
    """A planar region: one outer loop, any number of hole loops.

    Loops are normalised so the outer runs counter-clockwise and holes run
    clockwise, which makes the signed areas add up correctly and gives
    every derived normal a consistent direction.
    """

    outer: Loop2D
    inners: Tuple[Loop2D, ...] = ()

    @classmethod
    def create(cls, outer: Loop2D, inners: Sequence[Loop2D] = ()) -> "Face2D":
        return cls(outer.oriented(True), tuple(loop.oriented(False) for loop in inners))

    def area(self) -> float:
        """Exact for line/arc/ellipse boundaries: outer area less the holes."""
        return abs(self.outer.signed_area()) - sum(abs(loop.signed_area()) for loop in self.inners)

    def is_area_exact(self) -> bool:
        return self.outer.is_area_exact() and all(loop.is_area_exact() for loop in self.inners)

    def bounds(self) -> Tuple[float, float, float, float]:
        return self.outer.bounds()

    def loops(self) -> Tuple[Loop2D, ...]:
        return (self.outer,) + self.inners

    def tessellate(self, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> Tuple[List[Point2], List[List[Point2]]]:
        return (self.outer.tessellate(tolerance), [loop.tessellate(tolerance) for loop in self.inners])

    def contains_point(self, point: Point2, tolerance: float = DEFAULT_CHORD_TOLERANCE) -> int:
        if self.outer.contains_point(point, tolerance) <= 0:
            return self.outer.contains_point(point, tolerance)
        for loop in self.inners:
            inside_hole = loop.contains_point(point, tolerance)
            if inside_hole > 0:
                return -1
            if inside_hole == 0:
                return 0
        return 1


# --------------------------------------------------------------------------
# Surfaces
# --------------------------------------------------------------------------


class Surface:
    """Marker base class for the analytic surfaces a solid can carry."""


@dataclass(frozen=True)
class PlaneSurface(Surface):
    origin: Point3
    normal: Point3
    ref_direction: Point3


@dataclass(frozen=True)
class CylindricalSurface(Surface):
    """A true cylinder: what an extruded circular arc becomes.

    Carrying this instead of a fan of quads is the difference between a
    STEP file that names a 12mm bore and one that merely contains
    triangles near where a 12mm bore should be.
    """

    origin: Point3  # a point on the axis
    axis: Point3  # axis direction (unit)
    ref_direction: Point3  # reference direction for the angular parameter
    radius: float


@dataclass(frozen=True)
class LinearExtrusionSurface(Surface):
    """A general swept surface: an arbitrary profile curve dragged along a vector.

    Free-form and elliptical edges land here, which keeps them exact
    rather than forcing an approximating spline surface.
    """

    curve: Curve2D
    base_z: float
    direction: Point3


# --------------------------------------------------------------------------
# 3D topology
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Vertex3:
    position: Point3


@dataclass(frozen=True)
class EmbeddedCurve3:
    """A 2D curve lifted into the plane ``z = height``."""

    curve: Curve2D
    height: float

    def point(self, t: float) -> Point3:
        x, y = self.curve.point(t)
        return (x, y, self.height)

    @property
    def start(self) -> Point3:
        return self.point(0.0)

    @property
    def end(self) -> Point3:
        return self.point(1.0)


@dataclass(frozen=True)
class Line3:
    p0: Point3
    p1: Point3

    def point(self, t: float) -> Point3:
        return tuple(a + t * (b - a) for a, b in zip(self.p0, self.p1))  # type: ignore[return-value]

    @property
    def start(self) -> Point3:
        return self.p0

    @property
    def end(self) -> Point3:
        return self.p1


@dataclass(frozen=True)
class Edge3:
    """A bounded piece of a 3D curve between two vertices."""

    geometry: object  # EmbeddedCurve3 | Line3
    start_vertex: Vertex3
    end_vertex: Vertex3


@dataclass(frozen=True)
class OrientedEdge:
    edge: Edge3
    forward: bool


@dataclass(frozen=True)
class EdgeLoop:
    edges: Tuple[OrientedEdge, ...]


@dataclass(frozen=True)
class Face3:
    """A trimmed patch of an analytic surface."""

    surface: Surface
    outer_loop: EdgeLoop
    inner_loops: Tuple[EdgeLoop, ...] = ()
    same_sense: bool = True

    def loop_count(self) -> int:
        return 1 + len(self.inner_loops)


@dataclass
class Shell:
    faces: List[Face3] = field(default_factory=list)


@dataclass
class Solid:
    """A closed manifold solid, with the data needed to certify it."""

    shell: Shell
    genus: int = 0
    name: str = "part"

    # ---- topology accounting -------------------------------------------

    def faces(self) -> List[Face3]:
        return self.shell.faces

    def unique_edges(self) -> List[Edge3]:
        seen = {}
        for face in self.faces():
            for loop in (face.outer_loop,) + face.inner_loops:
                for oriented in loop.edges:
                    seen[id(oriented.edge)] = oriented.edge
        return list(seen.values())

    def unique_vertices(self) -> List[Vertex3]:
        seen = {}
        for edge in self.unique_edges():
            seen[id(edge.start_vertex)] = edge.start_vertex
            seen[id(edge.end_vertex)] = edge.end_vertex
        return list(seen.values())

    def loop_count(self) -> int:
        return sum(face.loop_count() for face in self.faces())

    def euler_poincare_defect(self, shells: int = 1) -> int:
        """``V - E + 2F - L - 2S + 2G``; zero for a valid solid.

        The plain ``V - E + F = 2`` is not applicable here: it assumes
        every face is a topological disc, and an extruded plate with a
        through-hole has annular caps and genus 1. The generalised form
        below handles multiply-connected faces and non-zero genus, which
        is exactly the class of part this tool produces.
        """
        v = len(self.unique_vertices())
        e = len(self.unique_edges())
        f = len(self.faces())
        loops = self.loop_count()
        return v - e + 2 * f - loops - 2 * shells + 2 * self.genus

    def is_topologically_valid(self, shells: int = 1) -> bool:
        return self.euler_poincare_defect(shells) == 0

    def every_edge_used_twice(self) -> bool:
        """Each edge must appear in exactly two faces for the shell to close."""
        counts: dict = {}
        for face in self.faces():
            for loop in (face.outer_loop,) + face.inner_loops:
                for oriented in loop.edges:
                    counts[id(oriented.edge)] = counts.get(id(oriented.edge), 0) + 1
        return all(count == 2 for count in counts.values())

    def validate(self, shells: int = 1) -> List[str]:
        """Return a list of problems; empty means the solid checks out."""
        problems = []
        if not self.every_edge_used_twice():
            problems.append("shell is not closed: some edge is not shared by exactly two faces")
        defect = self.euler_poincare_defect(shells)
        if defect != 0:
            problems.append(
                f"Euler-Poincare defect {defect} (V={len(self.unique_vertices())}, "
                f"E={len(self.unique_edges())}, F={len(self.faces())}, "
                f"L={self.loop_count()}, S={shells}, G={self.genus})"
            )
        return problems


# --------------------------------------------------------------------------
# Extrusion
# --------------------------------------------------------------------------


def _lateral_surface(curve: Curve2D, base_z: float, direction: Point3) -> Surface:
    """Pick the exact surface type an extruded edge sweeps out."""
    if isinstance(curve, Line2D):
        # A swept straight edge is planar; the normal is the in-plane edge
        # normal, since the sweep direction lies in the face.
        (x0, y0), (x1, y1) = curve.p0, curve.p1
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy)
        normal = (dy / norm, -dx / norm, 0.0) if norm else (1.0, 0.0, 0.0)
        return PlaneSurface(
            origin=(x0, y0, base_z),
            normal=normal,
            ref_direction=(dx / norm, dy / norm, 0.0) if norm else (0.0, 1.0, 0.0),
        )
    if isinstance(curve, Arc2D):
        # The exact reason this layer exists: a swept arc is a cylinder of
        # the arc's own radius, carried through to export unchanged.
        return CylindricalSurface(
            origin=(curve.center[0], curve.center[1], base_z),
            axis=direction,
            ref_direction=(1.0, 0.0, 0.0),
            radius=curve.radius,
        )
    return LinearExtrusionSurface(curve=curve, base_z=base_z, direction=direction)


def extrude_face(
    face: Face2D,
    depth: float,
    base_z: float = 0.0,
    name: str = "part",
) -> Solid:
    """Extrude a planar face along +Z into a closed, validated solid.

    Every boundary curve keeps its identity: line edges sweep planes, arc
    edges sweep cylinders, everything else sweeps an exact surface of
    linear extrusion. The genus is set from the hole count, so the
    Euler-Poincare check below is a real test rather than a tautology.
    """
    if depth <= 0.0:
        raise ValueError(f"extrusion depth must be positive, got {depth}")

    top_z = base_z + depth
    direction: Point3 = (0.0, 0.0, 1.0)
    faces: List[Face3] = []

    # Per loop: bottom edges, top edges, vertical edges, and one lateral
    # face per boundary curve.
    bottom_loops: List[EdgeLoop] = []
    top_loops: List[EdgeLoop] = []

    for loop_index, loop in enumerate(face.loops()):
        n = len(loop)
        bottom_vertices = [Vertex3((c.start[0], c.start[1], base_z)) for c in loop.curves]
        top_vertices = [Vertex3((c.start[0], c.start[1], top_z)) for c in loop.curves]

        bottom_edges: List[Edge3] = []
        top_edges: List[Edge3] = []
        vertical_edges: List[Edge3] = []

        for i, curve in enumerate(loop.curves):
            nxt = (i + 1) % n
            bottom_edges.append(
                Edge3(EmbeddedCurve3(curve, base_z), bottom_vertices[i], bottom_vertices[nxt])
            )
            top_edges.append(Edge3(EmbeddedCurve3(curve, top_z), top_vertices[i], top_vertices[nxt]))
            vertical_edges.append(
                Edge3(
                    Line3(bottom_vertices[i].position, top_vertices[i].position),
                    bottom_vertices[i],
                    top_vertices[i],
                )
            )

        for i, curve in enumerate(loop.curves):
            nxt = (i + 1) % n
            # Lateral face walks: bottom edge forward, next vertical up,
            # top edge backward, this vertical down.
            lateral_loop = EdgeLoop(
                (
                    OrientedEdge(bottom_edges[i], True),
                    OrientedEdge(vertical_edges[nxt], True),
                    OrientedEdge(top_edges[i], False),
                    OrientedEdge(vertical_edges[i], False),
                )
            )
            faces.append(
                Face3(
                    surface=_lateral_surface(curve, base_z, direction),
                    outer_loop=lateral_loop,
                    same_sense=(loop_index == 0),
                )
            )

        bottom_loops.append(EdgeLoop(tuple(OrientedEdge(e, True) for e in bottom_edges)))
        top_loops.append(EdgeLoop(tuple(OrientedEdge(e, True) for e in top_edges)))

    bottom_face = Face3(
        surface=PlaneSurface((0.0, 0.0, base_z), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
        outer_loop=bottom_loops[0],
        inner_loops=tuple(bottom_loops[1:]),
        same_sense=False,
    )
    top_face = Face3(
        surface=PlaneSurface((0.0, 0.0, top_z), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        outer_loop=top_loops[0],
        inner_loops=tuple(top_loops[1:]),
        same_sense=True,
    )
    faces.extend((bottom_face, top_face))

    # Each hole drills a handle through the plate, adding one to the genus.
    return Solid(shell=Shell(faces), genus=len(face.inners), name=name)


def solid_volume(face: Face2D, depth: float) -> float:
    """Exact extruded volume: closed-form face area times depth.

    Not a mesh volume -- no tessellation is involved, so a disc extrusion
    reports exactly ``pi*r**2*h``.
    """
    return face.area() * depth
