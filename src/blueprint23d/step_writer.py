"""STEP AP214 export (ISO 10303-21).

This is the payoff for everything upstream. A mesh format can only say
"here are triangles"; STEP says *what the geometry is*. A hole exported
here is a ``CYLINDRICAL_SURFACE`` with a radius field holding the nominal
value, bounded by ``CIRCLE`` edges holding the same radius. Opened in a CAM
system it is a hole, with a diameter that can be measured, dimensioned,
and matched to a drill; opened as STL it is a few hundred triangles that
nobody can turn back into a diameter.

The mapping from the B-rep is direct, because the B-rep was built to
preserve exactly what STEP wants to record:

===========================  ==================================
:mod:`brep` construct        STEP entity
===========================  ==================================
``Line2D`` edge              ``LINE`` + ``VECTOR``
``Arc2D`` edge               ``CIRCLE``
``EllipseArc2D`` edge        ``ELLIPSE``
``BezierCurve2D`` /          ``B_SPLINE_CURVE_WITH_KNOTS``
``NurbsCurve2D`` edge        (rational variant when weighted)
``PlaneSurface``             ``PLANE``
``CylindricalSurface``       ``CYLINDRICAL_SURFACE``
``LinearExtrusionSurface``   ``SURFACE_OF_LINEAR_EXTRUSION``
``Solid``                    ``MANIFOLD_SOLID_BREP`` / ``CLOSED_SHELL``
===========================  ==================================

No CAD kernel is available here to open the result, so correctness is
established structurally instead: :func:`validate_step` re-parses the
emitted file and checks that every entity reference resolves, that no
entity is defined twice, that the product/context scaffolding AP214
requires is present and reachable, and that the shell references only
faces. That catches the failure modes a writer actually has -- dangling
references and missing scaffolding -- rather than assuming them away.
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .brep import (
    CylindricalSurface,
    Edge3,
    EdgeLoop,
    EmbeddedCurve3,
    Face3,
    Line3,
    LinearExtrusionSurface,
    PlaneSurface,
    Solid,
    Vertex3,
)
from .curves import Arc2D, BezierCurve2D, Curve2D, EllipseArc2D, Line2D, NurbsCurve2D

Point3 = Tuple[float, float, float]

#: Length units expressible directly as an SI unit in the header.
SI_LENGTH_UNITS = {
    "MM": ".MILLI.",
    "M": "$",
    "CM": ".CENTI.",
}


def _fmt(value: float) -> str:
    """STEP real literal: always with an exponent or a decimal point.

    ISO 10303-21 reals must be distinguishable from integers, so a bare
    ``5`` is invalid where ``5.`` is required. Repr is used for the digits
    so no precision is thrown away in the file -- the whole point of this
    exporter is that the numbers survive.
    """
    if value == 0.0:
        return "0."
    text = repr(float(value))
    if "e" in text or "E" in text:
        mantissa, exponent = text.split("e")
        if "." not in mantissa:
            mantissa += "."
        return f"{mantissa}E{int(exponent)}"
    if "." not in text:
        text += "."
    return text


def _fmt_point(p: Sequence[float]) -> str:
    return "(" + ",".join(_fmt(v) for v in p) + ")"


def _normalize(v: Point3) -> Point3:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n == 0.0:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _perpendicular(axis: Point3) -> Point3:
    """Any unit vector orthogonal to ``axis``, chosen stably."""
    ax, ay, az = axis
    other = (1.0, 0.0, 0.0) if abs(ax) < 0.9 else (0.0, 1.0, 0.0)
    cross = (
        ay * other[2] - az * other[1],
        az * other[0] - ax * other[2],
        ax * other[1] - ay * other[0],
    )
    return _normalize(cross)


class StepWriter:
    """Accumulates STEP entities and renders a complete Part 21 file."""

    def __init__(self, name: str = "part", units: str = "MM", uncertainty: float = 1e-7):
        if units.upper() not in SI_LENGTH_UNITS:
            raise ValueError(f"unsupported unit {units!r}; expected one of {sorted(SI_LENGTH_UNITS)}")
        self.name = name
        self.units = units.upper()
        self.uncertainty = uncertainty
        self._lines: List[str] = []
        self._next_id = 1
        # Identical geometry is emitted once and shared; CAD files are
        # dominated by repeated points and directions.
        self._cache: Dict[str, int] = {}

    def add(self, body: str, cache_key: Optional[str] = None) -> int:
        if cache_key is not None:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return hit
        entity_id = self._next_id
        self._next_id += 1
        self._lines.append(f"#{entity_id} = {body};")
        if cache_key is not None:
            self._cache[cache_key] = entity_id
        return entity_id

    # ---- geometry primitives -------------------------------------------

    def point(self, p: Sequence[float]) -> int:
        key = f"P{_fmt_point(p)}"
        return self.add(f"CARTESIAN_POINT('',{_fmt_point(p)})", key)

    def direction(self, d: Sequence[float]) -> int:
        key = f"D{_fmt_point(d)}"
        return self.add(f"DIRECTION('',{_fmt_point(d)})", key)

    def vector(self, d: Sequence[float], magnitude: float) -> int:
        return self.add(f"VECTOR('',#{self.direction(d)},{_fmt(magnitude)})")

    def axis2_placement(self, origin: Sequence[float], axis: Point3, ref: Point3) -> int:
        return self.add(
            f"AXIS2_PLACEMENT_3D('',#{self.point(origin)},"
            f"#{self.direction(axis)},#{self.direction(ref)})"
        )

    def vertex_point(self, position: Sequence[float]) -> int:
        return self.add(f"VERTEX_POINT('',#{self.point(position)})")

    # ---- curves ---------------------------------------------------------

    def curve_for_edge(self, geometry: object) -> int:
        """Emit the exact STEP curve for one edge's geometry."""
        if isinstance(geometry, Line3):
            return self._line(geometry.p0, geometry.p1)
        if isinstance(geometry, EmbeddedCurve3):
            return self._curve2d(geometry.curve, geometry.height)
        raise TypeError(f"cannot export edge geometry of type {type(geometry).__name__}")

    def _line(self, p0: Sequence[float], p1: Sequence[float]) -> int:
        d = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        length = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        return self.add(f"LINE('',#{self.point(p0)},#{self.vector(_normalize(d), length)})")

    def _curve2d(self, curve: Curve2D, z: float) -> int:
        if isinstance(curve, Line2D):
            return self._line((curve.p0[0], curve.p0[1], z), (curve.p1[0], curve.p1[1], z))

        if isinstance(curve, Arc2D):
            # The exact radius reaches the file untouched: this is the
            # entity a CAM system reads to recognise a bore.
            placement = self.axis2_placement(
                (curve.center[0], curve.center[1], z),
                (0.0, 0.0, 1.0) if curve.ccw else (0.0, 0.0, -1.0),
                (1.0, 0.0, 0.0),
            )
            return self.add(f"CIRCLE('',#{placement},{_fmt(curve.radius)})")

        if isinstance(curve, EllipseArc2D):
            major = _normalize((curve.major[0], curve.major[1], 0.0))
            placement = self.axis2_placement(
                (curve.center[0], curve.center[1], z), (0.0, 0.0, 1.0), major
            )
            return self.add(
                f"ELLIPSE('',#{placement},{_fmt(curve.major_length)},{_fmt(curve.minor_length)})"
            )

        if isinstance(curve, BezierCurve2D):
            degree = curve.degree
            knots = (0.0, 1.0)
            multiplicities = (degree + 1, degree + 1)
            return self._bspline(
                [(p[0], p[1], z) for p in curve.control],
                degree,
                knots,
                multiplicities,
                curve.weights,
            )

        if isinstance(curve, NurbsCurve2D):
            distinct: List[float] = []
            multiplicities: List[int] = []
            for k in curve.knots:
                if distinct and k == distinct[-1]:
                    multiplicities[-1] += 1
                else:
                    distinct.append(k)
                    multiplicities.append(1)
            return self._bspline(
                [(p[0], p[1], z) for p in curve.control],
                curve.degree,
                tuple(distinct),
                tuple(multiplicities),
                curve.weights,
            )

        raise TypeError(f"cannot export curve of type {type(curve).__name__}")

    def _bspline(
        self,
        control: Sequence[Point3],
        degree: int,
        knots: Sequence[float],
        multiplicities: Sequence[int],
        weights: Optional[Sequence[float]],
    ) -> int:
        points = ",".join(f"#{self.point(p)}" for p in control)
        knot_text = ",".join(_fmt(k) for k in knots)
        mult_text = ",".join(str(m) for m in multiplicities)
        rational = weights is not None and any(w != weights[0] for w in weights)

        if not rational:
            return self.add(
                f"B_SPLINE_CURVE_WITH_KNOTS('',{degree},({points}),.UNSPECIFIED.,"
                f".F.,.F.,({mult_text}),({knot_text}),.UNSPECIFIED.)"
            )

        # A rational B-spline is expressed in STEP as an ANDOR combination
        # of the plain and rational subtypes, which is how the standard
        # carries the weights without a separate entity.
        weight_text = ",".join(_fmt(w) for w in weights)
        return self.add(
            "( BOUNDED_CURVE() B_SPLINE_CURVE(%d,(%s),.UNSPECIFIED.,.F.,.F.)"
            " B_SPLINE_CURVE_WITH_KNOTS((%s),(%s),.UNSPECIFIED.)"
            " CURVE() GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_CURVE((%s))"
            " REPRESENTATION_ITEM('') )" % (degree, points, mult_text, knot_text, weight_text)
        )

    # ---- surfaces -------------------------------------------------------

    def surface(self, surface: object) -> int:
        if isinstance(surface, PlaneSurface):
            axis = _normalize(surface.normal)
            ref = surface.ref_direction
            if abs(ref[0] * axis[0] + ref[1] * axis[1] + ref[2] * axis[2]) > 1e-9:
                ref = _perpendicular(axis)
            return self.add(f"PLANE('',#{self.axis2_placement(surface.origin, axis, _normalize(ref))})")

        if isinstance(surface, CylindricalSurface):
            axis = _normalize(surface.axis)
            ref = _perpendicular(axis)
            placement = self.axis2_placement(surface.origin, axis, ref)
            return self.add(f"CYLINDRICAL_SURFACE('',#{placement},{_fmt(surface.radius)})")

        if isinstance(surface, LinearExtrusionSurface):
            swept = self._curve2d(surface.curve, surface.base_z)
            direction = _normalize(surface.direction)
            return self.add(
                f"SURFACE_OF_LINEAR_EXTRUSION('',#{swept},#{self.vector(direction, 1.0)})"
            )

        raise TypeError(f"cannot export surface of type {type(surface).__name__}")

    # ---- topology -------------------------------------------------------

    def build_solid(self, solid: Solid) -> int:
        vertex_ids: Dict[int, int] = {}
        edge_ids: Dict[int, int] = {}

        def vertex_id(vertex: Vertex3) -> int:
            key = id(vertex)
            if key not in vertex_ids:
                vertex_ids[key] = self.vertex_point(vertex.position)
            return vertex_ids[key]

        def edge_id(edge: Edge3) -> int:
            key = id(edge)
            if key not in edge_ids:
                curve = self.curve_for_edge(edge.geometry)
                edge_ids[key] = self.add(
                    f"EDGE_CURVE('',#{vertex_id(edge.start_vertex)},"
                    f"#{vertex_id(edge.end_vertex)},#{curve},.T.)"
                )
            return edge_ids[key]

        def edge_loop_id(loop: EdgeLoop) -> int:
            oriented = [
                self.add(f"ORIENTED_EDGE('',*,*,#{edge_id(o.edge)},{'.T.' if o.forward else '.F.'})")
                for o in loop.edges
            ]
            return self.add("EDGE_LOOP('',(" + ",".join(f"#{o}" for o in oriented) + "))")

        face_ids: List[int] = []
        for face in solid.faces():
            surface_id = self.surface(face.surface)
            bounds = [self.add(f"FACE_OUTER_BOUND('',#{edge_loop_id(face.outer_loop)},.T.)")]
            for inner in face.inner_loops:
                bounds.append(self.add(f"FACE_BOUND('',#{edge_loop_id(inner)},.T.)"))
            bound_text = ",".join(f"#{b}" for b in bounds)
            face_ids.append(
                self.add(
                    f"ADVANCED_FACE('',({bound_text}),#{surface_id},"
                    f"{'.T.' if face.same_sense else '.F.'})"
                )
            )

        shell = self.add("CLOSED_SHELL('',(" + ",".join(f"#{f}" for f in face_ids) + "))")
        return self.add(f"MANIFOLD_SOLID_BREP('{solid.name}',#{shell})")

    # ---- file assembly ---------------------------------------------------

    def render(self, solid: Solid) -> str:
        """Emit the complete Part 21 file for ``solid``."""
        brep_id = self.build_solid(solid)

        length_unit = self.add(
            f"( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT({SI_LENGTH_UNITS[self.units]},.METRE.) )"
        )
        angle_unit = self.add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
        solid_angle_unit = self.add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
        uncertainty = self.add(
            f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE({_fmt(self.uncertainty)}),"
            f"#{length_unit},'distance_accuracy_value','confusion accuracy')"
        )
        context = self.add(
            f"( GEOMETRIC_REPRESENTATION_CONTEXT(3) "
            f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncertainty})) "
            f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length_unit},#{angle_unit},#{solid_angle_unit})) "
            f"REPRESENTATION_CONTEXT('Context','3D') )"
        )

        origin = self.axis2_placement((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
        shape_rep = self.add(
            f"ADVANCED_BREP_SHAPE_REPRESENTATION('{self.name}',(#{origin},#{brep_id}),#{context})"
        )

        app_context = self.add("APPLICATION_CONTEXT('automotive design')")
        self.add(
            f"APPLICATION_PROTOCOL_DEFINITION('international standard',"
            f"'automotive_design',2000,#{app_context})"
        )
        product_context = self.add(f"PRODUCT_CONTEXT('',#{app_context},'mechanical')")
        product = self.add(f"PRODUCT('{self.name}','{self.name}','',(#{product_context}))")
        formation = self.add(
            f"PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE('','',#{product},.NOT_KNOWN.)"
        )
        definition_context = self.add(
            f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app_context},'design')"
        )
        definition = self.add(f"PRODUCT_DEFINITION('design','',#{formation},#{definition_context})")
        product_shape = self.add(f"PRODUCT_DEFINITION_SHAPE('','',#{definition})")
        self.add(f"SHAPE_DEFINITION_REPRESENTATION(#{product_shape},#{shape_rep})")

        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        header = (
            "ISO-10303-21;\n"
            "HEADER;\n"
            "FILE_DESCRIPTION(('exact B-rep from 2D blueprint'),'2;1');\n"
            f"FILE_NAME('{self.name}','{timestamp}',('blueprint23d'),(''),"
            "'blueprint23d','blueprint23d','');\n"
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 3 1 1 }'));\n"
            "ENDSEC;\n"
            "DATA;\n"
        )
        return header + "\n".join(self._lines) + "\nENDSEC;\nEND-ISO-10303-21;\n"


def write_step(
    solid: Solid,
    path: Union[str, Path],
    name: Optional[str] = None,
    units: str = "MM",
    uncertainty: float = 1e-7,
) -> Path:
    """Write ``solid`` to ``path`` as a STEP AP214 part file."""
    path = Path(path)
    writer = StepWriter(name=name or solid.name, units=units, uncertainty=uncertainty)
    text = writer.render(solid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# --------------------------------------------------------------------------
# Structural validation
# --------------------------------------------------------------------------


def parse_entities(text: str) -> Dict[int, str]:
    """Crude Part 21 reader: entity id -> body. Enough to check structure."""
    entities: Dict[int, str] = {}
    in_data = False
    buffer = ""
    for raw in text.splitlines():
        line = raw.strip()
        if line == "DATA;":
            in_data = True
            continue
        if line == "ENDSEC;" and in_data:
            break
        if not in_data:
            continue
        buffer += line
        while ";" in buffer:
            statement, buffer = buffer.split(";", 1)
            statement = statement.strip()
            if not statement.startswith("#"):
                continue
            ref, _, body = statement.partition("=")
            entity_id = int(ref.strip()[1:])
            if entity_id in entities:
                raise ValueError(f"entity #{entity_id} defined more than once")
            entities[entity_id] = body.strip()
    return entities


def validate_step(text: str) -> List[str]:
    """Structural checks on an emitted file; empty list means it checks out.

    Verifies the things a writer actually gets wrong: dangling references,
    duplicate ids, missing AP214 scaffolding, and a shell that does not
    point at faces.
    """
    problems: List[str] = []

    if not text.startswith("ISO-10303-21;"):
        problems.append("missing ISO-10303-21 header")
    if "END-ISO-10303-21;" not in text:
        problems.append("missing END-ISO-10303-21 terminator")
    if "FILE_SCHEMA" not in text:
        problems.append("missing FILE_SCHEMA")

    try:
        entities = parse_entities(text)
    except ValueError as exc:
        return problems + [str(exc)]

    if not entities:
        return problems + ["no entities in DATA section"]

    # Every referenced id must exist.
    import re

    for entity_id, body in entities.items():
        for ref in re.findall(r"#(\d+)", body):
            if int(ref) not in entities:
                problems.append(f"#{entity_id} references undefined #{ref}")

    required = (
        "MANIFOLD_SOLID_BREP",
        "CLOSED_SHELL",
        "ADVANCED_FACE",
        "ADVANCED_BREP_SHAPE_REPRESENTATION",
        "APPLICATION_CONTEXT",
        "PRODUCT_DEFINITION_SHAPE",
        "SHAPE_DEFINITION_REPRESENTATION",
        "GEOMETRIC_REPRESENTATION_CONTEXT",
    )
    joined = " ".join(entities.values())
    for keyword in required:
        if keyword not in joined:
            problems.append(f"missing required entity type {keyword}")

    # The shell must reference only faces.
    for entity_id, body in entities.items():
        if body.startswith("CLOSED_SHELL"):
            for ref in re.findall(r"#(\d+)", body):
                target = entities.get(int(ref), "")
                if not target.startswith("ADVANCED_FACE"):
                    problems.append(f"CLOSED_SHELL #{entity_id} references non-face #{ref}")

    # Every face must have a surface and at least one bound.
    for entity_id, body in entities.items():
        if body.startswith("ADVANCED_FACE"):
            refs = [int(r) for r in re.findall(r"#(\d+)", body)]
            kinds = [entities.get(r, "").split("(")[0] for r in refs]
            if not any(k in ("PLANE", "CYLINDRICAL_SURFACE", "SURFACE_OF_LINEAR_EXTRUSION") for k in kinds):
                problems.append(f"ADVANCED_FACE #{entity_id} has no recognised surface")
            if not any(k in ("FACE_OUTER_BOUND", "FACE_BOUND") for k in kinds):
                problems.append(f"ADVANCED_FACE #{entity_id} has no bound")

    return problems


def summarize_step(text: str) -> Dict[str, int]:
    """Count entities by type -- useful for asserting a hole really is a cylinder."""
    counts: Dict[str, int] = {}
    for body in parse_entities(text).values():
        name = body.split("(")[0].strip()
        if not name:
            # Bodies beginning with "(" are STEP complex entities -- the
            # ANDOR combinations used for units, contexts, and rational
            # B-splines -- which have no single leading type name.
            name = "COMPLEX"
        counts[name] = counts.get(name, 0) + 1
    return counts
