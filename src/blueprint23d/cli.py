"""Command-line interface.

The output extension decides how much precision survives, and the tool says
so rather than leaving it implicit:

* ``.step`` / ``.stp`` -- exact B-rep. Arcs stay arcs, holes become
  cylindrical surfaces carrying their nominal radius. Nothing is
  tessellated at any point.
* ``.stl`` / ``.obj`` / ``.glb`` / ``.ply`` -- triangle mesh, tessellated
  to ``--tolerance`` with a deviation bound that is printed with the
  result.

Examples::

    blueprint23d extrude --input plate.dxf --depth 8 --output plate.step
    blueprint23d extrude --input plate.dxf --depth 8 --output plate.stl --tolerance 1e-4
    blueprint23d multiview --top top.dxf --front front.dxf --output part.step
    blueprint23d inspect --input plate.dxf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .brep import extrude_face
from .curves import Arc2D, BezierCurve2D, EllipseArc2D, Line2D, NurbsCurve2D
from .loaders import load_face, load_faces
from .multiview import ViewSpec, reconstruct
from .step_writer import validate_step, write_step
from .tessellate import tessellate_extrusion

STEP_SUFFIXES = {".step", ".stp"}
MESH_SUFFIXES = {".stl", ".obj", ".glb", ".ply"}


def _describe_curve(curve) -> str:
    if isinstance(curve, Arc2D):
        return f"arc r={curve.radius:.6g}"
    if isinstance(curve, EllipseArc2D):
        return f"ellipse {curve.major_length:.6g}x{curve.minor_length:.6g}"
    if isinstance(curve, Line2D):
        return "line"
    if isinstance(curve, BezierCurve2D):
        return f"bezier deg {curve.degree}"
    if isinstance(curve, NurbsCurve2D):
        return f"nurbs deg {curve.degree}"
    return type(curve).__name__


def _curve_census(face) -> str:
    counts = {}
    for loop in face.loops():
        for curve in loop.curves:
            key = type(curve).__name__.replace("2D", "").lower()
            counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))


def _write_solid(face, depth: float, output: Path, tolerance: float, name: str) -> int:
    suffix = output.suffix.lower()

    if suffix in STEP_SUFFIXES:
        solid = extrude_face(face, depth, name=name)
        problems = solid.validate()
        if problems:
            print(f"warning: solid topology check reported {problems}", file=sys.stderr)
        write_step(solid, output, name=name)
        issues = validate_step(output.read_text())
        if issues:
            print(f"warning: emitted STEP failed validation: {issues}", file=sys.stderr)
            return 1
        cylinders = sum(1 for f in solid.faces() if type(f.surface).__name__ == "CylindricalSurface")
        print(
            f"wrote {output}  exact B-rep: {len(solid.faces())} faces "
            f"({cylinders} cylindrical), genus {solid.genus}, "
            f"volume {face.area() * depth:.6f} (closed form)"
        )
        return 0

    if suffix in MESH_SUFFIXES:
        mesh = tessellate_extrusion(face, depth, tolerance=tolerance)
        mesh.to_trimesh().export(str(output))
        exact_volume = face.area() * depth
        difference = mesh.volume() - exact_volume
        # The sign is informative: an outer boundary tessellates inside the
        # true curve (losing material) while a hole tessellates inside its
        # own curve (leaving extra material), so a part that is mostly
        # holes comes out heavier, not lighter.
        direction = "over" if difference > 0 else "under"
        print(f"wrote {output}  {mesh.certificate}")
        print(
            f"  exact volume {exact_volume:.6f}, mesh volume {mesh.volume():.6f} "
            f"({direction} by {abs(difference):.3e})"
        )
        return 0

    print(
        f"error: unsupported output '{suffix}'. Use .step for exact CAD, "
        f"or one of {', '.join(sorted(MESH_SUFFIXES))} for a mesh.",
        file=sys.stderr,
    )
    return 1


def cmd_extrude(args: argparse.Namespace) -> int:
    face = load_face(
        args.input,
        layer=args.layer,
        scale=args.scale,
        invert=args.invert,
        fit_tolerance=args.fit_tolerance,
    )
    output = Path(args.output)
    return _write_solid(face, args.depth, output, args.tolerance, args.name or output.stem)


def cmd_multiview(args: argparse.Namespace) -> int:
    top = ViewSpec(load_face(args.top, layer=args.top_layer), "top")
    front = ViewSpec(load_face(args.front, layer=args.front_layer), "front")

    result = reconstruct(top, front, tolerance=args.tolerance, name=args.name or "part")
    print(f"reconstruction: {result.certificate}")
    for note in result.certificate.notes:
        print(f"  note: {note}")
    print(f"  volume {result.volume():.6f}")

    output = Path(args.output)
    if result.solid is not None:
        suffix = output.suffix.lower()
        if suffix in STEP_SUFFIXES:
            write_step(result.solid, output, name=args.name or output.stem)
            issues = validate_step(output.read_text())
            if issues:
                print(f"warning: emitted STEP failed validation: {issues}", file=sys.stderr)
                return 1
            print(f"wrote {output}  exact B-rep ({len(result.solid.faces())} faces)")
            return 0
        face, z0, z1 = result.layers[0]
        return _write_solid(face, z1 - z0, output, args.tolerance, args.name or output.stem)

    # Stepped result: emit the layers as one mesh.
    if output.suffix.lower() in STEP_SUFFIXES:
        print(
            "error: this reconstruction is a stepped approximation, not an exact "
            "B-rep, so it cannot be written as STEP. Use a mesh format, or supply "
            "a rectangular front view to take the exact path.",
            file=sys.stderr,
        )
        return 1

    import trimesh

    meshes = []
    for face, z0, z1 in result.layers:
        mesh = tessellate_extrusion(face, z1 - z0, base_z=z0, tolerance=args.tolerance)
        meshes.append(mesh.to_trimesh())
    combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    combined.export(str(output))
    print(f"wrote {output}  {len(result.layers)} layers, {len(combined.faces)} triangles")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    faces, open_chains = load_faces(
        args.input,
        layer=args.layer,
        scale=args.scale,
        invert=args.invert,
        fit_tolerance=args.fit_tolerance,
    )
    print(f"{args.input}: {len(faces)} closed face(s)")
    for index, face in enumerate(sorted(faces, key=lambda f: -f.area())):
        minx, miny, maxx, maxy = face.bounds()
        exact = "closed-form" if face.is_area_exact() else "quadrature"
        print(
            f"  [{index}] area={face.area():.6f} ({exact})  holes={len(face.inners)}  "
            f"bounds=({minx:.3f}, {miny:.3f})-({maxx:.3f}, {maxy:.3f})"
        )
        print(f"        curves: {_curve_census(face)}")
        if args.verbose:
            for loop_index, loop in enumerate(face.loops()):
                kind = "outer" if loop_index == 0 else f"hole {loop_index}"
                print(f"        {kind}: " + ", ".join(_describe_curve(c) for c in loop.curves))
            certificate = face.outer.curves[0].deviation_certificate(args.tolerance)
            print(f"        at tol {args.tolerance:g}: {certificate}")

    if open_chains:
        print(f"  {len(open_chains)} chain(s) did not close (dimensions, leaders, or a gap)")
    if not faces:
        print("  no closed profile found -- check the layer, units, or threshold")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blueprint23d",
        description="Turn 2D blueprints into 3D parts, exactly where possible",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_input_args(p, prefix=""):
        p.add_argument(f"--{prefix}layer" if prefix else "--layer", default=None,
                       help="DXF layer to read (DXF input only)")
        if not prefix:
            p.add_argument("--scale", type=float, default=1.0,
                           help="Real units per pixel (raster input only)")
            p.add_argument("--invert", action="store_true",
                           help="Light lines on a dark background (raster input only)")
            p.add_argument("--fit-tolerance", type=float, default=0.6, dest="fit_tolerance",
                           help="Primitive-fitting tolerance in pixels (raster input only)")

    p_extrude = sub.add_parser("extrude", help="Extrude one profile into a solid")
    p_extrude.add_argument("--input", required=True, help="Blueprint (.dxf, .svg, or an image)")
    add_input_args(p_extrude)
    p_extrude.add_argument("--depth", type=float, required=True, help="Extrusion depth")
    p_extrude.add_argument("--output", required=True,
                           help="Output file: .step for exact CAD, .stl/.obj/.glb/.ply for a mesh")
    p_extrude.add_argument("--tolerance", type=float, default=1e-3,
                           help="Chordal tolerance for mesh output (default 1e-3)")
    p_extrude.add_argument("--name", default=None, help="Part name recorded in the file")
    p_extrude.set_defaults(func=cmd_extrude)

    p_multi = sub.add_parser("multiview", help="Reconstruct a solid from top and front views")
    p_multi.add_argument("--top", required=True, help="Top view (profile in XY)")
    p_multi.add_argument("--front", required=True, help="Front view (profile in XZ)")
    p_multi.add_argument("--top-layer", dest="top_layer", default=None, help="DXF layer for the top view")
    p_multi.add_argument("--front-layer", dest="front_layer", default=None, help="DXF layer for the front view")
    p_multi.add_argument("--output", required=True, help="Output file")
    p_multi.add_argument("--tolerance", type=float, default=1e-3,
                         help="Tolerance for stepping and tessellation (default 1e-3)")
    p_multi.add_argument("--name", default=None, help="Part name recorded in the file")
    p_multi.set_defaults(func=cmd_multiview)

    p_inspect = sub.add_parser("inspect", help="Report what was read, without building anything")
    p_inspect.add_argument("--input", required=True, help="Blueprint file")
    add_input_args(p_inspect)
    p_inspect.add_argument("--tolerance", type=float, default=1e-3, help="Tolerance to report bounds at")
    p_inspect.add_argument("--verbose", "-v", action="store_true", help="List every curve")
    p_inspect.set_defaults(func=cmd_inspect)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
