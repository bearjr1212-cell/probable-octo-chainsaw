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
    blueprint23d check --input plate.dxf
    blueprint23d diff --before revB.dxf --after revC.dxf

``check`` exits ``0`` when the drawing passes, ``2`` when something
blocking was found, and ``1`` only on a real failure; ``diff`` exits ``0``
when the two revisions are the same part and ``2`` when they are not. Both
can be dropped straight into an intake script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .brep import extrude_face
from .curves import Arc2D, BezierCurve2D, EllipseArc2D, Line2D, NurbsCurve2D
from .diagnostics import Severity
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


#: Exit code for "we read it fine, and it is not fit to cut".
NOT_CUTTABLE = 2

#: Same code from ``diff``, meaning "the two revisions are not the same part".
DIFFERENCES_FOUND = 2

#: Formats that carry a units header and dimension entities.
ANNOTATED_SUFFIXES = {".dxf", ".dwg"}


def cmd_check(args: argparse.Namespace) -> int:
    """Everything an intake desk needs to know about one file."""
    from . import audit, diagnostics, units, validate

    path = Path(args.input)
    faces, report = validate.validate_drawing(path, layer=args.layer)

    inference = None
    findings: List = []
    if path.suffix.lower() in ANNOTATED_SUFFIXES and path.exists():
        try:
            inference = units.infer_from_dxf(path)
            report = diagnostics.merge(report, units.report_units(inference))
        except Exception as exc:  # a units guess is never worth failing over
            print(f"note: units could not be inferred ({exc})", file=sys.stderr)
        if not args.no_audit:
            findings, audit_report = audit.audit_drawing(path, layer=args.layer)
            report = diagnostics.merge(report, audit_report)

    if args.process and args.thickness is not None and faces:
        from . import processes

        part = max(faces, key=lambda f: f.area())
        report = diagnostics.merge(
            report,
            processes.check_manufacturability(
                part, processes.process(args.process), args.thickness
            ),
        )

    report.source = str(path)

    if args.json:
        print(report.to_json())
    else:
        if inference is not None:
            print(inference.describe())
        print(report.describe())
        if args.verbose and findings:
            print(audit.describe_findings(findings))

    return 0 if report.cuttable else NOT_CUTTABLE


def cmd_diff(args: argparse.Namespace) -> int:
    """What changed between two revisions of the same part."""
    from . import revisions

    diff = revisions.diff_drawings(
        args.before,
        args.after,
        layer=args.layer,
        revision_before=args.revision_before,
        revision_after=args.revision_after,
    )

    if args.json:
        payload = diff.to_dict()
        payload["report"] = diff.report.to_dict()
        import json

        print(json.dumps(payload, indent=2))
    else:
        print(diff.describe(include_unchanged=not args.changes_only))
        # The per-feature defects are the table above; anything else the
        # comparison turned up is said once, here.
        from .diagnostics import Code as _Code

        inline = {
            _Code.FEATURE_ADDED,
            _Code.FEATURE_REMOVED,
            _Code.FEATURE_CHANGED,
            _Code.REVISION_AMBIGUOUS,
        }
        for defect in diff.report.sorted_defects():
            if defect.code not in inline:
                print("  " + defect.describe())

    if diff.report.at_least(Severity.CRITICAL):
        return 1
    # Like ``diff`` itself: nothing to report is 0, differences are 2. A
    # revision that changed something is not a failure, but a script
    # waiting on "is this the part we already quoted?" needs to hear it.
    return 0 if diff.identical else DIFFERENCES_FOUND


def cmd_quote(args: argparse.Namespace) -> int:
    """Can this machine make the part, and what does the cut cost?"""
    from . import processes

    if args.compare:
        from .loaders import load_faces

        faces, _ = load_faces(args.input, layer=args.layer)
        if not faces:
            print("error: no closed profile to cut", file=sys.stderr)
            return 1
        face = max(faces, key=lambda f: f.area())
        results = processes.compare(face, args.thickness, args.processes or ())
        if args.json:
            import json

            print(
                json.dumps(
                    [
                        {"quote": q.to_dict(), "report": r.to_dict()}
                        for q, r in results
                    ],
                    indent=2,
                )
            )
        else:
            print(processes.describe_comparison(results, quantity=args.quantity))
        return 0 if results and results[0][1].cuttable else NOT_CUTTABLE

    estimate, report = processes.quote_drawing(
        args.input, args.process, args.thickness, layer=args.layer
    )
    if estimate is None:
        print(report.describe(), file=sys.stderr)
        return 1

    if args.json:
        import json

        print(json.dumps({"quote": estimate.to_dict(), "report": report.to_dict()}, indent=2))
    else:
        print(estimate.describe())
        if args.quantity > 1:
            print(
                f"  batch        {estimate.batch_seconds(args.quantity) / 60:10.1f} min "
                f"for {args.quantity}, {estimate.unit_cost(args.quantity):.2f} each"
            )
        print()
        print(report.describe())

    return 0 if report.cuttable else NOT_CUTTABLE


def cmd_certify(args: argparse.Namespace) -> int:
    """What every reported number is worth, and the fingerprint of the part."""
    from . import certificates

    certificate = certificates.certify_drawing(
        args.input,
        layer=args.layer,
        tolerance=args.tolerance,
        depth=args.depth,
        scale=args.scale,
        invert=args.invert,
        fit_tolerance=args.fit_tolerance,
    )

    if args.json:
        print(certificate.to_json())
    else:
        print(certificate.describe(limit=None if args.verbose else 12))

    if certificate.unresolvable:
        return NOT_CUTTABLE
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

    p_check = sub.add_parser(
        "check", help="Intake QA: units, topology, and whether the drawing agrees with itself"
    )
    p_check.add_argument("--input", required=True, help="Blueprint file")
    p_check.add_argument("--layer", default=None, help="DXF layer to read")
    p_check.add_argument("--json", action="store_true", help="Machine-readable report")
    p_check.add_argument("--no-audit", dest="no_audit", action="store_true",
                         help="Skip the dimension-versus-geometry cross-check")
    p_check.add_argument("--verbose", "-v", action="store_true",
                         help="List every dimension that was checked")
    p_check.add_argument("--process", default=None,
                         help="Also check manufacturability on this process")
    p_check.add_argument("--thickness", type=float, default=None,
                         help="Material thickness; required with --process")
    p_check.set_defaults(func=cmd_check)

    p_diff = sub.add_parser("diff", help="Compare two revisions feature by feature")
    p_diff.add_argument("--before", required=True, help="The earlier drawing")
    p_diff.add_argument("--after", required=True, help="The later drawing")
    p_diff.add_argument("--layer", default=None, help="DXF layer to read in both files")
    p_diff.add_argument("--revision-before", dest="revision_before", default=None,
                        help="Revision letter written on the earlier print")
    p_diff.add_argument("--revision-after", dest="revision_after", default=None,
                        help="Revision letter written on the later print")
    p_diff.add_argument("--changes-only", dest="changes_only", action="store_true",
                        help="Hide features that did not change")
    p_diff.add_argument("--json", action="store_true", help="Machine-readable report")
    p_diff.set_defaults(func=cmd_diff)

    p_certify = sub.add_parser(
        "certify", help="What every reported number is worth, and the part's fingerprint"
    )
    p_certify.add_argument("--input", required=True, help="Blueprint file")
    add_input_args(p_certify)
    p_certify.add_argument("--depth", type=float, default=None,
                           help="Extrusion depth, to certify the volume as well")
    p_certify.add_argument("--tolerance", type=float, default=1e-3,
                           help="Tolerance that confidence is measured against (default 1e-3)")
    p_certify.add_argument("--json", action="store_true", help="Machine-readable certificate")
    p_certify.add_argument("--verbose", "-v", action="store_true",
                           help="Certify every feature, not just the first twelve")
    p_certify.set_defaults(func=cmd_certify)

    from .processes import PROCESSES

    p_quote = sub.add_parser(
        "quote", help="Can this machine make the part, and what does the cut cost?"
    )
    p_quote.add_argument("--input", required=True, help="Blueprint file")
    p_quote.add_argument("--layer", default=None, help="DXF layer to read")
    p_quote.add_argument("--process", default="fiber_laser", choices=sorted(PROCESSES),
                         help="Cutting process (default fiber_laser)")
    p_quote.add_argument("--thickness", type=float, required=True,
                         help="Material thickness, in drawing units")
    p_quote.add_argument("--quantity", type=int, default=1,
                         help="Batch size; setup is charged once (default 1)")
    p_quote.add_argument("--compare", action="store_true",
                         help="Quote every process, producible ones first")
    p_quote.add_argument("--processes", nargs="*", default=None, metavar="NAME",
                         help="Limit --compare to these processes")
    p_quote.add_argument("--json", action="store_true", help="Machine-readable output")
    p_quote.set_defaults(func=cmd_quote)

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
