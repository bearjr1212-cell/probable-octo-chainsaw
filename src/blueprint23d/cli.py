"""Command-line interface for blueprint23d.

    blueprint23d extrude   --input part.dxf --depth 10 --output part.stl
    blueprint23d multiview --top top.dxf --top-depth 20 \\
                            --front front.dxf --front-depth 8 \\
                            --output part.stl
    blueprint23d inspect   --input part.dxf
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from . import reconstruct
from .export import export as export_mesh
from .loaders import load_profile


def _add_common_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Blueprint file (.dxf, .svg, or an image)")
    parser.add_argument("--layer", default=None, help="DXF layer to read (DXF input only)")
    parser.add_argument("--scale", type=float, default=1.0, help="Real units per pixel (raster input only)")
    parser.add_argument("--invert", action="store_true", help="Light lines on dark background (raster input only)")


def _load(args, prefix: str = "") -> object:
    key = prefix.replace("-", "_")
    path = getattr(args, f"{key}input" if prefix else "input")
    layer = getattr(args, f"{key}layer" if prefix else "layer", None)
    scale = getattr(args, f"{key}scale" if prefix else "scale", 1.0)
    invert = getattr(args, f"{key}invert" if prefix else "invert", False)
    return load_profile(path, layer=layer, scale=scale, invert=invert)


def cmd_extrude(args: argparse.Namespace) -> int:
    profile = _load(args)
    mesh = reconstruct.extrude(profile, depth=args.depth)
    out = export_mesh(mesh, args.output)
    print(f"wrote {out}  (volume={mesh.volume:.4f}, watertight={mesh.is_watertight})")
    return 0


def cmd_multiview(args: argparse.Namespace) -> int:
    specs = {}
    for name in ("top", "front", "side"):
        input_path = getattr(args, f"{name}_input")
        if input_path is None:
            continue
        profile = _load(args, prefix=f"{name}-")
        depth = getattr(args, f"{name}_depth")
        if depth is None:
            raise SystemExit(f"--{name}-depth is required when --{name} is given")
        specs[name] = reconstruct.ViewSpec(profile=profile, depth=depth)

    if len(specs) < 2:
        raise SystemExit("multiview needs at least two of --top/--front/--side")

    mesh = reconstruct.from_views(align=args.align, engine=args.engine, **specs)
    out = export_mesh(mesh, args.output)
    print(f"wrote {out}  (volume={mesh.volume:.4f}, watertight={mesh.is_watertight})")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    profile = _load(args)
    polygons = [] if profile.is_empty else list(profile.geoms)
    print(f"{args.input}: {len(polygons)} solid loop(s)")
    for i, poly in enumerate(polygons):
        minx, miny, maxx, maxy = poly.bounds
        print(
            f"  [{i}] area={poly.area:.4f}  holes={len(poly.interiors)}  "
            f"bounds=({minx:.3f}, {miny:.3f}) - ({maxx:.3f}, {maxy:.3f})"
        )
    if not polygons:
        print("  no closed loops were found -- check units/layer/threshold settings")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blueprint23d", description="Turn 2D blueprints into 3D parts")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extrude = sub.add_parser("extrude", help="Extrude a single flat profile into a solid")
    _add_common_input_args(p_extrude)
    p_extrude.add_argument("--depth", type=float, required=True, help="Extrusion depth, in the same units as the drawing")
    p_extrude.add_argument("--output", required=True, help="Output mesh file (.stl, .obj, .glb, .ply)")
    p_extrude.set_defaults(func=cmd_extrude)

    p_multi = sub.add_parser("multiview", help="Reconstruct a solid from 2-3 orthogonal views")
    for name in ("top", "front", "side"):
        p_multi.add_argument(f"--{name}", dest=f"{name}_input", default=None, help=f"{name.capitalize()} view blueprint file")
        p_multi.add_argument(f"--{name}-layer", dest=f"{name}_layer", default=None, help="DXF layer (DXF input only)")
        p_multi.add_argument(f"--{name}-scale", dest=f"{name}_scale", type=float, default=1.0, help="Real units per pixel (raster input only)")
        p_multi.add_argument(f"--{name}-invert", dest=f"{name}_invert", action="store_true", help="Light lines on dark background (raster input only)")
        p_multi.add_argument(f"--{name}-depth", dest=f"{name}_depth", type=float, default=None, help=f"Extent to extrude the {name} view through, along its own axis")
    p_multi.add_argument("--align", choices=["center", "min"], default="center", help="How to align views that share an axis (default: center)")
    p_multi.add_argument("--engine", choices=["manifold", "blender"], default=None, help="Boolean CSG backend (default: manifold)")
    p_multi.add_argument("--output", required=True, help="Output mesh file (.stl, .obj, .glb, .ply)")
    p_multi.set_defaults(func=cmd_multiview)

    p_inspect = sub.add_parser("inspect", help="Parse a blueprint and report the profile found, without building a solid")
    _add_common_input_args(p_inspect)
    p_inspect.set_defaults(func=cmd_inspect)

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
