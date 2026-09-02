# blueprint23d

Turn 2D blueprints (DXF, SVG, or scanned/photographed raster drawings) into
3D printable/machinable parts (STL/OBJ/GLB/PLY).

```
blueprint23d extrude --input plate.dxf --depth 5 --output plate.stl

blueprint23d multiview \
  --top top.dxf   --top-depth 8 \
  --front front.dxf --front-depth 20 \
  --output part.stl
```

## Design

### The problem, and the two strategies that actually work

"2D blueprint to 3D part" is under-determined in general: a single flat
drawing doesn't say how thick the part is, and even three orthogonal views
of a truly free-form object can't capture every concave feature. Rather
than attempt full general-purpose 3D reconstruction (a research problem),
this tool implements the two well-defined cases that cover the large
majority of real parts:

1. **Extrusion** (`extrude`) — one profile, pushed straight through a
   fixed depth. Exactly right for prismatic parts: laser/waterjet/CNC-router
   cut parts, gaskets, brackets, panels — anything with a constant
   cross-section.

2. **Multiview reconstruction** (`multiview`) — the classic engineering-
   drawing layout of top/front/side views, each showing the part's
   silhouette from one axis. Each silhouette is extruded through the
   *other* axes' extent, and the resulting solids are intersected with a
   boolean CSG operation. Two views are enough for most parts (front +
   top handles anything without side-specific detail); three refines it
   further.

   Worked example: a top view showing a 10×10 square (extruded 10 deep,
   along Z) intersected with a front view showing a 6×20 rectangle
   (extruded 6 deep, along Y) gives a 6×6×10 box — the footprint is
   clipped to the front view's width, and the height is clipped to the
   top view's extrusion depth. This is exactly how a CNC machinist reads
   a 3-view drawing: each view rules out material the others don't show.

   **Limitation, precisely stated:** this recovers any solid that is the
   intersection of its own three orthogonal projections — true for the
   overwhelming majority of machined/printed/cast parts. It *cannot*
   recover a feature invisible from all three axes, e.g. a hole drilled at
   a compound angle, or a pocket whose footprint is identical to the
   surrounding material in every silhouette. There is no way to detect
   this from the input alone; check the output mesh against the intended
   part.

### Pipeline

```
 blueprint file            profile                      solid
┌───────────────┐   parse  ┌──────────────┐  extrude/  ┌──────────────┐  export  ┌─────────┐
│ .dxf / .svg /  │ ───────▶│ MultiPolygon │  from_views│ trimesh.     │ ───────▶ │ .stl/   │
│ .png/.jpg/...  │         │ (shapely,    │ ──────────▶│ Trimesh      │          │ .obj/   │
└───────────────┘         │  holes nested)│            │ (watertight) │          │ .glb/.ply│
                            └──────────────┘            └──────────────┘          └─────────┘
```

* **Parse** (`parsers/`) — format-specific: DXF via `ezdxf` (LINE, ARC,
  CIRCLE, LWPOLYLINE/POLYLINE with bulge-encoded arcs, SPLINE, ELLIPSE),
  SVG via `svgelements` (any shape element, curves flattened, group
  transforms applied), raster via OpenCV (see below). Every parser reduces
  its input to the same intermediate form: a flat list of 2D point
  chains — independent LINE/ARC entities are not necessarily pre-joined
  into loops, so this is the lowest common denominator across formats.

* **Profile** (`geometry.py`) — shared cleanup, used by every parser:
  `weld_chains` stitches independent open chains back into closed loops by
  matching endpoints within a tolerance; `loops_to_polygons` turns the
  resulting loops into `shapely` polygons with holes correctly nested to
  arbitrary depth (a loop inside a solid is a hole, a loop inside a hole is
  a new solid island, and so on — the same even-odd rule used for font
  glyph rendering). This is also where near-duplicate points from
  floating-point arc sampling get snapped shut, since a ring that doesn't
  close exactly breaks watertight triangulation downstream.

* **Solid** (`reconstruct.py`) — `extrude` for the simple case;
  `from_views` for multiview reconstruction. Multiview alignment: since
  each view is normally authored as an independent file with its own
  origin, views are aligned on the axes they share by centering (default)
  or by their minimum corner (`--align min`) before extrusion — see the
  module docstring for the exact axis convention. Boolean intersection
  uses the `manifold3d` engine (pure-Python-installable, no external CAD
  binary required).

* **Export** (`export.py`) — mesh cleanup (dedupe/degenerate-face removal,
  hole filling, consistent outward normals) then `trimesh`'s exporters.

### Raster input is a best-effort fallback

DXF and SVG describe idealized zero-width curves, so their profiles are
exact modulo curve-flattening resolution. A raster image has no such
guarantee — a scanned or photographed line has real pixel width, uneven
contrast, and no vector precision. `parsers/raster_parser.py` handles this
by flood-filling from the image border through everything that isn't ink;
whatever the flood can't reach is the set of regions a closed ink loop
encircles, which sidesteps the natural failure mode of naively contouring
ink strokes (finding the stroke's *two* edges instead of the one curve a
vector parser would give). It's still an approximation: expect the
extracted profile's area to be within a few percent of the true part for a
clean scan, worse for a noisy photo. Prefer DXF/SVG whenever a vector
source exists at all. Use `scale_from_reference()` (or `--*-scale`) to
convert pixels to real units from one known dimension in the photo, and
`inspect` to sanity-check a parsed profile (area, hole count, bounds)
before committing to a full reconstruction.

## Installation

```
pip install -e .
```

## CLI

```
blueprint23d extrude --input FILE --depth D --output OUT.stl
    [--layer NAME]              # DXF layer to read
    [--scale S] [--invert]      # raster input only

blueprint23d multiview --output OUT.stl \
    --top FILE --top-depth D [--top-layer L] [--top-scale S] [--top-invert] \
    --front FILE --front-depth D [...] \
    --side FILE --side-depth D [...] \
    [--align center|min] [--engine manifold|blender]
    # at least two of --top/--front/--side are required

blueprint23d inspect --input FILE [--layer NAME] [--scale S] [--invert]
    # parse only; reports loop count, area, hole count, bounds -- use this
    # to debug a blueprint before extruding/reconstructing it
```

Supported input: `.dxf`, `.svg`, and raster images (`.png`, `.jpg`,
`.jpeg`, `.bmp`, `.tif`, `.tiff`). Supported output: `.stl`, `.obj`,
`.glb`, `.ply`.

## Library

```python
from blueprint23d import extrude, from_views, ViewSpec, export
from blueprint23d.loaders import load_profile

profile = load_profile("plate.dxf")
mesh = extrude(profile, depth=5.0)
export(mesh, "plate.stl")

mesh = from_views(
    top=ViewSpec(profile=load_profile("top.dxf"), depth=8),
    front=ViewSpec(profile=load_profile("front.dxf"), depth=20),
)
export(mesh, "part.stl")
```

## Development

```
pip install -e ".[dev]"
pytest
```

Tests build synthetic DXF/SVG/raster fixtures on the fly (via `ezdxf`
and OpenCV) and check reconstructed volumes against the analytic ground
truth, so they don't depend on any checked-in sample files.
