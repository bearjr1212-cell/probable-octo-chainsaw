# blueprint23d

Turn 2D blueprints (DXF, SVG, or scanned raster drawings) into 3D parts —
as **exact CAD geometry** where the mathematics permits, and as a mesh with
a **proven error bound** where it does not.

```console
$ blueprint23d extrude --input plate.dxf --depth 8 --output plate.step
wrote plate.step  exact B-rep: 8 faces (2 cylindrical), genus 2, volume 45973.170084 (closed form)

$ blueprint23d extrude --input plate.dxf --depth 8 --output plate.stl --tolerance 1e-4
wrote plate.stl  4500 triangles / 2248 vertices, deviation <= 9.992e-05 (tol 1.000e-04), watertight=True [OK]
  exact volume 45973.170084, mesh volume 45973.212609 (over by 4.253e-02)
```

The first command produces a file in which a 6.35 mm bore is literally

```step
#96 = CYLINDRICAL_SURFACE('',#95,6.35);
```

A CAM system reads that as a hole with a diameter. A mesh of the same part
is a few thousand triangles that nobody can turn back into 6.35.

---

## The central idea

Precision in a CAD pipeline is decided at the **front door**. If a DXF
`CIRCLE` becomes a 64-gon the moment it is read, then:

- the hole is permanently undersized by the polygon's inscribed error,
- its wall is permanently faceted, so it can never be a cylinder,
- and the nominal diameter — the number the drawing existed to communicate —
  is gone and unrecoverable.

So nothing is flattened on input. A circle is carried as a circle
(centre, radius, angles) through parsing, assembly, clipping, extrusion,
and export. Tessellation happens **once, last, and only if a mesh was
asked for** — at which point the deviation is a stated, proven quantity.

```
                 exact all the way across
  ┌────────┐   ┌─────────┐   ┌────────┐   ┌────────┐   ┌──────────┐
  │  DXF   │──▶│ Curve2D │──▶│ Face2D │──▶│ Solid  │──▶│  STEP    │
  │  SVG   │   │ exact   │   │ loops  │   │ B-rep  │   │ AP214    │
  │ raster │   │ arcs,   │   │ nested │   │ +cylin-│   └──────────┘
  └────────┘   │ NURBS   │   │ holes  │   │ ders   │        │
               └─────────┘   └────────┘   └────────┘   ┌──────────┐
                                               └──────▶│ mesh +   │
                                            tessellate │ error    │
                                              (once)   │ bound    │
                                                       └──────────┘
```

---

## The mathematics

### 1. Exact predicates — because a wrong *sign* is not a small error

Every topological decision (Delaunay flips, loop nesting, curve
classification) is a sign question. Floating point answers them wrongly
whenever the true value is smaller than the accumulated rounding error —
exactly the near-degenerate cases real drawings are full of.

Two independent implementations, required to agree:

- **C**: Shewchuk adaptive-precision expansions — error-free
  transformations (`two_sum`, `two_product`) that capture each operation's
  rounding residual exactly, run as a four-stage cascade that stops at the
  first stage whose error bound is provably below the result's magnitude.
- **Python**: a floating-point filter with exact rational fallback
  (`fractions.Fraction`), used as the test oracle.

> **Verified.** Over 20,000 adversarial near-collinear inputs where naive
> floating point returns the wrong sign in **1,172 cases**, both
> implementations were wrong **0 times**.

The in-circle determinant is evaluated *fully exactly in C* rather than
deferred, because tessellating an arc produces **exactly cocircular**
points by construction — on this workload the degenerate branch is the
common case, not the rare one. Measured **40× faster** than the Python path
on both degenerate predicates.

### 2. Tessellation bounds that are theorems

| Curve | Bound | Source |
|---|---|---|
| Line | exactly 0 | nothing to approximate |
| **Circular arc** | <code>r(1 − cos(Δ/2))</code> | the sagitta identity, **exact** |
| Ellipse | <code>h²‖C″‖/8</code> | linear-interpolation error, ‖C″‖ from the semi-axes |
| Bézier / NURBS | <code>maxᵢ dist(Pᵢ, chord)</code> | convex-hull property |

The arc bound is *inverted* to choose the step: Δ_max = 2·arccos(1 − ε/r),
so the segment count is optimal — no sample wasted, none missing.

The Bézier bound is rigorous for rational curves too: the curve lies in the
convex hull of its control points, and distance-to-a-segment is convex, so
its maximum over the hull is attained at a control point.

> **Verified.** Measured deviation matches the arc bound to 4 decimal
> places (ratio 1.0000 — the bound is *attained*, not merely respected).
> Segment counts follow the predicted `tol^(−1/2)` law to within 0.03%
> (ratios 3.141 → 3.162 vs √10 = 3.1623).

### 3. Closed-form areas via Green's theorem

Area is ∮(x dy − y dx)/2, which has an elementary antiderivative for lines,
circular arcs, and elliptical arcs. For an arc, the integrand collapses to
`cₓ r cos θ + c_y r sin θ + r²`.

> **Verified.** A disc reports area **exactly** `math.pi * r * r` — 0 ULP
> error — where an inscribed n-gon is short by O(1/n²). An ellipse reports
> exactly `π a b`.

A real bug this caught: over a *closed* turn the centre-dependent terms
cancel analytically, but evaluating them numerically folds in
`sin(2π) = −2.4e−16` scaled by the centre offset — so a disc's area
depended on where on the sheet it was drawn. Now short-circuited.

### 4. Topology validated by generalised Euler–Poincaré

```
V − E + 2F − L − 2S + 2G = 0
```

The familiar `V − E + F = 2` does **not** apply to these parts: a plate
with a through-hole has annular cap faces and genus 1. The general form —
with loops `L`, shells `S`, genus `G` — is the one that actually tests
them.

> **Verified.** Zero defect across plates, discs, fillets, and multi-hole
> parts, with every edge shared by exactly two faces.

### 5. Constrained Delaunay triangulation (C)

Delaunay rather than ear clipping because it **maximises the minimum
angle**, keeping slivers out — slivers have numerically meaningless normals
and get rejected by slicers and FEA meshers downstream.

Native because the inner loop is one exact predicate per decision, and a
few thousand boundary points generate millions of them.

Two bugs found by *conservation testing* rather than inspection:

1. The two triangles either side of an edge traverse it in **opposite**
   directions, so the neighbour's adjacency lookups were swapped — a square
   meshed to one triangle.
2. Flooding inward from outside and stopping at constraints leaves holes
   **solid**, since nothing outside can reach an enclosed void. Replaced
   with **crossing parity**: depth 1 is material, 2 is a hole, 3 an island
   inside it — arbitrarily deep nesting in one pass.

> **Performance.** Replacing two linear scans with a hash set and a
> vertex-fan walk took 3,000 points from **2366 ms → 49 ms (48×)**; 12,000
> points now triangulate in 0.6 s.

### 6. Meshing with a certified bound

Cap faces are planar and contribute **zero** error. Walls are extrusions of
the profile curves, so they deviate by exactly the 2D chordal error —
extrusion adds nothing. The Hausdorff distance from the true solid is
therefore bounded by the largest per-curve bound.

Watertightness is **structural**, not repaired afterwards: caps and walls
index one shared vertex array, so a wall vertex and the cap vertex above it
are literally the same index.

> **Verified.** Volume error falls **~10× per decade** of tolerance
> (9.47×, 9.63×, 9.84×, 9.95× → 10). A purely polygonal part meshes with
> **exactly zero** volume error. Every vertex lies on the true surface to
> 1e−14.

### 7. Raster → real CAD geometry

Three stages, each beating the pixel grid for a different reason:

1. **Sub-pixel localisation.** An edge is where the gradient peaks, almost
   always *between* pixels. A parabola through the gradient magnitude at
   ±1 px along the gradient normal gives the vertex:
   `δ = (g₋ − g₊) / 2(g₋ − 2g₀ + g₊)`.
2. **Segmentation** into line and arc spans.
3. **Fitting** — where the precision comes from. Per-point error is
   independent, so a fit over N points averages it down like **1/√N**.

The naive choice is wrong at each step:

- **Ordinary** least squares isn't rotation invariant and fails outright on
  a vertical edge. **Total** least squares minimises perpendicular distance.
- The obvious algebraic circle fit is badly biased on short arcs — exactly
  the fillet case. **Taubin** normalises by the gradient of the algebraic
  form; **Gauss-Newton** then refines against the true orthogonal residual.
- A noisy *straight* edge fits a circle of radius 3000 within any
  tolerance. Curvature must be **resolvable**, not merely fittable: an arc
  is rejected when its **sagitta** falls below the noise floor.

> **Verified.** Radius error follows the 1/√N law to within a factor of
> three over 40 trials. A rendered circle recovers as **one arc** to better
> than half a pixel; a rounded rectangle recovers **exactly 4 fillets and
> 4 edges** with under 1% area error; a plain rectangle yields **no arcs at
> all**.

### 8. Exact clipping, and multiview reconstruction

Clipping happens on curves, not tessellations: an arc clipped by a line is
still an arc with the same centre and radius. Intersections are closed form
for lines, arcs and ellipses; free-form curves bracket a sign change and
bisect to full precision.

Reconstruction from a top view (XY) and front view (XZ) has two paths:

- **Exact (prismatic).** If the front view is a plain rectangle — which is
  what a front view *is* for any part of constant thickness — the solid is
  the top profile clipped to its x-range and extruded through its z-range.
  Both exact. **Result is a real B-rep with its cylinders intact.**
- **Certified (swept slabs).** Otherwise the cross-section genuinely varies
  with height, so the solid is cut at the profile's critical heights and
  refined until the cross-section moves less than the tolerance across each
  slab.

> **Verified.** Clipped areas match closed-form circular segments to 1e−12.
> A shouldered part reconstructs in **2 slabs with zero deviation** and
> exact volume. A dome converges from 0.11% to 0.0002% volume error with
> the reported bound always honouring the request.

---

## Where the guarantees stop

Stated plainly, because a precision tool that overclaims is worse than one
that doesn't try:

- **Silhouette reconstruction is fundamentally limited.** A solid is
  recoverable from orthogonal views only if it *is* the intersection of
  them. A hole drilled at a compound angle is invisible to all three views
  and no amount of exactness recovers it.
- **Raster accuracy depends on resolvable curvature.** A short,
  small-radius fillet under heavy noise is genuinely unidentifiable. Every
  fit reports its RMS residual and sagitta so you can reject it; the tool
  will not hand you a confident number it cannot justify.
- **Raster contours carry a systematic bias** of roughly a third of a pixel
  from thresholded extraction, on top of the random error the fitting
  averages down.
- **A stepped reconstruction is not exact CAD** and the tool refuses to
  write one as STEP rather than passing it off.
- **No CAD kernel was available to open the STEP output here.** Validity is
  established structurally — every reference resolves, no duplicate ids,
  required AP214 scaffolding present, shell references only faces, every
  face has a surface and a bound — and the validator is itself tested
  against deliberately corrupted files. That is not the same as a
  round-trip through SolidWorks.
- **Bézier/NURBS areas** fall back to Gauss-Legendre quadrature;
  `is_area_exact()` tells you which path was taken.

---

## Reading a drawing you did not draw

Everything above assumes the file is correct. Files that arrive from
customers are frequently not, and the failures are quiet ones — a contour
that misses closing by 0.13 mm, two entities lying on top of each other, a
header that says inches over geometry drawn in millimetres. Nothing looks
wrong, every step of the job is right, and the part comes out wrong.

So the package reports rather than raises. Every check returns a
[`Report`](src/blueprint23d/diagnostics.py) of `Defect`s carrying a stable
code, a location in drawing coordinates *and* the source entity handle, the
actual numbers, and a remedy written for whoever has to fix it.

**Units** (`units.py`) are inferred from four independent signals — the
`$INSUNITS` header, the ratio between what dimensions say and what they
span, whether the overall size is plausible, and whether coordinates land
on round metric or round imperial steps — and reported with a confidence
that accounts for how much evidence there actually was. One weak signal
agreeing with itself is 100% of the evidence and nothing like certainty, so
the confidence is `agreement × saturation`, and a lone plausibility guess
comes out under 60% and asks a human.

The valuable case is contradiction: geometry spanning 101.6 against a
dimension reading 4.00 is a ratio of 25.4, which is not noise. It is a
drawing carrying two unit systems, and it is how a part gets cut 25× wrong.

**Topology** (`validate.py`) finds open contours (with the gap measured and
located), self-intersections (at the crossing point, decided with the exact
`orient2d` predicate, so a tessellated circle's cocircular samples are not
a false positive), and duplicate or partially overlapping edges — collinear
overlap is tested exactly, and the shared length is reported exactly,
including for concentric arcs.

**The drawing against itself** (`audit.py`) is the check nothing else does.
A dimension makes a claim; the geometry makes another; inside a live CAD
model they cannot disagree because one is derived from the other, but an
exported file has lost that link. Definition points are snapped onto real
curves and re-measured there. A hole labelled `Ø12.70` whose arc measures
`Ø12.60` is a quarter-inch drill that will not fit.

Radial dimensions are matched to arcs **by position, never by radius** —
matching on radius would assume the answer and could never detect a
disagreement. And a dimension that no longer touches the part is not
quietly trusted: the detachment distance is measured and reported, because
a dimension floating 3 mm off the edge is one the geometry was edited out
from under.

**Revisions** (`revisions.py`) answers what the revision note refuses to.
Features are paired by position, again never by size, and a pairing is only
made when it is obvious: two features match when they are each other's
nearest candidate and closer than the features themselves are spaced. A
hole that moved further than its neighbours are apart is genuinely
indistinguishable from one deleted and another added, so it is reported
that way rather than guessed at.

Because geometry here is held exactly, *unchanged means unchanged* — a
bit-identical description, canonicalised over where the loop starts, which
way it runs, and what angle a full circle begins at. Nothing is "within an
epsilon somebody picked". Which is why a circle re-exported as a 64-segment
polyline is not reported as unchanged: it is reported as a hole that became
a cutout, one edge that became sixty-four, and a fraction of a square
millimetre smaller.

```console
$ blueprint23d diff --before revB.dxf --after revC.dxf
revB → revC
  hole (25, 30)                     Ø6.35 → Ø6.5                      changed
  slot (80, 15)                     new                               added
  outer profile (50, 30)                                              unchanged
  hole (60, 45)                                                       unchanged
```

**DWG** input (`parsers/dwg.py`) goes through LibreDWG's `dwg2dxf` or the
ODA File Converter, whichever is on `PATH`. The converter is *invoked*, not
linked — the GPL boundary is a process boundary, and the module docstring
says why. Layer names do not survive LibreDWG conversion, so a layer filter
silently matching nothing is reported as `CONVERSION_METADATA_LOST` rather
than left to be discovered downstream.

---

## Install

```console
pip install -e .
```

The C extensions are **optional**: without a compiler the pure-Python
implementations take over, more slowly but with identical results
(`tests/test_exact.py` pins that equivalence). The build pins
`-ffp-contract=off` and `-fexcess-precision=standard`, because Shewchuk's
error-free transformations break if the compiler contracts a multiply-add
into an FMA — an FMA computes the product to infinite precision before
adding, destroying the rounding residual the algorithm exists to capture.

## CLI

```console
blueprint23d extrude   --input FILE --depth D --output OUT[.step|.stl|.obj|.glb|.ply]
                       [--layer NAME] [--tolerance T] [--name N]
                       [--scale S] [--invert] [--fit-tolerance F]   # raster only

blueprint23d multiview --top FILE --front FILE --output OUT [--tolerance T]
                       [--top-layer L] [--front-layer L]

blueprint23d inspect   --input FILE [-v] [--layer NAME] [--tolerance T]

blueprint23d check     --input FILE [--layer NAME] [--json] [--no-audit] [-v]

blueprint23d diff      --before FILE --after FILE [--layer NAME] [--json]
                       [--revision-before R] [--revision-after R] [--changes-only]
```

`inspect` reports curve types and radii, whether each area came from the
closed-form integral or quadrature, and which chains failed to close —
use it to debug a drawing before building anything from it.

`check` runs the whole intake pass: units, topology, and the drawing
against its own dimensions. `diff` compares two revisions feature by
feature. Both exit `0` on a clean result, `2` on "we read it fine and it is
not what you want" (not cuttable; the revisions differ), and `1` only when
something genuinely failed — so they drop straight into an intake script:

```console
$ blueprint23d check --input customer.dxf
units: mm (confidence 74%)
  header: mm (weight 3.00) — $INSUNITS=4
  size: mm (weight 1.20) — 100 units is 100.0 mm in mm, a plausible part size
customer.dxf: NOT CUTTABLE AS SUPPLIED
  1 error
  [ERROR] DIMENSION_MISMATCH: drawing calls this diameter 12.7 but the geometry
          measures 12.6 (off by -0.1)  at (27.298, 32.298) layer 0 entity #8C
```

`--json` on either gives the same content as a machine-readable report.

## Library

```python
from blueprint23d import load_face, extrude_face, write_step, tessellate_extrusion

face = load_face("plate.dxf", layer="OUTLINE")   # exact curves, holes nested
print(face.area(), face.is_area_exact())          # closed-form where possible

solid = extrude_face(face, depth=8.0)             # arcs become true cylinders
assert solid.validate() == []                     # Euler-Poincare check
write_step(solid, "plate.step")                   # real CAD

mesh = tessellate_extrusion(face, 8.0, tolerance=1e-4)
print(mesh.certificate)   # deviation bound, triangle count, watertightness
```

Multiview:

```python
from blueprint23d import load_face, reconstruct, ViewSpec

result = reconstruct(ViewSpec(load_face("top.dxf")), ViewSpec(load_face("front.dxf")))
print(result.certificate)          # which path, and what it guarantees
if result.solid:                   # present only on the exact path
    write_step(result.solid, "part.step")
```

## Module map

| Module | Role |
|---|---|
| `exact.py` + `_native/predicates.c` | filtered-exact orient2d / incircle, interval arithmetic |
| `curves.py` | exact Line/Arc/Ellipse/Bézier/NURBS with deviation certificates |
| `brep.py` | loops, faces, solids; closed-form areas; Euler–Poincaré |
| `assembly.py` | welding loose curves into loops, nesting by parity |
| `booleans2d.py` | exact clipping against axis-aligned lines |
| `multiview.py` | orthogonal-view reconstruction, exact and certified paths |
| `tessellate.py` + `_native/cdt.c` | constrained Delaunay, certified meshing |
| `fitting.py` + `_native/fitkernels.c` | sub-pixel edges, TLS lines, Taubin+GN circles |
| `step_writer.py` | STEP AP214 output and structural validation |
| `parsers/dxf_exact.py`, `parsers/svg_exact.py` | readers that preserve geometry |
| `parsers/dwg.py` | DWG via LibreDWG / ODA, invoked as a subprocess |
| `diagnostics.py` | stable defect codes, locations, measurements, remedies |
| `units.py` | unit inference from four signals, with honest confidence |
| `validate.py` | open contours, self-intersections, duplicate and overlapping edges |
| `annotations.py` | dimension entities and their overridden text |
| `audit.py` | the drawing checked against its own geometry |
| `revisions.py` | feature-by-feature comparison of two revisions |

## Development

```console
pip install -e ".[dev]"
pytest
```

307 tests. They are written to check *exactness and conservation* rather
than appearance: predicate signs against rational ground truth, measured
deviation against proven bounds, triangulated area against analytic area,
recovered radii against rendered shapes, and convergence *rates* against
their theoretical exponents.

The intake-QA tests are written the other way round — each one builds a
drawing that lies in a specific way a real one does (a contour 0.13 mm
short of closing, text typed over a dimension, a part narrowed with the
dimension left behind, geometry in millimetres dimensioned in inches) and
checks that the defect is named, located and measured, not merely noticed.
