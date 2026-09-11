"""Reconstructing a solid from orthogonal views, exactly where possible.

Given a top view (a profile in XY) and a front view (a profile in XZ), the
part is the set of points whose shadow falls inside both:

.. math:: \\{(x,y,z) : (x,y) \\in A,\\; (x,z) \\in B\\}

The previous implementation meshed both views, extruded them, and handed
the result to a mesh CSG engine. That works, but it discards the exact
geometry first, so a bore that was a cylinder in both inputs comes out as
triangles -- and the answer carries no error statement at all.

This does it by sweeping instead, and the payoff is that one very common
case falls out **exactly**:

**Prismatic (exact).** If the front view is a plain rectangle -- which is
what a front view *is* for any part of constant thickness, i.e. most
plate, laser-cut and machined-from-flat parts -- then the intersection is
just the top profile restricted to the rectangle's x-range and extruded
through its z-range. Both operations are exact on curves, so the result is
a full B-rep with its cylinders intact, and exports to STEP as real
geometry. No tessellation is involved anywhere.

**Swept slabs (certified).** Otherwise the cross-section genuinely varies
with height, so the solid is cut into slabs at the critical heights where
the front view's structure changes, and each slab is clipped exactly at a
sample height. The result is a stepped approximation, and the step size is
driven down until the cross-section moves less than the tolerance across
each slab -- so the deviation is reported, not hidden.

The reconstruction's fundamental limit is unchanged and worth restating: a
solid is only recoverable from its silhouettes if it *is* the intersection
of them. A hole drilled at a compound angle is invisible to all three
orthogonal views and no amount of exactness recovers it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .booleans2d import X, Y, clip_face_to_slab, loop_spans_at
from .brep import Face2D, Loop2D, Solid, extrude_face
from .curves import DEFAULT_CHORD_TOLERANCE, Line2D

#: How closely a view must match a rectangle to take the exact path.
RECTANGLE_TOLERANCE = 1e-9


@dataclass
class ViewSpec:
    """One orthogonal view, as an exact face."""

    profile: Face2D
    name: str = "view"


@dataclass
class ReconstructionCertificate:
    """How the solid was obtained, and what that guarantees."""

    method: str
    exact: bool
    deviation: float
    slabs: int
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.exact:
            return f"{self.method}: exact (no tessellation), {self.slabs} slab(s)"
        return f"{self.method}: {self.slabs} slabs, cross-section deviation <= {self.deviation:.3e}"


@dataclass
class Reconstruction:
    """The recovered solid, plus how it was arrived at.

    ``solid`` is present only on the exact path, where the result is a real
    B-rep. On the swept path the geometry is the list of ``layers``, each an
    exact face over a height range; the stepping between them is the
    approximation.
    """

    certificate: ReconstructionCertificate
    solid: Optional[Solid] = None
    layers: List[Tuple[Face2D, float, float]] = field(default_factory=list)

    def volume(self) -> float:
        """Closed-form volume: exact on the exact path, quantified on the other."""
        return sum(face.area() * (z1 - z0) for face, z0, z1 in self.layers)


# --------------------------------------------------------------------------
# Rectangle detection
# --------------------------------------------------------------------------


def as_axis_aligned_rectangle(face: Face2D, tolerance: float = RECTANGLE_TOLERANCE):
    """Return ``(x0, x1, y0, y1)`` if the face is exactly an axis-aligned box.

    This is the test that unlocks the exact path, so it is deliberately
    strict: holes disqualify, curved edges disqualify, and every edge must
    be axis-parallel. A "nearly rectangular" front view is not a rectangle,
    and quietly treating it as one would silently change the part.
    """
    if face.inners:
        return None
    curves = face.outer.curves
    if not all(isinstance(c, Line2D) for c in curves):
        return None

    minx, miny, maxx, maxy = face.bounds()
    for curve in curves:
        dx = abs(curve.p1[0] - curve.p0[0])
        dy = abs(curve.p1[1] - curve.p0[1])
        if dx > tolerance and dy > tolerance:
            return None  # a diagonal edge

    # Area equal to the bounding box means it fills the box exactly.
    box_area = (maxx - minx) * (maxy - miny)
    if box_area <= 0.0:
        return None
    if abs(face.area() - box_area) > tolerance * max(1.0, box_area):
        return None
    return (minx, maxx, miny, maxy)


# --------------------------------------------------------------------------
# Sweep helpers
# --------------------------------------------------------------------------


def _critical_heights(face: Face2D) -> List[float]:
    """Heights where the front view's cross-section structure can change.

    Those are its vertices, and the extremes of any curved edge -- the
    points where spans appear, vanish, or reverse direction. Slab
    boundaries are placed there so that no slab straddles a topological
    change.
    """
    heights = set()
    for loop in face.loops():
        for curve in loop.curves:
            heights.add(curve.start[1])
            heights.add(curve.end[1])
            _, miny, _, maxy = curve.bounds()
            heights.add(miny)
            heights.add(maxy)
    return sorted(heights)


def _span_width_change(face: Face2D, z0: float, z1: float) -> float:
    """How far the front view's x-spans shift between two heights.

    This is the quantity the slab stepping must control: if the spans move
    by less than the tolerance across a slab, replacing the true ruled wall
    with a vertical one is within that tolerance.
    """
    # Sample strictly inside the slab. A step in the profile sits exactly
    # on a slab boundary by construction, and evaluating there picks up the
    # discontinuity itself -- reporting a huge "motion" for a slab whose
    # interior is perfectly prismatic, and subdividing it pointlessly.
    inset = 1e-6 * (z1 - z0)
    spans0 = loop_spans_at(face.outer, Y, z0 + inset)
    spans1 = loop_spans_at(face.outer, Y, z1 - inset)
    if not spans0 and not spans1:
        return 0.0
    if not spans0 or not spans1 or len(spans0) != len(spans1):
        # The number of spans changed, so a feature appeared, vanished or
        # merged inside the slab. That is a real change in cross-section
        # and must drive subdivision, so it is measured by how far the
        # overall extent moved rather than reported as unmeasurable --
        # returning infinity here previously made the caller accept the
        # slab with a recorded deviation of zero, which is how a dome came
        # out as a single block.
        def extent(spans):
            if not spans:
                return None
            return (min(a for a, _ in spans), max(b for _, b in spans))

        e0, e1 = extent(spans0), extent(spans1)
        if e0 is None or e1 is None:
            present = e0 or e1
            return abs(present[1] - present[0])
        return max(abs(e1[0] - e0[0]), abs(e1[1] - e0[1]))

    worst = 0.0
    for (a0, b0), (a1, b1) in zip(spans0, spans1):
        worst = max(worst, abs(a1 - a0), abs(b1 - b0))
    return worst


# --------------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------------


def reconstruct(
    top: ViewSpec,
    front: ViewSpec,
    tolerance: float = DEFAULT_CHORD_TOLERANCE,
    max_slabs: int = 512,
    name: str = "part",
) -> Reconstruction:
    """Recover the solid whose top and front silhouettes are the given faces.

    The top profile lives in XY; the front profile lives in XZ, with its
    own y-coordinate read as the part's z (height). Both are exact faces,
    and the x axis is shared -- which is exactly how the two views relate
    on a drawing sheet.
    """
    rectangle = as_axis_aligned_rectangle(front.profile)
    if rectangle is not None:
        x0, x1, z0, z1 = rectangle
        clipped = clip_face_to_slab(top.profile, X, x0, x1)
        if clipped is None:
            raise ValueError("the two views do not overlap: the reconstruction is empty")
        solid = extrude_face(clipped, z1 - z0, base_z=z0, name=name)
        return Reconstruction(
            certificate=ReconstructionCertificate(
                method="exact-prismatic",
                exact=True,
                deviation=0.0,
                slabs=1,
                notes=[
                    "front view is an axis-aligned rectangle, so the solid is the "
                    "top profile clipped to its x-range and extruded through its "
                    "z-range -- both exact operations on curves"
                ],
            ),
            solid=solid,
            layers=[(clipped, z0, z1)],
        )

    # General case: cut at the critical heights, then refine until the
    # cross-section barely moves across each slab.
    breaks = _critical_heights(front.profile)
    if len(breaks) < 2:
        raise ValueError("the front view has no height extent")

    slabs: List[Tuple[float, float]] = list(zip(breaks, breaks[1:]))
    slabs = [(a, b) for a, b in slabs if b - a > 1e-12]

    worst_deviation = 0.0
    refined: List[Tuple[float, float]] = []
    for z0, z1 in slabs:
        pieces = [(z0, z1)]
        # Bisect while the span motion across a piece exceeds tolerance.
        while pieces and len(refined) + len(pieces) < max_slabs:
            a, b = pieces.pop(0)
            motion = _span_width_change(front.profile, a, b)
            too_thin = (b - a) <= 1e-9
            if too_thin or (math.isfinite(motion) and motion <= tolerance):
                refined.append((a, b))
                worst_deviation = max(worst_deviation, motion if math.isfinite(motion) else 0.0)
                continue
            mid = 0.5 * (a + b)
            pieces.insert(0, (mid, b))
            pieces.insert(0, (a, mid))
        for a, b in pieces:
            # Whatever the budget cut short still counts toward the bound.
            motion = _span_width_change(front.profile, a, b)
            worst_deviation = max(worst_deviation, motion if math.isfinite(motion) else 0.0)
        refined.extend(pieces)

    layers: List[Tuple[Face2D, float, float]] = []
    notes: List[str] = []
    for z0, z1 in refined:
        mid = 0.5 * (z0 + z1)
        spans = loop_spans_at(front.profile.outer, Y, mid)
        if not spans:
            continue
        for span_low, span_high in spans:
            piece = clip_face_to_slab(top.profile, X, span_low, span_high)
            if piece is not None and piece.area() > 0.0:
                layers.append((piece, z0, z1))

    if not layers:
        raise ValueError("the two views do not overlap: the reconstruction is empty")

    if len(refined) >= max_slabs:
        notes.append(
            f"slab budget of {max_slabs} reached; the cross-section still moves "
            f"up to {worst_deviation:.3e} across the widest slab"
        )

    return Reconstruction(
        certificate=ReconstructionCertificate(
            method="swept-slabs",
            exact=False,
            deviation=worst_deviation,
            slabs=len(refined),
            notes=notes
            or [
                "front view is not a rectangle, so the cross-section varies with "
                "height and the solid is stepped; each layer's profile is exact"
            ],
        ),
        layers=layers,
    )


def rectangle_face(x0: float, x1: float, y0: float, y1: float) -> Face2D:
    """Convenience: the axis-aligned rectangular face used by a plain view."""
    return Face2D.create(
        Loop2D(
            [
                Line2D((x0, y0), (x1, y0)),
                Line2D((x1, y0), (x1, y1)),
                Line2D((x1, y1), (x0, y1)),
                Line2D((x0, y1), (x0, y0)),
            ]
        )
    )


def thickness_view(width_low: float, width_high: float, thickness: float) -> ViewSpec:
    """A plain front view: a part of constant thickness, seen edge on."""
    return ViewSpec(rectangle_face(width_low, width_high, 0.0, thickness), name="front")
