"""Meshing a B-rep solid with a certified deviation bound.

Tessellation is where exactness is finally, deliberately given up -- a
triangle mesh cannot represent a cylinder. The point of doing it last, and
doing it here, is that the loss is *quantified*: every vertex of the output
lies on the true surface, and every point of the true surface lies within a
stated distance of the mesh.

The bound is easy to state because of how the mesh is built:

* **Cap faces contribute zero error.** They are planar, and a planar region
  triangulated between points that lie exactly on its boundary curves is
  exact in the interior. All of the deviation is on the boundary.
* **Lateral faces contribute exactly the 2D chordal error.** A wall is the
  extrusion of a profile curve along Z, so the mesh and the true surface
  differ only in the XY cross-section, by precisely the chordal deviation
  of the tessellated boundary curve. Extrusion adds nothing.

So the Hausdorff distance between the output mesh and the true solid is
bounded by the largest per-curve chordal bound from :mod:`curves`, which is
itself a proven quantity. :attr:`MeshCertificate.deviation` reports it.

Watertightness is structural rather than repaired-after-the-fact: the caps
and the walls index into one shared vertex array built from a single
tessellation of each boundary curve, so coincident points are literally the
same vertex and no crack can open between a wall and a cap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .brep import Face2D, Loop2D
from .curves import DEFAULT_CHORD_TOLERANCE

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]
Triangle = Tuple[int, int, int]

try:  # pragma: no cover - depends on whether the extension was built
    from ._native import cdt as _native_cdt
except ImportError:  # pragma: no cover
    _native_cdt = None

CAP_BACKEND = "c" if _native_cdt is not None else "fallback"


@dataclass(frozen=True)
class MeshCertificate:
    """What the mesh guarantees about its relationship to the true solid."""

    deviation: float
    tolerance: float
    vertices: int
    triangles: int
    watertight: bool
    cap_backend: str

    @property
    def satisfied(self) -> bool:
        return self.deviation <= self.tolerance and self.watertight

    def __str__(self) -> str:
        status = "OK" if self.satisfied else "NOT MET"
        return (
            f"{self.triangles} triangles / {self.vertices} vertices, "
            f"deviation <= {self.deviation:.3e} (tol {self.tolerance:.3e}), "
            f"watertight={self.watertight} [{status}]"
        )


@dataclass
class TriangleMesh:
    """A triangle mesh plus the certificate describing its accuracy."""

    vertices: List[Point3]
    triangles: List[Triangle]
    certificate: MeshCertificate

    def volume(self) -> float:
        """Signed volume by the divergence theorem over the closed surface.

        Meaningful only because the mesh is watertight and consistently
        oriented; compare it against the exact ``area * depth`` to see how
        much volume the tessellation actually costs.
        """
        total = 0.0
        for i, j, k in self.triangles:
            ax, ay, az = self.vertices[i]
            bx, by, bz = self.vertices[j]
            cx, cy, cz = self.vertices[k]
            total += (
                ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
            )
        return total / 6.0

    def area(self) -> float:
        total = 0.0
        for i, j, k in self.triangles:
            ax, ay, az = self.vertices[i]
            bx, by, bz = self.vertices[j]
            cx, cy, cz = self.vertices[k]
            ux, uy, uz = bx - ax, by - ay, bz - az
            vx, vy, vz = cx - ax, cy - ay, cz - az
            total += 0.5 * math.hypot(math.hypot(uy * vz - uz * vy, uz * vx - ux * vz), ux * vy - uy * vx)
        return total

    def edge_manifold_report(self) -> Dict[str, int]:
        """Count how many times each undirected edge is used.

        A closed manifold surface uses every edge exactly twice, once in
        each direction. Anything else is a crack or a non-manifold join.
        """
        counts: Dict[Tuple[int, int], int] = {}
        directed: Dict[Tuple[int, int], int] = {}
        for i, j, k in self.triangles:
            for u, v in ((i, j), (j, k), (k, i)):
                key = (min(u, v), max(u, v))
                counts[key] = counts.get(key, 0) + 1
                directed[(u, v)] = directed.get((u, v), 0) + 1
        return {
            "edges": len(counts),
            "boundary_edges": sum(1 for c in counts.values() if c == 1),
            "nonmanifold_edges": sum(1 for c in counts.values() if c > 2),
            "inconsistent_edges": sum(1 for (u, v), c in directed.items() if directed.get((v, u), 0) != c),
        }

    def is_watertight(self) -> bool:
        report = self.edge_manifold_report()
        return report["boundary_edges"] == 0 and report["nonmanifold_edges"] == 0

    def to_trimesh(self):
        """Hand the mesh to trimesh for export, without letting it re-process.

        ``process=False`` matters: trimesh's default vertex merging would
        rebuild the very adjacency this module constructed exactly, and can
        weld vertices that are close but genuinely distinct.
        """
        import trimesh

        return trimesh.Trimesh(vertices=self.vertices, faces=self.triangles, process=False)


def _triangulate_cap(
    loops_points: Sequence[List[Point2]],
) -> Tuple[List[Point2], List[Triangle]]:
    """Triangulate a polygon with holes, returning shared points and triangles."""
    points: List[Point2] = []
    segments: List[Tuple[int, int]] = []
    for loop in loops_points:
        offset = len(points)
        count = len(loop)
        if count < 3:
            continue
        points.extend(loop)
        segments.extend((offset + i, offset + (i + 1) % count) for i in range(count))

    if not points:
        return [], []

    if _native_cdt is not None:
        return points, [tuple(t) for t in _native_cdt.triangulate(points, segments)]

    return points, _fallback_triangulate(loops_points, points)


def _fallback_triangulate(loops_points: Sequence[List[Point2]], points: List[Point2]) -> List[Triangle]:
    """Used only when the compiled CDT is unavailable.

    Delegates to shapely/trimesh, then maps the result back onto the shared
    vertex indices so the caller's watertightness argument still holds. The
    triangles are not guaranteed Delaunay, so mesh quality is worse; the
    geometry is still correct.
    """
    from shapely.geometry import Polygon
    from trimesh.creation import triangulate_polygon

    if not loops_points:
        return []
    polygon = Polygon(loops_points[0], [list(h) for h in loops_points[1:]])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    verts, faces = triangulate_polygon(polygon, engine="earcut")

    index_of = {}
    for i, p in enumerate(points):
        index_of.setdefault((round(p[0], 12), round(p[1], 12)), i)

    triangles: List[Triangle] = []
    remapped = []
    for vx, vy in verts:
        key = (round(float(vx), 12), round(float(vy), 12))
        if key not in index_of:
            index_of[key] = len(points)
            points.append((float(vx), float(vy)))
        remapped.append(index_of[key])
    for a, b, c in faces:
        triangles.append((remapped[a], remapped[b], remapped[c]))
    return triangles


def tessellate_extrusion(
    face: Face2D,
    depth: float,
    base_z: float = 0.0,
    tolerance: float = DEFAULT_CHORD_TOLERANCE,
) -> TriangleMesh:
    """Mesh the extrusion of ``face`` with deviation provably within ``tolerance``.

    Both caps and the walls are built over one tessellation of each
    boundary curve, so a wall vertex and the cap vertex above it are the
    same index -- the surface is closed by construction rather than by a
    post-hoc weld.
    """
    if depth <= 0.0:
        raise ValueError(f"extrusion depth must be positive, got {depth}")

    loops: Tuple[Loop2D, ...] = face.loops()
    loops_points = [loop.tessellate(tolerance) for loop in loops]
    if any(len(p) < 3 for p in loops_points):
        raise ValueError("a boundary loop tessellated to fewer than three points")

    cap_points, cap_triangles = _triangulate_cap(loops_points)
    if not cap_triangles:
        raise ValueError("cap triangulation produced no triangles")

    top_z = base_z + depth
    vertices: List[Point3] = [(x, y, base_z) for x, y in cap_points]
    top_offset = len(cap_points)
    vertices.extend((x, y, top_z) for x, y in cap_points)

    triangles: List[Triangle] = []

    # Bottom cap, wound backwards so its normal faces -Z.
    for a, b, c in cap_triangles:
        triangles.append((a, c, b))
    # Top cap keeps the 2D winding, so its normal faces +Z.
    for a, b, c in cap_triangles:
        triangles.append((a + top_offset, b + top_offset, c + top_offset))

    # Walls. Loop points were laid into cap_points in loop order, so the
    # indices of each loop are a contiguous run.
    start = 0
    for loop_points in loops_points:
        count = len(loop_points)
        for i in range(count):
            j = (i + 1) % count
            b0 = start + i
            b1 = start + j
            t0 = b0 + top_offset
            t1 = b1 + top_offset
            # Outward for a CCW outer loop; the reversed winding of a hole
            # loop flips these the same way, so hole walls face into the
            # void, which is outward from the material.
            triangles.append((b0, b1, t0))
            triangles.append((b1, t1, t0))
        start += count

    deviation = max(
        (curve.deviation_certificate(tolerance).bound for loop in loops for curve in loop.curves),
        default=0.0,
    )

    mesh = TriangleMesh(
        vertices=vertices,
        triangles=triangles,
        certificate=MeshCertificate(
            deviation=deviation,
            tolerance=tolerance,
            vertices=len(vertices),
            triangles=len(triangles),
            watertight=False,
            cap_backend=CAP_BACKEND,
        ),
    )
    mesh.certificate = MeshCertificate(
        deviation=deviation,
        tolerance=tolerance,
        vertices=len(vertices),
        triangles=len(triangles),
        watertight=mesh.is_watertight(),
        cap_backend=CAP_BACKEND,
    )
    return mesh
