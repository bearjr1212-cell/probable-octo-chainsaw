"""Turn 2D profiles into 3D parts.

Two reconstruction strategies are provided:

* :func:`extrude` -- the simple, always-correct case: one profile, pushed
  straight through a fixed depth. Right for genuinely prismatic parts
  (gaskets, brackets, plates, laser/waterjet-cut parts).

* :func:`from_views` -- reconstructs a solid from two or three orthogonal
  silhouettes (top / front / side), the classic multiview engineering-
  drawing layout. Each silhouette is extruded through the *other* axis'
  extent and the results are intersected (CSG boolean intersection). This
  recovers any part whose 3D shape is fully determined by its orthogonal
  silhouettes -- true of the overwhelming majority of machined, printed,
  and cast parts -- but it *cannot* recover a feature that is invisible
  from all three axes (e.g. a hole drilled at a compound angle, or a
  pocket whose footprint exactly matches its surroundings in every
  silhouette). See the README for a worked example of the limitation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import trimesh
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon

Align = Literal["center", "min"]
Profile = Union[MultiPolygon, Polygon]


def _polygons(profile: Profile) -> List[Polygon]:
    if isinstance(profile, Polygon):
        return [] if profile.is_empty else [profile]
    return [p for p in profile.geoms if not p.is_empty]


def _mesh_from_polygon(
    polygon: Polygon,
    height: float,
    transform: Optional[np.ndarray] = None,
    mid_plane: bool = False,
) -> trimesh.Trimesh:
    mesh = trimesh.creation.extrude_polygon(polygon, height=height, mid_plane=mid_plane)
    if transform is not None:
        mesh.apply_transform(transform)
    mesh.fix_normals()
    return mesh


def _concatenate(meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)


def extrude(profile: Profile, depth: float) -> trimesh.Trimesh:
    """Extrude a flat profile straight along +Z by ``depth``.

    Every disjoint polygon in the profile becomes a separate solid in the
    output mesh (e.g. a drawing sheet with several parts nested on it).
    """
    polygons = _polygons(profile)
    if not polygons:
        raise ValueError("profile contains no closed loops to extrude")
    return _concatenate([_mesh_from_polygon(p, depth) for p in polygons])


@dataclass
class ViewSpec:
    """One orthogonal silhouette feeding :func:`from_views`.

    ``depth`` is the real-world extent, along that view's own extrusion
    axis, that the profile should be pushed through -- e.g. for the "top"
    view this is the part's overall height (Z), since the top view is
    extruded downward through it.
    """

    profile: Profile
    depth: float


# Maps a view name to (u_axis, v_axis, extrusion_axis): u/v are the
# profile's own in-plane (paper horizontal/vertical) axes, expressed as
# which world axis they represent.
_VIEW_AXES: Dict[str, Tuple[str, str, str]] = {
    "top": ("x", "y", "z"),  # looking down -Z: paper = world XY, extrude along Z
    "front": ("x", "z", "y"),  # looking along +Y: paper = world XZ, extrude along Y
    "side": ("y", "z", "x"),  # looking along -X: paper = world YZ, extrude along X
}

_AXIS_VECTOR = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}


def _view_transform(view: str) -> np.ndarray:
    """Rotation embedding a view's local (u, v, extrusion) frame into world axes."""
    u_axis, v_axis, w_axis = _VIEW_AXES[view]
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([_AXIS_VECTOR[u_axis], _AXIS_VECTOR[v_axis], _AXIS_VECTOR[w_axis]])
    return transform


def _alignment_offset(profile: Profile, align: Align) -> Tuple[float, float]:
    polygons = _polygons(profile)
    if not polygons:
        raise ValueError("profile contains no closed loops")
    combined = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
    minx, miny, maxx, maxy = combined.bounds
    if align == "center":
        return -(minx + maxx) / 2, -(miny + maxy) / 2
    return -minx, -miny


def from_views(
    top: Optional[ViewSpec] = None,
    front: Optional[ViewSpec] = None,
    side: Optional[ViewSpec] = None,
    align: Align = "center",
    engine: Optional[str] = None,
) -> trimesh.Trimesh:
    """Reconstruct a solid by intersecting extrusions of 2-3 orthogonal views.

    At least two of ``top``/``front``/``side`` must be given. Each profile
    is aligned on the in-plane axes it shares with the other views (see
    ``align``): ``"center"`` centers every view's bounding box -- and its
    extrusion -- on the origin, which is robust when the individual
    drawing files don't share a common origin; ``"min"`` instead aligns
    every view's lower-left corner (and its extrusion) at 0, matching
    drawings that were authored to a common datum. Either way, a view's
    ``depth`` must equal the real extent the *other* views expect along
    that axis, or the intersection will be clipped short.
    """
    views = {"top": top, "front": front, "side": side}
    given = {name: spec for name, spec in views.items() if spec is not None}
    if len(given) < 2:
        raise ValueError("from_views needs at least two of top/front/side")

    mid_plane = align == "center"
    solids = []
    for name, spec in given.items():
        dx, dy = _alignment_offset(spec.profile, align)
        polygons = [translate(p, xoff=dx, yoff=dy) for p in _polygons(spec.profile)]
        transform = _view_transform(name)
        meshes = [_mesh_from_polygon(p, spec.depth, transform=transform, mid_plane=mid_plane) for p in polygons]
        solids.append(_concatenate(meshes))

    result = solids[0]
    for other in solids[1:]:
        result = trimesh.boolean.intersection([result, other], engine=engine)

    if result.is_empty:
        raise ValueError(
            "the intersection of the given views is empty; the profiles are "
            "likely misaligned or a `depth` doesn't match another view's "
            "extent along that axis (see the `align` parameter)"
        )
    return result
