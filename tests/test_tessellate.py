"""Certified meshing: watertightness by construction, quantified convergence."""

import math

import pytest

from blueprint23d.brep import Face2D, Loop2D
from blueprint23d.curves import Arc2D, Line2D
from blueprint23d.tessellate import tessellate_extrusion


def rectangle(w=4.0, h=3.0):
    return Loop2D(
        [
            Line2D((0.0, 0.0), (w, 0.0)),
            Line2D((w, 0.0), (w, h)),
            Line2D((w, h), (0.0, h)),
            Line2D((0.0, h), (0.0, 0.0)),
        ]
    )


def plate_with_hole():
    return Face2D.create(rectangle(10.0, 6.0), [Loop2D([Arc2D.full_circle((5.0, 3.0), 1.5)])])


FACES = {
    "plate": lambda: Face2D.create(rectangle()),
    "disc": lambda: Face2D.create(Loop2D([Arc2D.full_circle((0.0, 0.0), 5.0)])),
    "plate_with_hole": plate_with_hole,
    "plate_with_two_holes": lambda: Face2D.create(
        rectangle(10.0, 6.0),
        [Loop2D([Arc2D.full_circle((3.0, 3.0), 1.0)]), Loop2D([Arc2D.full_circle((7.0, 3.0), 1.2)])],
    ),
}


@pytest.mark.parametrize("name", sorted(FACES))
def test_mesh_is_watertight_by_construction(name):
    """Caps and walls share vertices, so no crack can open between them."""
    mesh = tessellate_extrusion(FACES[name](), 3.0, tolerance=1e-3)
    report = mesh.edge_manifold_report()
    assert report["boundary_edges"] == 0, "open edge: the surface has a hole in it"
    assert report["nonmanifold_edges"] == 0
    assert report["inconsistent_edges"] == 0, "neighbouring faces disagree on orientation"
    assert mesh.certificate.watertight


@pytest.mark.parametrize("name", sorted(FACES))
def test_certificate_meets_requested_tolerance(name):
    mesh = tessellate_extrusion(FACES[name](), 3.0, tolerance=1e-3)
    assert mesh.certificate.deviation <= 1e-3
    assert mesh.certificate.satisfied


def test_polygonal_part_meshes_with_zero_volume_error():
    """With no curved edges there is nothing to approximate, so error is exact zero."""
    face = Face2D.create(rectangle(4.0, 3.0))
    mesh = tessellate_extrusion(face, 3.0, tolerance=1e-3)
    assert mesh.volume() == pytest.approx(36.0, abs=1e-12)


def test_volume_error_falls_linearly_with_tolerance():
    """Volume error should scale like the chordal deviation, i.e. ~10x per decade.

    This is the quantitative statement that the tolerance parameter means
    something: asking for ten times tighter really does deliver about ten
    times less volume error.
    """
    face = plate_with_hole()
    exact = face.area() * 4.0

    errors = {}
    for tolerance in (1e-2, 1e-3, 1e-4, 1e-5):
        mesh = tessellate_extrusion(face, 4.0, tolerance=tolerance)
        errors[tolerance] = abs(mesh.volume() - exact)

    for coarse, fine in ((1e-2, 1e-3), (1e-3, 1e-4), (1e-4, 1e-5)):
        ratio = errors[coarse] / errors[fine]
        assert 8.0 < ratio < 12.0, f"expected ~10x improvement, got {ratio:.2f}x"


def test_mesh_volume_is_inscribed_in_the_true_solid():
    """A chord lies inside its arc, so the mesh must under-report volume.

    Getting a mesh volume *above* the exact one would mean vertices were
    off the true surface -- a sign the tessellation had drifted rather than
    merely being coarse.
    """
    face = Face2D.create(Loop2D([Arc2D.full_circle((0.0, 0.0), 5.0)]))
    exact = face.area() * 2.0
    for tolerance in (1e-2, 1e-3, 1e-4):
        mesh = tessellate_extrusion(face, 2.0, tolerance=tolerance)
        assert mesh.volume() < exact
        assert mesh.volume() > exact * 0.99


def test_every_vertex_lies_exactly_on_the_true_surface():
    """Vertices are sampled from the analytic curve, not from a prior polyline."""
    radius = 5.0
    face = Face2D.create(Loop2D([Arc2D.full_circle((0.0, 0.0), radius)]))
    mesh = tessellate_extrusion(face, 2.0, tolerance=1e-3)
    for x, y, z in mesh.vertices:
        assert math.isclose(math.hypot(x, y), radius, rel_tol=1e-14)
        assert z in (0.0, 2.0)


def test_hole_walls_are_present_and_oriented():
    """A hole must be walled, not just cut out of the caps."""
    mesh = tessellate_extrusion(plate_with_hole(), 4.0, tolerance=1e-2)
    assert mesh.is_watertight()
    # Positive volume means outward normals overall; a hole wound the wrong
    # way would subtract rather than add and flip the sign.
    assert mesh.volume() > 0.0


def test_rejects_bad_depth():
    with pytest.raises(ValueError):
        tessellate_extrusion(Face2D.create(rectangle()), 0.0)


def test_mesh_exports_to_trimesh_without_reprocessing():
    pytest.importorskip("trimesh")
    mesh = tessellate_extrusion(plate_with_hole(), 4.0, tolerance=1e-3)
    tm = mesh.to_trimesh()
    assert tm.is_watertight
    assert tm.volume == pytest.approx(mesh.volume(), rel=1e-12)
    assert len(tm.faces) == mesh.certificate.triangles
