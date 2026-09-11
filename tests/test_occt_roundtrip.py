"""The STEP output, opened by a real CAD kernel.

Everything else in this suite checks the writer against itself: that every
reference resolves, that the AP214 scaffolding is present, that the entity
graph is well formed. That establishes the file is *structurally* sound and
says nothing at all about whether a kernel will agree with it, which is the
only thing a customer cares about.

So these tests hand the emitted files to OpenCASCADE — the kernel behind
FreeCAD, KiCad's 3D viewer, and a great deal of commercial CAD — and ask it
three questions that the writer cannot answer about itself:

* does it parse, and does ``BRepCheck_Analyzer`` call the solid valid;
* does the volume the kernel integrates over the imported B-rep match the
  volume computed in closed form from the 2D profile, and
* does a hole come back as a genuine ``GeomAbs_Cylinder`` carrying its
  nominal radius, rather than as a surface that merely looks round.

That last one is the whole thesis of the package. A Ø12.70 bore has to
arrive at the other end as a cylinder of radius exactly 6.35, because a
CAM system reading 6.3499987 selects a different tool.

OpenCASCADE is an optional dependency (``pip install -e ".[occt]"``); the
module skips without it rather than failing, so the suite still runs on a
machine that has no kernel.
"""

import math

import pytest

pytest.importorskip("OCP", reason="OpenCASCADE (cadquery-ocp) not installed")

from OCP.BRep import BRep_Tool  # noqa: E402
from OCP.BRepCheck import BRepCheck_Analyzer  # noqa: E402
from OCP.BRepGProp import BRepGProp  # noqa: E402
from OCP.GeomAbs import GeomAbs_SurfaceType  # noqa: E402
from OCP.GeomAdaptor import GeomAdaptor_Surface  # noqa: E402
from OCP.GProp import GProp_GProps  # noqa: E402
from OCP.IFSelect import IFSelect_ReturnStatus  # noqa: E402
from OCP.STEPControl import STEPControl_Reader  # noqa: E402
from OCP.TopAbs import TopAbs_ShapeEnum  # noqa: E402
from OCP.TopExp import TopExp_Explorer  # noqa: E402
from OCP.TopoDS import TopoDS  # noqa: E402

from blueprint23d.brep import Face2D, Loop2D, extrude_face  # noqa: E402
from blueprint23d.curves import (  # noqa: E402
    Arc2D,
    BezierCurve2D,
    EllipseArc2D,
    Line2D,
)
from blueprint23d.step_writer import write_step  # noqa: E402


# --------------------------------------------------------------------------
# Talking to the kernel
# --------------------------------------------------------------------------


def read_step(path):
    """Import a STEP file and return the shape, failing loudly if it will not."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    assert status == IFSelect_ReturnStatus.IFSelect_RetDone, f"OCCT refused to read {path}"
    assert reader.TransferRoots() > 0, "OCCT read the file but transferred no roots"
    shape = reader.OneShape()
    assert not shape.IsNull()
    return shape


def volume_of(shape) -> float:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def surface_types(shape):
    """Every face's surface type, as the kernel classifies it."""
    types = []
    explorer = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        adaptor = GeomAdaptor_Surface(BRep_Tool.Surface_s(face))
        types.append(adaptor.GetType())
        explorer.Next()
    return types


def cylinder_radii(shape):
    """The radius of every face the kernel calls a cylinder."""
    radii = []
    explorer = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        adaptor = GeomAdaptor_Surface(BRep_Tool.Surface_s(face))
        if adaptor.GetType() == GeomAbs_SurfaceType.GeomAbs_Cylinder:
            radii.append(adaptor.Cylinder().Radius())
        explorer.Next()
    return radii


def count(shape, kind) -> int:
    total = 0
    explorer = TopExp_Explorer(shape, kind)
    while explorer.More():
        total += 1
        explorer.Next()
    return total


def roundtrip(tmp_path, face, depth, name):
    """Write a solid to STEP, read it back with OCCT, return (shape, path)."""
    solid = extrude_face(face, depth, name=name)
    assert solid.validate() == [], "our own topology check failed before OCCT saw it"
    path = tmp_path / f"{name}.step"
    write_step(solid, path, name=name)
    return read_step(path), path


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def rectangle(x0, y0, x1, y1):
    return Loop2D(
        [
            Line2D((x0, y0), (x1, y0)),
            Line2D((x1, y0), (x1, y1)),
            Line2D((x1, y1), (x0, y1)),
            Line2D((x0, y1), (x0, y0)),
        ]
    )


def circle(centre, radius):
    return Loop2D([Arc2D.full_circle(centre, radius)])


# --------------------------------------------------------------------------
# It parses, and the kernel calls it valid
# --------------------------------------------------------------------------


def test_a_plain_block_imports_and_checks_out(tmp_path):
    face = Face2D.create(rectangle(0.0, 0.0, 100.0, 60.0))
    shape, _ = roundtrip(tmp_path, face, 8.0, "block")

    assert BRepCheck_Analyzer(shape).IsValid()
    assert count(shape, TopAbs_ShapeEnum.TopAbs_SOLID) == 1
    assert count(shape, TopAbs_ShapeEnum.TopAbs_FACE) == 6
    assert volume_of(shape) == pytest.approx(100.0 * 60.0 * 8.0, rel=1e-12)


def test_a_bored_plate_imports_and_checks_out(tmp_path):
    face = Face2D.create(rectangle(0.0, 0.0, 100.0, 60.0), [circle((50.0, 30.0), 6.35)])
    shape, _ = roundtrip(tmp_path, face, 8.0, "bored")
    assert BRepCheck_Analyzer(shape).IsValid()


# --------------------------------------------------------------------------
# The hole is a real cylinder, carrying its nominal radius
# --------------------------------------------------------------------------


@pytest.mark.parametrize("radius", [6.35, 25.0, 1.5])
def test_a_bore_comes_back_as_a_cylinder_of_exactly_the_right_radius(tmp_path, radius):
    """Not "within 1e-6". Exactly, bit for bit.

    The radius is written as a decimal literal and read back as the same
    double, because it was never approximated by anything in between.
    """
    face = Face2D.create(rectangle(0.0, 0.0, 100.0, 80.0), [circle((50.0, 40.0), radius)])
    shape, _ = roundtrip(tmp_path, face, 8.0, f"bore_{radius}")

    radii = cylinder_radii(shape)
    assert radii, "the kernel found no cylindrical surface at all"
    assert radius in radii, f"kernel read back {radii}, expected an exact {radius}"


def test_a_bore_is_not_tessellated_into_planes(tmp_path):
    """The failure this package exists to avoid: a hole arriving as facets."""
    face = Face2D.create(rectangle(0.0, 0.0, 100.0, 60.0), [circle((50.0, 30.0), 6.35)])
    shape, _ = roundtrip(tmp_path, face, 8.0, "not_faceted")

    types = surface_types(shape)
    planes = sum(1 for t in types if t == GeomAbs_SurfaceType.GeomAbs_Plane)
    cylinders = sum(1 for t in types if t == GeomAbs_SurfaceType.GeomAbs_Cylinder)
    # Four walls plus two caps, and the bore. Nothing else.
    assert planes == 6
    assert cylinders == 1


def test_several_bores_all_survive_with_their_own_radii(tmp_path):
    wanted = [3.0, 4.5, 6.35]
    holes = [circle((20.0 + 30.0 * i, 30.0), r) for i, r in enumerate(wanted)]
    face = Face2D.create(rectangle(0.0, 0.0, 100.0, 60.0), holes)
    shape, _ = roundtrip(tmp_path, face, 8.0, "three_bores")

    assert sorted(cylinder_radii(shape)) == sorted(wanted)
    assert BRepCheck_Analyzer(shape).IsValid()


# --------------------------------------------------------------------------
# The kernel's volume agrees with the closed-form volume
# --------------------------------------------------------------------------


def test_the_kernel_integrates_the_same_volume_as_greens_theorem(tmp_path):
    """Two completely independent computations of the same number.

    Ours comes from Green's theorem on the 2D profile, in closed form and
    never touching a tessellation. OpenCASCADE's comes from integrating
    over the imported B-rep. They agree to the last few bits, which is the
    strongest end-to-end statement available: the solid the kernel built
    from the file is the solid the profile describes.
    """
    depth = 8.0
    face = Face2D.create(rectangle(0.0, 0.0, 100.0, 60.0), [circle((50.0, 30.0), 6.35)])
    shape, _ = roundtrip(tmp_path, face, depth, "volume")

    exact = (100.0 * 60.0 - math.pi * 6.35**2) * depth
    assert volume_of(shape) == pytest.approx(exact, rel=1e-11)


def test_volume_agrees_for_a_plate_riddled_with_holes(tmp_path):
    """Genus 7: seven independent through-holes, seven cylinders, one solid."""
    depth = 6.0
    radius = 4.0
    centres = [(15.0 + 20.0 * i, 30.0) for i in range(7)]
    face = Face2D.create(
        rectangle(0.0, 0.0, 160.0, 60.0), [circle(c, radius) for c in centres]
    )
    shape, _ = roundtrip(tmp_path, face, depth, "genus7")

    assert BRepCheck_Analyzer(shape).IsValid()
    assert len(cylinder_radii(shape)) == 7
    exact = (160.0 * 60.0 - 7 * math.pi * radius**2) * depth
    assert volume_of(shape) == pytest.approx(exact, rel=1e-11)


# --------------------------------------------------------------------------
# The other exact surface types
# --------------------------------------------------------------------------


def test_an_elliptical_hole_survives_as_a_curved_surface(tmp_path):
    """An ellipse extrudes to a surface of linear extrusion, not a cylinder,
    and the kernel has to accept that too."""
    hole = Loop2D(
        [EllipseArc2D((50.0, 30.0), (12.0, 0.0), 0.5, 0.0, 2.0 * math.pi)]
    )
    face = Face2D.create(rectangle(0.0, 0.0, 100.0, 60.0), [hole])
    depth = 5.0
    shape, _ = roundtrip(tmp_path, face, depth, "ellipse")

    assert BRepCheck_Analyzer(shape).IsValid()
    exact = (100.0 * 60.0 - math.pi * 12.0 * 6.0) * depth
    assert volume_of(shape) == pytest.approx(exact, rel=1e-9)


def test_a_bezier_profile_survives_as_a_swept_surface(tmp_path):
    """A free-form edge reaches the kernel as a real swept surface rather
    than as a polyline pretending to be one."""
    outline = Loop2D(
        [
            Line2D((0.0, 0.0), (100.0, 0.0)),
            Line2D((100.0, 0.0), (100.0, 40.0)),
            BezierCurve2D([(100.0, 40.0), (70.0, 80.0), (30.0, 0.0), (0.0, 40.0)]),
            Line2D((0.0, 40.0), (0.0, 0.0)),
        ]
    )
    face = Face2D.create(outline)
    shape, _ = roundtrip(tmp_path, face, 5.0, "bezier")

    assert BRepCheck_Analyzer(shape).IsValid()
    types = surface_types(shape)
    assert any(
        t
        in (
            GeomAbs_SurfaceType.GeomAbs_SurfaceOfExtrusion,
            GeomAbs_SurfaceType.GeomAbs_BezierSurface,
            GeomAbs_SurfaceType.GeomAbs_BSplineSurface,
        )
        for t in types
    ), f"the Bezier edge came back as {types}"


def test_a_filleted_outline_keeps_its_corner_radii(tmp_path):
    """Fillets are arcs on the outer profile, so they leave the kernel as
    cylinders of the fillet radius -- which is how a CAM system knows what
    tool will cut the corner."""
    r = 10.0
    w, h = 100.0, 60.0
    outline = Loop2D(
        [
            Line2D((r, 0.0), (w - r, 0.0)),
            Arc2D((w - r, r), r, -math.pi / 2, 0.0),
            Line2D((w, r), (w, h - r)),
            Arc2D((w - r, h - r), r, 0.0, math.pi / 2),
            Line2D((w - r, h), (r, h)),
            Arc2D((r, h - r), r, math.pi / 2, math.pi),
            Line2D((0.0, h - r), (0.0, r)),
            Arc2D((r, r), r, math.pi, 1.5 * math.pi),
        ]
    )
    depth = 4.0
    shape, _ = roundtrip(tmp_path, face := Face2D.create(outline), depth, "filleted")

    assert BRepCheck_Analyzer(shape).IsValid()
    assert cylinder_radii(shape) == [r] * 4
    assert volume_of(shape) == pytest.approx(face.area() * depth, rel=1e-11)


# --------------------------------------------------------------------------
# Topology, as the kernel counts it
# --------------------------------------------------------------------------


def test_the_kernel_agrees_the_solid_is_closed(tmp_path):
    """A shell that is not closed is the classic STEP export failure; it
    imports, it renders, and it has no volume."""
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.gp import gp_Pnt

    face = Face2D.create(rectangle(0.0, 0.0, 100.0, 60.0), [circle((50.0, 30.0), 6.35)])
    shape, _ = roundtrip(tmp_path, face, 8.0, "closed")

    classifier = BRepClass3d_SolidClassifier(shape)
    classifier.Perform(gp_Pnt(10.0, 10.0, 4.0), 1e-7)
    from OCP.TopAbs import TopAbs_State

    assert classifier.State() == TopAbs_State.TopAbs_IN, "a point in the material read as outside"

    classifier.Perform(gp_Pnt(50.0, 30.0, 4.0), 1e-7)  # down the bore
    assert classifier.State() == TopAbs_State.TopAbs_OUT, "the hole is not actually a hole"


def test_our_face_count_matches_the_kernels(tmp_path):
    """If OCCT splits or drops faces on import, the writer and the kernel
    disagree about what was written."""
    face = Face2D.create(
        rectangle(0.0, 0.0, 100.0, 60.0),
        [circle((30.0, 30.0), 5.0), circle((70.0, 30.0), 5.0)],
    )
    solid = extrude_face(face, 8.0, name="counts")
    path = tmp_path / "counts.step"
    write_step(solid, path, name="counts")
    shape = read_step(path)

    assert count(shape, TopAbs_ShapeEnum.TopAbs_FACE) == len(solid.faces())
