"""blueprint23d: turn 2D blueprints into 3D parts, exactly where possible.

The pipeline keeps geometry analytic from the file to the output. A DXF
circle stays a circle, becomes a cylindrical surface when extruded, and
reaches a STEP file carrying its nominal radius -- so a hole in the output
is a hole with a diameter, not triangles near where one should be. Meshes
are produced only when a mesh is asked for, and then with a proven
deviation bound.

Typical use::

    from blueprint23d import load_face, extrude_face, write_step

    face = load_face("plate.dxf")          # exact curves, holes nested
    solid = extrude_face(face, depth=8.0)  # arcs become true cylinders
    write_step(solid, "plate.step")        # real CAD, not a mesh

See the README for the mathematics and for where the guarantees stop.
"""

from .assembly import faces_from_curves, nest_loops, weld_curves
from .booleans2d import clip_face_to_halfplane, clip_face_to_slab
from .brep import Face2D, Loop2D, Solid, extrude_face
from .curves import (
    Arc2D,
    BezierCurve2D,
    Curve2D,
    DeviationCertificate,
    EllipseArc2D,
    Line2D,
    NurbsCurve2D,
)
# Deliberately not re-exporting ``audit.audit`` at the package root: the
# name would shadow the ``blueprint23d.audit`` submodule itself.
from .audit import Finding, audit_drawing
from .certificates import (
    Certificate,
    FeatureCertificate,
    Measurement,
    certify_drawing,
    certify_face,
    drawing_fingerprint,
    face_fingerprint,
)
from .diagnostics import Code, Defect, Report, Severity
from .exact import BACKEND as PREDICATE_BACKEND
from .loaders import load_face, load_faces
from .multiview import Reconstruction, ViewSpec, reconstruct
from .processes import (
    PROCESSES,
    Process,
    Quote,
    check_manufacturability,
    compare,
    process,
    quote,
    quote_drawing,
)
from .revisions import Change, Diff, Feature, diff_drawings, diff_faces
from .step_writer import validate_step, write_step
from .tessellate import MeshCertificate, TriangleMesh, tessellate_extrusion
from .units import Unit, infer_from_dxf
from .validate import validate_curves, validate_drawing, validate_face

__all__ = [
    # reading
    "load_face",
    "load_faces",
    # exact curves
    "Curve2D",
    "Line2D",
    "Arc2D",
    "EllipseArc2D",
    "BezierCurve2D",
    "NurbsCurve2D",
    "DeviationCertificate",
    # topology
    "Loop2D",
    "Face2D",
    "Solid",
    "extrude_face",
    "weld_curves",
    "nest_loops",
    "faces_from_curves",
    # operations
    "clip_face_to_halfplane",
    "clip_face_to_slab",
    "reconstruct",
    "ViewSpec",
    "Reconstruction",
    # output
    "write_step",
    "validate_step",
    "tessellate_extrusion",
    "TriangleMesh",
    "MeshCertificate",
    # intake QA
    "Report",
    "Defect",
    "Code",
    "Severity",
    "validate_drawing",
    "validate_face",
    "validate_curves",
    "Unit",
    "infer_from_dxf",
    "audit_drawing",
    "Finding",
    # certificates and determinism
    "certify_drawing",
    "certify_face",
    "Certificate",
    "FeatureCertificate",
    "Measurement",
    "face_fingerprint",
    "drawing_fingerprint",
    # processes and cost
    "PROCESSES",
    "Process",
    "Quote",
    "process",
    "quote",
    "quote_drawing",
    "compare",
    "check_manufacturability",
    # revision comparison
    "diff_drawings",
    "diff_faces",
    "Diff",
    "Change",
    "Feature",
    # diagnostics
    "PREDICATE_BACKEND",
]

__version__ = "0.2.0"
