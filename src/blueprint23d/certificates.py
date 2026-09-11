"""What every reported number is worth, and how to prove the file is the file.

A CAD tool that prints ``radius 6.34982`` has said two things: a number,
and — silently — that the number is worth printing to five decimals. The
second claim is usually the false one. A radius read out of a DXF is worth
every digit, because nothing approximated it. The same radius recovered
from a scan is worth as many digits as the fit residual and the arc's own
curvature allow, which for a short shallow arc can be *no digits at all*.

Tools that do not distinguish these hand over the same-looking number in
both cases, and the shop quotes against it.

So this module produces a certificate: for every quantity the pipeline
reports, the value, how it was obtained, and a bound or an uncertainty on
it. Three kinds of statement appear, and they are deliberately not mixed:

* **Exact.** The value is what the file says, carried through arithmetic
  that cannot lose it. Bound zero. A DXF arc's radius; a polygon's area by
  Green's theorem; a mesh vertex on a planar cap.
* **Bounded.** The value differs from the truth by at most a proven
  amount. A tessellated arc deviates by at most ``r(1 − cos(Δ/2))`` — a
  theorem, not a measurement.
* **Estimated.** The value has an uncertainty derived from a model of the
  noise, which is the best available statement when the input was pixels.
  Labelled as an estimate, never as a bound.

The uncertainty on a fitted arc comes from propagating the fit residual
through the sagitta relation. For a chord ``c`` and sagitta ``s``,

.. code::

    r = c² / (8s) + s / 2        so        dr/ds = 1/2 − c² / (8s²)

and the sagitta, being an average over ``N`` residual-bearing points, is
itself uncertain by about ``σ/√N``. So

.. code::

    σ_r ≈ |1/2 − c²/(8s²)| · σ / √N

which is the formula that matters, because it *blows up as s → 0*. A
nearly-straight arc has an enormously uncertain radius no matter how
cleanly the points fit it, and this is the only honest way to say so: the
residual can be tiny while the radius is meaningless. A tool that reports
only the residual will happily certify ``r = 3000`` on an edge that is
straight.

Confidence is never a free-floating number here. It is always the fraction
of a *stated* tolerance the uncertainty leaves unspent, so "92% confident"
means "the uncertainty is 8% of the tolerance you asked for", and changing
the tolerance changes the number, as it should.

The second job of this module is the fingerprint. Two runs of the same
pipeline on the same input must produce the same part, and must be
provably the same part — not "looks the same when we open it". The
fingerprint is a SHA-256 over the exact geometry, canonicalised the same
way the revision comparison canonicalises it, so it is invariant to where
a loop starts and which way it runs and to nothing else. With
``SOURCE_DATE_EPOCH`` set, the emitted STEP file is byte-identical too.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .brep import Face2D, Loop2D, _quadrature_area_term, has_exact_area
from .curves import Arc2D, Curve2D, EllipseArc2D, Line2D

Point2 = Tuple[float, float]

#: How a value was arrived at. These are API; branch on them freely.
EXACT = "exact"
BOUNDED = "bounded"
ESTIMATED = "estimated"

#: Systematic contour offset left in raster extraction, in pixels.
#:
#: Averaging more points drives the *random* error down as 1/√N and does
#: nothing at all to a bias, because every point on the contour is
#: displaced the same way. So it is carried separately: a bias does not
#: combine in quadrature with a standard error and must not be quietly
#: added to one.
#:
#: The raw contour that ``findContours`` returns sits about 0.43 px inside
#: the true edge, because it is made of boundary *pixel centres*. Walking
#: each point to the gradient ridge removes almost all of that; what is
#: left, measured against analytically rendered ground truth across
#: circles, axis-aligned edges and 45° edges, is under 0.06 px and still
#: inward. This figure is rounded up from that, and
#: ``tests/test_fitting.py`` holds it there.
RASTER_SYSTEMATIC_BIAS = 0.08


# --------------------------------------------------------------------------
# One reported quantity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Measurement:
    """A number, and what it is worth.

    ``uncertainty`` is in the same units as ``value``. Its meaning depends
    on ``basis``: for :data:`EXACT` it is zero, for :data:`BOUNDED` it is a
    proven maximum error, and for :data:`ESTIMATED` it is a one-sigma
    figure from a noise model.
    """

    name: str
    value: float
    uncertainty: float
    basis: str
    method: str
    note: str = ""
    #: What the value measures: "length", "area", "volume", "angle" or
    #: "dimensionless". A tolerance is a length, so only length quantities
    #: can be compared against one -- an area uncertainty of 88 mm² next to
    #: a 1 mm tolerance is not a 0%-confident anything, it is a units error.
    dimension: str = "length"
    #: True when this error is one the caller *asked for* rather than one
    #: the pipeline could not avoid -- a mesh tessellated to 1e-3 deviates
    #: by nearly 1e-3 by construction, and rolling that into the part's
    #: confidence would make every certificate read as 1% confident.
    budgeted: bool = False

    @property
    def exact(self) -> bool:
        return self.basis == EXACT

    @property
    def relative(self) -> Optional[float]:
        if not self.value:
            return None
        return self.uncertainty / abs(self.value)

    def confidence(self, tolerance: float) -> float:
        """How much of ``tolerance`` this quantity's uncertainty leaves unspent.

        ``1.0`` means the uncertainty is nothing next to what was asked
        for; ``0.0`` means it has consumed the whole budget and the number
        should not be quoted against. Deliberately a function of a stated
        tolerance rather than a standalone score -- a radius good enough
        for a plasma table is not good enough for a reamed bore, and no
        single number can mean both.

        Only meaningful for length quantities; see :attr:`dimension`.
        """
        if tolerance <= 0.0:
            return 1.0 if self.uncertainty == 0.0 else 0.0
        if not math.isfinite(self.uncertainty):
            return 0.0
        return max(0.0, min(1.0, 1.0 - self.uncertainty / tolerance))

    def digits(self) -> int:
        """How many decimal places this value can honestly be printed to."""
        if self.uncertainty <= 0.0:
            return 12
        if not math.isfinite(self.uncertainty):
            return 0
        return max(0, int(math.floor(-math.log10(self.uncertainty))))

    def format(self) -> str:
        """The value printed to exactly the precision it supports."""
        return f"{self.value:.{min(self.digits(), 9)}f}"

    def describe(self) -> str:
        if self.exact:
            return f"{self.name} = {self.value:.9g} exactly ({self.method})"
        if self.basis == BOUNDED:
            return (
                f"{self.name} = {self.value:.9g}, error ≤ {self.uncertainty:.3e} "
                f"({self.method})"
            )
        return f"{self.name} = {self.format()} ± {self.uncertainty:.3e} ({self.method})"

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "name": self.name,
            "value": self.value,
            "uncertainty": self.uncertainty,
            "basis": self.basis,
            "method": self.method,
            "dimension": self.dimension,
            "significant_decimals": self.digits(),
        }
        if self.budgeted:
            data["budgeted"] = True
        if self.note:
            data["note"] = self.note
        return data


def exact(
    name: str, value: float, method: str, note: str = "", dimension: str = "length"
) -> Measurement:
    return Measurement(name, value, 0.0, EXACT, method, note, dimension=dimension)


# --------------------------------------------------------------------------
# Uncertainty of a fitted primitive
# --------------------------------------------------------------------------


def radius_uncertainty(sagitta: float, span: float, residual: float, points: int) -> float:
    """Propagate a fit residual through the sagitta relation to the radius.

    ``span`` is the chord, ``sagitta`` the bulge of the arc away from it.
    The result grows without limit as the sagitta approaches zero, which is
    the point: a shallow arc's radius is unknowable however tightly the
    points sit on it.
    """
    if points <= 0 or residual <= 0.0:
        return 0.0
    if sagitta <= 0.0 or not math.isfinite(sagitta):
        return math.inf
    sigma_sagitta = residual / math.sqrt(points)
    sensitivity = abs(0.5 - (span * span) / (8.0 * sagitta * sagitta))
    return sensitivity * sigma_sagitta


def position_uncertainty(residual: float, points: int) -> float:
    """Standard error of a least-squares fitted position: ``σ/√N``."""
    if points <= 0:
        return math.inf
    return residual / math.sqrt(points)


@dataclass(frozen=True)
class FeatureCertificate:
    """One recovered feature, and how far it can be trusted."""

    kind: str
    measurements: Tuple[Measurement, ...]
    resolvable: bool = True
    note: str = ""

    def of(self, name: str) -> Optional[Measurement]:
        return next((m for m in self.measurements if m.name == name), None)

    def confidence(self, tolerance: float) -> float:
        """The weakest measurement decides; a feature is as good as its worst number."""
        return min(
            (
                m.confidence(tolerance)
                for m in self.measurements
                if not m.budgeted and m.dimension == "length"
            ),
            default=1.0,
        )

    def describe(self, tolerance: Optional[float] = None) -> str:
        head = self.kind
        if tolerance is not None:
            head += f" [{self.confidence(tolerance) * 100:.0f}% of tol {tolerance:g}]"
        lines = [head] + ["    " + m.describe() for m in self.measurements]
        if not self.resolvable:
            lines.append("    NOT RESOLVABLE: " + (self.note or "below the noise floor"))
        elif self.note:
            lines.append("    " + self.note)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "kind": self.kind,
            "measurements": [m.to_dict() for m in self.measurements],
            "resolvable": self.resolvable,
        }
        if self.note:
            data["note"] = self.note
        return data


def certify_fit(report) -> FeatureCertificate:
    """Turn a :class:`~blueprint23d.fitting.FitReport` into a certificate."""
    points = report.points
    residual = report.rms

    if report.kind == "arc" and report.radius is not None:
        sagitta = report.sagitta or 0.0
        # Chord recovered from the sagitta and the radius: s(2r - s) = (c/2)^2.
        inner = max(0.0, sagitta * (2.0 * report.radius - sagitta))
        span = 2.0 * math.sqrt(inner)
        sigma_r = radius_uncertainty(sagitta, span, residual, points)
        resolvable = report.is_resolvable(residual)
        note = (
            ""
            if resolvable
            else (
                f"the arc bulges {sagitta:.4g} from its chord while the points "
                f"scatter by {residual:.4g}; the curvature is not measurable here "
                "and the radius should not be quoted"
            )
        )
        return FeatureCertificate(
            kind="arc",
            measurements=(
                Measurement(
                    "radius",
                    report.radius,
                    sigma_r,
                    ESTIMATED,
                    "Taubin fit refined by Gauss-Newton, sagitta error propagation",
                ),
                Measurement(
                    "centre position",
                    0.0,
                    position_uncertainty(residual, points),
                    ESTIMATED,
                    f"least squares over {points} sub-pixel points",
                ),
            ),
            resolvable=resolvable,
            note=note,
        )

    return FeatureCertificate(
        kind=report.kind,
        measurements=(
            Measurement(
                "offset",
                0.0,
                position_uncertainty(residual, points),
                ESTIMATED,
                f"total least squares over {points} sub-pixel points",
            ),
        ),
    )


# --------------------------------------------------------------------------
# Certifying geometry the pipeline already holds
# --------------------------------------------------------------------------


def certify_curve(curve: Curve2D, tolerance: float) -> FeatureCertificate:
    """What is known about a curve that came from a vector file.

    Nothing here is a measurement -- the file *stated* these numbers and
    nothing in the pipeline has touched them. The uncertainty that does
    appear is the tessellation deviation, which is a property of the
    output, not of the geometry.
    """
    measurements: List[Measurement] = []

    if isinstance(curve, Arc2D):
        measurements.append(exact("radius", curve.radius, "carried from the source entity"))
        measurements.append(
            exact(
                "sweep",
                abs(curve.sweep),
                "carried from the source entity",
                dimension="angle",
            )
        )
        measurements.append(exact("arc length", curve.length(), "r·Δθ in closed form"))
    elif isinstance(curve, Line2D):
        measurements.append(
            Measurement(
                "length",
                curve.length(),
                _hypot_rounding(curve.start, curve.end),
                BOUNDED,
                "hypot of exact endpoints, bounded by one rounding",
            )
        )
    elif isinstance(curve, EllipseArc2D):
        measurements.append(exact("semi-major", curve.major_length, "carried from the source"))
        measurements.append(exact("semi-minor", curve.minor_length, "carried from the source"))

    certificate = curve.deviation_certificate(tolerance)
    measurements.append(
        Measurement(
            "tessellation deviation",
            certificate.bound,
            certificate.bound,
            BOUNDED,
            "closed-form sagitta identity" if certificate.exact else "subdivision criterion",
            note=f"{certificate.segments} segments at chordal tolerance {tolerance:g}",
            budgeted=True,
        )
    )
    return FeatureCertificate(kind=type(curve).__name__, measurements=tuple(measurements))


def _hypot_rounding(a: Point2, b: Point2) -> float:
    """One ulp of the computed length: the only error in an exact segment."""
    length = math.dist(a, b)
    return abs(math.nextafter(length, math.inf) - length)


def certify_area(face: Face2D, quadrature_check: int = 48) -> Measurement:
    """The face's area, exact where Green's theorem closes in elementary terms.

    Where it does not -- Bézier and NURBS edges -- the area comes from
    Gauss-Legendre quadrature, and the uncertainty is estimated by
    recomputing at a finer subdivision and taking the difference. That is a
    convergence estimate, not a bound, and is labelled as one.
    """
    if face.is_area_exact():
        return exact("area", face.area(), "Green's theorem, closed form", dimension="area")

    coarse = face.area()
    fine = 0.0
    for index, loop in enumerate(face.loops()):
        total = 0.0
        for curve in loop.curves:
            if has_exact_area(curve):
                from .brep import curve_area_term

                total += curve_area_term(curve)
            else:
                total += _quadrature_area_term(curve, subdivisions=quadrature_check)
        fine += abs(total) if index == 0 else -abs(total)

    return Measurement(
        "area",
        coarse,
        abs(fine - coarse),
        ESTIMATED,
        f"Gauss-Legendre quadrature, Richardson check at {quadrature_check} subdivisions",
        note="a free-form edge has no elementary closed form for the area integral",
        dimension="area",
    )


# --------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------


def _canonical(value: Any) -> Any:
    if isinstance(value, float):
        # repr() of a float round-trips exactly, and hex() would too, but
        # repr keeps the fingerprint's inputs readable when debugging.
        return repr(value)
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    return value


def loop_fingerprint(loop: Loop2D) -> str:
    """SHA-256 of a loop's exact geometry.

    Canonicalised over where the loop starts, which way it runs, and the
    phase of a full circle -- all artefacts of whoever exported it -- and
    over nothing else. Two loops with the same fingerprint are the same
    loop; two with different fingerprints differ somewhere in a stated
    coordinate.
    """
    from .revisions import loop_signature

    payload = json.dumps(_canonical(loop_signature(loop)), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def face_fingerprint(face: Face2D) -> str:
    """SHA-256 of a face: its outer loop and every hole, holes sorted."""
    inner = sorted(loop_fingerprint(loop) for loop in face.inners)
    payload = json.dumps([loop_fingerprint(face.outer), inner], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def drawing_fingerprint(faces: Sequence[Face2D]) -> str:
    """SHA-256 of a whole drawing, independent of the order the faces arrived in."""
    payload = json.dumps(sorted(face_fingerprint(f) for f in faces), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# The whole document
# --------------------------------------------------------------------------


@dataclass
class Certificate:
    """Everything the pipeline is willing to claim about one part."""

    source: Optional[str] = None
    provenance: str = "vector"  # "vector" or "raster"
    fingerprint: str = ""
    measurements: List[Measurement] = field(default_factory=list)
    features: List[FeatureCertificate] = field(default_factory=list)
    tolerance: float = 1e-3

    def _unbudgeted(self) -> List[Measurement]:
        out = [m for m in self.measurements if not m.budgeted]
        out += [m for f in self.features for m in f.measurements if not m.budgeted]
        return out

    @property
    def exact(self) -> bool:
        """True when no reported number rests on a noise model.

        Not the same as "every uncertainty is zero": the length of a line
        between two exact endpoints still carries the one rounding of the
        square root, and saying so is the point of the module. What this
        asks is whether the part was *read* or *measured*.
        """
        return all(m.basis != ESTIMATED for m in self._unbudgeted())

    @property
    def worst_uncertainty(self) -> float:
        """The largest error the pipeline could not avoid.

        Excludes budgeted errors -- a mesh deviating by just under the
        tolerance it was asked for is the tessellator working, not a defect.
        """
        return max(
            (m.uncertainty for m in self._unbudgeted() if m.dimension == "length"),
            default=0.0,
        )

    @property
    def unresolvable(self) -> List[FeatureCertificate]:
        return [f for f in self.features if not f.resolvable]

    def confidence(self) -> float:
        """The weakest link, against this certificate's stated tolerance."""
        scores = [
            m.confidence(self.tolerance)
            for m in self.measurements
            if not m.budgeted and m.dimension == "length"
        ]
        scores += [f.confidence(self.tolerance) for f in self.features]
        return min(scores, default=1.0)

    def of(self, name: str) -> Optional[Measurement]:
        return next((m for m in self.measurements if m.name == name), None)

    def describe(self, limit: Optional[int] = 12) -> str:
        lines = [f"{self.source or 'part'}  [{self.provenance}]"]
        lines.append(f"  fingerprint {self.fingerprint[:16]}…")
        verdict = (
            "exact throughout"
            if self.exact
            else f"worst uncertainty {self.worst_uncertainty:.3e}, "
            f"confidence {self.confidence() * 100:.0f}% of tolerance {self.tolerance:g}"
        )
        lines.append(f"  {verdict}")
        for measurement in self.measurements:
            lines.append("  " + measurement.describe())
        shown = self.features if limit is None else self.features[:limit]
        for feature in shown:
            lines.append("  " + feature.describe(self.tolerance).replace("\n", "\n  "))
        remaining = len(self.features) - len(shown)
        if remaining > 0:
            lines.append(f"  ... and {remaining} more features")
        if self.unresolvable:
            lines.append(
                f"  {len(self.unresolvable)} feature(s) could not be resolved above the noise"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "provenance": self.provenance,
            "fingerprint": self.fingerprint,
            "tolerance": self.tolerance,
            "exact": self.exact,
            "worst_uncertainty": self.worst_uncertainty,
            "confidence": self.confidence(),
            "measurements": [m.to_dict() for m in self.measurements],
            "features": [f.to_dict() for f in self.features],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def certify_face(
    face: Face2D,
    tolerance: float = 1e-3,
    source: Optional[str] = None,
    fit_reports: Sequence = (),
    depth: Optional[float] = None,
    pixel_scale: float = 1.0,
) -> Certificate:
    """Certify one face, and the solid it would extrude to.

    Pass ``fit_reports`` when the face came from raster input; the
    certificate then reports the recovered primitives as estimates with
    propagated uncertainties instead of as exact values, which is the whole
    distinction the module exists to keep. ``pixel_scale`` (drawing units
    per pixel) converts the systematic extraction bias into drawing units.
    """
    raster = bool(fit_reports)
    bias = RASTER_SYSTEMATIC_BIAS * pixel_scale if raster else 0.0
    scatter = max(
        (position_uncertainty(r.rms, r.points) for r in fit_reports), default=0.0
    )
    # How far the recovered boundary can sit from the true one, anywhere.
    offset = bias + scatter if raster else 0.0

    area = certify_area(face)
    if raster:
        # A boundary displaced by delta changes the enclosed area by about
        # perimeter x delta to first order. Green's theorem is still exact
        # on the curves that were fitted; what is uncertain is the curves.
        perimeter = sum(loop.length() for loop in face.loops())
        area = Measurement(
            "area",
            face.area(),
            perimeter * offset,
            ESTIMATED,
            "Green's theorem on the fitted curves; a boundary offset δ moves "
            "the area by ≈ perimeter·δ",
            dimension="area",
        )
    measurements = [area]

    x0, y0, x1, y1 = face.bounds()
    extent_basis = ESTIMATED if raster else EXACT
    # Both opposite edges are displaced, so an extent carries twice the offset.
    extent_sigma = 2.0 * offset
    measurements.append(
        Measurement("width", x1 - x0, extent_sigma, extent_basis, "extent of the curve bounds")
    )
    measurements.append(
        Measurement("height", y1 - y0, extent_sigma, extent_basis, "extent of the curve bounds")
    )

    if depth is not None:
        measurements.append(
            Measurement(
                "volume",
                face.area() * depth,
                area.uncertainty * depth,
                area.basis,
                "area × depth; extrusion introduces no error of its own",
                dimension="volume",
            )
        )

    deviation = max(
        (loop.deviation_bound(tolerance) for loop in face.loops()), default=0.0
    )
    measurements.append(
        Measurement(
            "mesh deviation",
            deviation,
            deviation,
            BOUNDED,
            "largest proven per-curve chordal bound; planar caps contribute zero",
            note=f"only applies to mesh output at tolerance {tolerance:g}",
            budgeted=True,
        )
    )

    if raster:
        measurements.append(
            Measurement(
                "systematic contour bias",
                bias,
                bias,
                BOUNDED,
                "thresholded edge extraction, measured against rendered ground truth",
                note=(
                    "every contour point is displaced inward by about this much. "
                    "It is a bias, not noise: fitting more points does not reduce "
                    "it, and it does not combine in quadrature with the standard "
                    "errors below -- read them as a spread about a value that is "
                    "itself offset"
                ),
            )
        )
        features = [certify_fit(report) for report in fit_reports]
    else:
        features = [
            certify_curve(curve, tolerance)
            for loop in face.loops()
            for curve in loop.curves
        ]

    return Certificate(
        source=source,
        provenance="raster" if raster else "vector",
        fingerprint=face_fingerprint(face),
        measurements=measurements,
        features=features,
        tolerance=tolerance,
    )


def certify_drawing(
    path,
    layer: Optional[str] = None,
    tolerance: float = 1e-3,
    depth: Optional[float] = None,
    scale: float = 1.0,
    invert: bool = False,
    fit_tolerance: float = 0.6,
) -> Certificate:
    """Read a drawing and certify the largest part on it."""
    from pathlib import Path

    from .loaders import RASTER_EXTENSIONS, load_faces, load_raster_faces

    path = Path(path)
    if path.suffix.lower() in RASTER_EXTENSIONS:
        faces, reports = load_raster_faces(
            path, scale=scale, invert=invert, fit_tolerance=fit_tolerance
        )
    else:
        faces, _ = load_faces(path, layer=layer)
        reports = []

    if not faces:
        raise ValueError(f"no closed profile found in {path}")

    face = max(faces, key=lambda f: f.area())
    certificate = certify_face(
        face,
        tolerance=tolerance,
        source=str(path),
        fit_reports=reports,
        depth=depth,
        pixel_scale=scale,
    )
    # The fingerprint covers the whole drawing, not just the part chosen.
    certificate.fingerprint = drawing_fingerprint(faces)
    return certificate
