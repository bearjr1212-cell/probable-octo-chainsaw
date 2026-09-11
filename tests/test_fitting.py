"""Raster reconstruction: does fitting actually beat the pixel grid?

The claim being tested is quantitative. A contour traced from an image is
quantised to whole pixels, so any single point carries roughly half a pixel
of error. Fitting a primitive to N such points should do far better than
that, because the per-point errors are independent and average down. These
tests render shapes of exactly known geometry, recover them, and check the
recovered numbers against the truth.
"""

import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
fitkernels = pytest.importorskip(
    "blueprint23d._native.fitkernels", reason="compiled fitting kernel not built"
)

from blueprint23d.curves import Arc2D, Line2D  # noqa: E402
from blueprint23d.fitting import fit_chain, loop_from_points, refine_subpixel  # noqa: E402


def render_contour(draw, size=(200, 200), supersample=8):
    """Render a shape anti-aliased, threshold it, and return a refined contour.

    Supersampling then downscaling reproduces the grey edge pixels a real
    scan has, which is what the sub-pixel step needs in order to work.
    """
    h, w = size
    big = np.zeros((h * supersample, w * supersample), np.uint8)
    draw(big, supersample)
    img = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    raw = [(float(p[0][0]), float(p[0][1])) for p in contours[0]]
    return img, raw, refine_subpixel(img, raw)


def draw_circle(radius, cx=100, cy=100):
    def draw(big, s):
        cv2.circle(big, (cx * s, cy * s), int(radius * s), 255, -1)

    return draw


def draw_rounded_rect(big, s):
    w, h, r = 340, 220, 45
    cv2.rectangle(big, ((40 + r) * s, 40 * s), ((40 + w - r) * s, (40 + h) * s), 255, -1)
    cv2.rectangle(big, (40 * s, (40 + r) * s), ((40 + w) * s, (40 + h - r) * s), 255, -1)
    for cx, cy in [(40 + r, 40 + r), (40 + w - r, 40 + r), (40 + r, 40 + h - r), (40 + w - r, 40 + h - r)]:
        cv2.circle(big, (cx * s, cy * s), r * s, 255, -1)


# --------------------------------------------------------------------------
# Fitting numerics
# --------------------------------------------------------------------------


def test_circle_fit_error_follows_the_inverse_sqrt_n_law():
    """A fit over N points must do much better than any single point.

    With independent per-point error the radius estimate has standard
    error about noise/sqrt(N). That is a statistical claim, so it is
    checked over many trials rather than one -- a single draw can land a
    couple of standard errors out and says nothing either way.
    """
    import random
    import statistics

    radius, noise, count = 50.0, 0.5, 200
    errors = []
    for seed in range(40):
        rng = random.Random(seed)
        points = [
            (
                100 + radius * math.cos(2 * math.pi * i / count) + rng.gauss(0, noise),
                60 + radius * math.sin(2 * math.pi * i / count) + rng.gauss(0, noise),
            )
            for i in range(count)
        ]
        errors.append(abs(fitkernels.fit_circle(points)["radius"] - radius))

    rms_error = math.sqrt(statistics.fmean(e * e for e in errors))
    predicted = noise / math.sqrt(count)

    # Within a factor of three of the theoretical standard error, and at
    # least an order of magnitude better than any single measurement.
    assert rms_error < 3.0 * predicted, f"{rms_error:.4f} vs predicted {predicted:.4f}"
    assert rms_error < noise / 10.0


def test_line_fit_is_rotation_invariant():
    """Total least squares must give the same answer at every angle.

    Ordinary least squares minimises vertical offsets, so its answer
    changes when the sheet is rotated and it fails outright on a vertical
    edge. A drawing has no privileged axis.
    """
    import random

    random.seed(5)
    for degrees in (0, 30, 60, 89, 90, 135):
        theta = math.radians(degrees)
        points = [
            (50 + t * math.cos(theta) + random.gauss(0, 0.2), 50 + t * math.sin(theta) + random.gauss(0, 0.2))
            for t in range(-40, 41)
        ]
        fit = fitkernels.fit_line(points)
        recovered = math.degrees(math.atan2(fit["direction"][1], fit["direction"][0])) % 180
        expected = degrees % 180
        error = min(abs(recovered - expected), 180 - abs(recovered - expected))
        assert error < 0.5, f"{degrees} deg recovered as {recovered}"


def test_circle_fit_reports_its_conditioning():
    """Every fit carries the sagitta, so an unidentifiable arc can be rejected."""
    points = [(math.cos(t / 100.0), math.sin(t / 100.0)) for t in range(30)]
    fit = fitkernels.fit_circle(points)
    assert "sagitta" in fit and "span" in fit
    assert fit["sagitta"] >= 0.0


def test_collinear_points_have_no_circle():
    assert fitkernels.fit_circle([(float(i), 0.0) for i in range(20)]) is None


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------


def test_straight_edges_never_become_giant_arcs():
    """The regression that motivated the sagitta guard.

    A noisy straight edge can be fitted by a circle of enormous radius
    within any tolerance. Accepting that turns a flat face into a 3-metre
    arc. Curvature must be resolvable above the noise, not merely fittable.
    """
    _, _, refined = render_contour(
        lambda big, s: cv2.rectangle(big, (40 * s, 40 * s), (260 * s, 160 * s), 255, -1),
        size=(200, 300),
        supersample=4,
    )
    curves, _ = fit_chain(refined, tolerance=0.5, min_arc_points=8)
    assert all(isinstance(c, Line2D) for c in curves)


def test_a_rendered_circle_recovers_as_one_arc():
    """Not thirty line segments: a circle is a circle."""
    for radius in (20.0, 37.5, 60.0):
        _, _, refined = render_contour(draw_circle(radius))
        curves, reports = fit_chain(refined, tolerance=0.5, min_arc_points=8)
        arcs = [c for c in curves if isinstance(c, Arc2D)]
        assert len(arcs) == 1, f"r={radius} gave {len(curves)} primitives"
        # Sub-pixel accuracy on a shape whose edge was only ever known to
        # the nearest pixel.
        assert abs(arcs[0].radius - radius) < 0.5


def test_rounded_rectangle_recovers_its_structure():
    """Four straight edges and four fillets, from an image."""
    _, _, refined = render_contour(draw_rounded_rect, size=(300, 420), supersample=4)
    curves, reports = fit_chain(refined, tolerance=0.6, min_arc_points=8)

    arcs = [c for c in curves if isinstance(c, Arc2D)]
    lines = [c for c in curves if isinstance(c, Line2D)]
    assert len(arcs) == 4, f"expected 4 fillets, got {len(arcs)}"
    assert len(lines) == 4, f"expected 4 edges, got {len(lines)}"
    for arc in arcs:
        assert abs(arc.radius - 45.0) < 6.0


def test_recovered_loop_area_matches_the_rendered_shape():
    _, _, refined = render_contour(draw_rounded_rect, size=(300, 420), supersample=4)
    loop, _ = loop_from_points(refined, tolerance=0.6, min_arc_points=8)
    true_area = 340 * 220 - (4 - math.pi) * 45 * 45
    assert abs(loop.area() - true_area) / true_area < 0.01
    # The recovered loop is made of exact primitives, so its area comes
    # from the closed-form integral rather than a polygon sum.
    assert loop.is_area_exact()


# --------------------------------------------------------------------------
# Sub-pixel refinement
# --------------------------------------------------------------------------


def test_subpixel_refinement_improves_on_integer_contours():
    """Refined points must recover the radius better than whole-pixel ones."""
    improvements = []
    for radius in (40.0, 25.5, 12.25):
        _, raw, refined = render_contour(draw_circle(radius))
        raw_fit = fitkernels.fit_circle(raw)
        refined_fit = fitkernels.fit_circle(refined)
        raw_error = abs(raw_fit["radius"] - radius)
        refined_error = abs(refined_fit["radius"] - radius)
        improvements.append(raw_error / max(refined_error, 1e-9))
    assert all(i > 1.0 for i in improvements), f"no improvement: {improvements}"


def test_subpixel_offsets_stay_within_half_a_pixel():
    """The parabola vertex is only meaningful inside the sample triple."""
    img, raw, refined = render_contour(draw_circle(30.0))
    for (rx, ry), (fx, fy) in zip(raw, refined):
        assert math.hypot(fx - rx, fy - ry) <= 1.0 + 1e-9


def test_refine_handles_empty_input():
    img = np.zeros((10, 10), np.uint8)
    assert refine_subpixel(img, []) == []


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_raster_reaches_step_as_a_real_cylinder():
    """The whole point: a scanned hole exports as a cylinder, not triangles."""
    from blueprint23d.brep import Face2D, extrude_face
    from blueprint23d.step_writer import StepWriter, summarize_step, validate_step

    _, _, refined = render_contour(draw_circle(37.5))
    loop, _ = loop_from_points(refined, tolerance=0.5, min_arc_points=8)
    solid = extrude_face(Face2D.create(loop), 10.0, name="scanned_disc")

    text = StepWriter(name="scanned_disc").render(solid)
    assert validate_step(text) == []
    counts = summarize_step(text)
    assert counts["CYLINDRICAL_SURFACE"] == 1
    assert counts["CIRCLE"] == 2


def test_fit_chain_rejects_tiny_input():
    with pytest.raises(ValueError):
        fit_chain([(0.0, 0.0), (1.0, 0.0)], tolerance=0.5)


# ==========================================================================
# The systematic bias
# ==========================================================================


def analytic_disc(radius, size=200, supersample=16, centre=100.0):
    """A disc rendered by exact coverage, not by a rasteriser.

    ``cv2.circle`` draws a Bresenham circle, whose own quantisation is
    several times larger than the error being measured here -- it swamps
    the measurement and looks like scatter in the reconstruction. Ground
    truth for a sub-pixel claim has to be better than the claim.
    """
    yy, xx = np.mgrid[0 : size * supersample, 0 : size * supersample]
    x = (xx + 0.5) / supersample
    y = (yy + 0.5) / supersample
    inside = ((x - centre) ** 2 + (y - centre) ** 2) <= radius**2
    big = (inside * 255).astype(np.uint8)
    return cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)


def analytic_rectangle(x0, x1, y0, y1, size=200, supersample=16):
    yy, xx = np.mgrid[0 : size * supersample, 0 : size * supersample]
    x = (xx + 0.5) / supersample
    y = (yy + 0.5) / supersample
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return cv2.resize((inside * 255).astype(np.uint8), (size, size), interpolation=cv2.INTER_AREA)


def traced(image):
    """The raw pixel contour, as the pipeline gets it."""
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    _, ink = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [(float(p[0][0]), float(p[0][1])) for p in contours[0]]


def radial_error(points, radius, centre=100.0):
    """Signed distance from the true circle, in pixels.

    Pixel *j* covers world ``[j, j+1)``, so its centre is at ``j + 0.5``.
    """
    return np.array(
        [math.hypot(x + 0.5 - centre, y + 0.5 - centre) - radius for x, y in points]
    )


@pytest.mark.parametrize("radius", [20.0, 30.0, 37.5, 50.0])
def test_refinement_removes_the_contour_tracers_systematic_offset(radius):
    """The bias that no amount of averaging removes.

    ``findContours`` returns the centres of boundary *pixels*, which sit
    about 0.43 px inside the true edge -- the same way for every point on
    the contour, so it survives fitting, survives averaging over a
    thousand points, and comes out as a hole that is consistently
    undersize. Only moving the points onto the actual gradient ridge
    removes it.
    """
    image = analytic_disc(radius)
    raw = radial_error(traced(image), radius)
    refined = radial_error(refine_subpixel(image, traced(image)), radius)

    assert raw.mean() < -0.3  # the tracer's own offset, inward
    assert abs(refined.mean()) < 0.08  # what is left after the walk
    assert abs(refined.mean()) < abs(raw.mean()) / 4.0


def test_the_refined_scatter_is_a_small_fraction_of_a_pixel():
    image = analytic_disc(30.0)
    refined = radial_error(refine_subpixel(image, traced(image)), 30.0)
    assert refined.std() < 0.06


def test_a_straight_edge_is_localised_as_well_as_a_curved_one():
    """Nothing here should privilege one edge orientation.

    The rectangle is deliberately placed off the pixel grid, so an
    implementation that quietly snaps to pixel boundaries fails.
    """
    x0, x1, y0, y1 = 40.3, 160.7, 30.2, 150.9
    image = analytic_rectangle(x0, x1, y0, y1)
    points = np.array(refine_subpixel(image, traced(image)))

    for axis, low, high in ((0, x0, x1), (1, y0, y1)):
        world = points[:, axis] + 0.5
        near_low = world[np.abs(world - low) < 1.5]
        near_high = world[np.abs(world - high) < 1.5]
        assert abs(near_low.mean() - low) < 0.08
        assert abs(near_high.mean() - high) < 0.08


def test_a_diagonal_edge_is_localised_without_bias():
    """A 45° edge crosses the pixel grid at every sample, which is the
    worst case for anything that interpolates along an axis."""
    size, supersample = 200, 16
    yy, xx = np.mgrid[0 : size * supersample, 0 : size * supersample]
    x = (xx + 0.5) / supersample
    y = (yy + 0.5) / supersample
    diamond = (np.abs(x - 100.0) + np.abs(y - 100.0)) <= 60.0
    image = cv2.resize((diamond * 255).astype(np.uint8), (size, size), interpolation=cv2.INTER_AREA)

    points = np.array(refine_subpixel(image, traced(image)))
    # L1 residual over sqrt(2) is the perpendicular distance to the edge.
    error = (
        np.abs(points[:, 0] + 0.5 - 100.0) + np.abs(points[:, 1] + 0.5 - 100.0) - 60.0
    ) / math.sqrt(2.0)
    assert abs(error.mean()) < 0.05
    assert error.std() < 0.08


def test_the_certificate_quotes_the_bias_that_is_actually_left():
    """The certificate's figure is a claim about this code, and drifting
    apart silently is how a certificate becomes decoration."""
    from blueprint23d.certificates import RASTER_SYSTEMATIC_BIAS

    worst = max(
        abs(radial_error(refine_subpixel(analytic_disc(r), traced(analytic_disc(r))), r).mean())
        for r in (20.0, 30.0, 50.0)
    )
    assert worst <= RASTER_SYSTEMATIC_BIAS
    assert RASTER_SYSTEMATIC_BIAS < 2.0 * worst  # and not wildly padded either
