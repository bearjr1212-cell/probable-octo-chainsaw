import math

import cv2
import numpy as np
import pytest

from blueprint23d.parsers import raster_parser
from blueprint23d.reconstruct import extrude


def _draw_rect_with_hole(path, invert=False):
    img = np.full((140, 200), 255 if not invert else 0, dtype=np.uint8)
    ink = 0 if not invert else 255
    cv2.rectangle(img, (20, 20), (180, 120), ink, thickness=1)
    cv2.circle(img, (100, 70), 15, ink, thickness=1)
    cv2.imwrite(str(path), img)


def test_raster_rect_with_hole_recovers_approximate_shape(tmp_path):
    path = tmp_path / "blueprint.png"
    _draw_rect_with_hole(path)

    profile = raster_parser.load_profile(path)
    assert not profile.is_empty

    total_area = sum(p.area for p in profile.geoms)
    expected_area = 160 * 100 - math.pi * 15**2
    # Raster tracing is inherently approximate (finite line width, blur,
    # polygon simplification), so allow a generous tolerance.
    assert math.isclose(total_area, expected_area, rel_tol=0.15)

    mesh = extrude(profile, depth=3.0)
    assert mesh.is_watertight
    assert math.isclose(mesh.volume, total_area * 3.0, rel_tol=1e-6)


def test_raster_invert_flag_handles_light_on_dark(tmp_path):
    path = tmp_path / "inverted.png"
    _draw_rect_with_hole(path, invert=True)

    profile = raster_parser.load_profile(path, invert=True)
    assert not profile.is_empty
    total_area = sum(p.area for p in profile.geoms)
    expected_area = 160 * 100 - math.pi * 15**2
    assert math.isclose(total_area, expected_area, rel_tol=0.15)


def test_scale_from_reference():
    assert math.isclose(raster_parser.scale_from_reference(pixel_distance=100, real_distance=25.4), 0.254)
    with pytest.raises(ValueError):
        raster_parser.scale_from_reference(pixel_distance=0, real_distance=1)
