import math

import ezdxf
import pytest

from blueprint23d.parsers import dxf_parser
from blueprint23d.reconstruct import extrude


def _write_rect_with_hole(path, layer="0"):
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline(
        [(0, 0), (10, 0), (10, 6), (0, 6)], close=True, dxfattribs={"layer": layer}
    )
    msp.add_circle((5, 3), 1.0, dxfattribs={"layer": layer})
    doc.saveas(path)


def _write_rect_from_loose_lines(path):
    doc = ezdxf.new()
    msp = doc.modelspace()
    # Four independent LINE entities, out of order, as many real exports
    # produce rather than a single closed polyline.
    msp.add_line((10, 10), (10, 0))
    msp.add_line((0, 0), (10, 0))
    msp.add_line((0, 10), (0, 0))
    msp.add_line((10, 10), (0, 10))
    doc.saveas(path)


def test_dxf_rect_with_hole_extrudes_to_expected_volume(tmp_path):
    path = tmp_path / "part.dxf"
    _write_rect_with_hole(path)

    profile = dxf_parser.load_profile(path)
    mesh = extrude(profile, depth=4.0)

    expected_area = 10 * 6 - math.pi * 1.0**2
    assert mesh.is_watertight
    assert math.isclose(mesh.volume, expected_area * 4.0, rel_tol=1e-3)


def test_dxf_welds_independent_line_entities(tmp_path):
    path = tmp_path / "loose.dxf"
    _write_rect_from_loose_lines(path)

    profile = dxf_parser.load_profile(path)
    assert len(profile.geoms) == 1
    assert math.isclose(profile.geoms[0].area, 100.0, rel_tol=1e-9)

    mesh = extrude(profile, depth=2.0)
    assert mesh.is_watertight
    assert math.isclose(mesh.volume, 200.0, rel_tol=1e-6)


def test_dxf_layer_filter(tmp_path):
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (10, 0), (10, 10), (0, 10)], close=True, dxfattribs={"layer": "PART"})
    msp.add_lwpolyline([(20, 20), (25, 20), (25, 25), (20, 25)], close=True, dxfattribs={"layer": "DIMENSIONS"})
    path = tmp_path / "layered.dxf"
    doc.saveas(path)

    profile = dxf_parser.load_profile(path, layer="PART")
    assert len(profile.geoms) == 1
    assert math.isclose(profile.geoms[0].area, 100.0, rel_tol=1e-9)


def test_extrude_raises_on_empty_profile(tmp_path):
    doc = ezdxf.new()
    doc.saveas(tmp_path / "empty.dxf")
    profile = dxf_parser.load_profile(tmp_path / "empty.dxf")
    with pytest.raises(ValueError):
        extrude(profile, depth=1.0)
