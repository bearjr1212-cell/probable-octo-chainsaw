import math

from blueprint23d.parsers import svg_parser
from blueprint23d.reconstruct import extrude

SVG_RECT_WITH_HOLE = """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <path d="M 0,0 L 10,0 L 10,6 L 0,6 Z" />
  <circle cx="5" cy="3" r="1" />
</svg>
"""

SVG_GROUP_WITH_TRANSFORM = """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <g transform="translate(20,0)">
    <rect x="0" y="0" width="4" height="4" />
  </g>
</svg>
"""


def test_svg_rect_with_hole_extrudes_to_expected_volume(tmp_path):
    path = tmp_path / "part.svg"
    path.write_text(SVG_RECT_WITH_HOLE)

    profile = svg_parser.load_profile(path)
    mesh = extrude(profile, depth=4.0)

    expected_area = 10 * 6 - math.pi * 1.0**2
    assert mesh.is_watertight
    assert math.isclose(mesh.volume, expected_area * 4.0, rel_tol=1e-2)


def test_svg_group_transform_is_applied(tmp_path):
    path = tmp_path / "grouped.svg"
    path.write_text(SVG_GROUP_WITH_TRANSFORM)

    profile = svg_parser.load_profile(path)
    assert len(profile.geoms) == 1
    minx, miny, maxx, maxy = profile.geoms[0].bounds
    assert math.isclose(minx, 20.0, abs_tol=1e-6)
    assert math.isclose(maxx, 24.0, abs_tol=1e-6)
