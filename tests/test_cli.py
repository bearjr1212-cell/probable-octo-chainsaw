"""CLI behaviour, including that the output format decides what survives."""

import math
import re

import ezdxf
import pytest

from blueprint23d.cli import main
from blueprint23d.step_writer import summarize_step, validate_step


@pytest.fixture
def plate_dxf(tmp_path):
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline(
        [(0, 0), (100, 0), (100, 60), (0, 60)], close=True, dxfattribs={"layer": "OUTLINE"}
    )
    msp.add_circle((25, 30), 6.35, dxfattribs={"layer": "OUTLINE"})
    msp.add_circle((75, 30), 6.35, dxfattribs={"layer": "OUTLINE"})
    msp.add_text("100.00", dxfattribs={"layer": "DIMS"})
    path = tmp_path / "top.dxf"
    doc.saveas(path)
    return path


@pytest.fixture
def front_dxf(tmp_path):
    doc = ezdxf.new()
    doc.modelspace().add_lwpolyline([(0, 0), (100, 0), (100, 8), (0, 8)], close=True)
    path = tmp_path / "front.dxf"
    doc.saveas(path)
    return path


def test_extrude_to_step_keeps_the_bore_exact(plate_dxf, tmp_path):
    out = tmp_path / "plate.step"
    assert main(["extrude", "--input", str(plate_dxf), "--layer", "OUTLINE",
                 "--depth", "8", "--output", str(out)]) == 0

    text = out.read_text()
    assert validate_step(text) == []
    assert summarize_step(text)["CYLINDRICAL_SURFACE"] == 2
    assert "6.35" in set(re.findall(r"CYLINDRICAL_SURFACE\('',#\d+,([0-9.E+-]+)\)", text))


def test_extrude_to_mesh_is_watertight_and_certified(plate_dxf, tmp_path):
    trimesh = pytest.importorskip("trimesh")
    out = tmp_path / "plate.stl"
    assert main(["extrude", "--input", str(plate_dxf), "--layer", "OUTLINE",
                 "--depth", "8", "--output", str(out), "--tolerance", "1e-4"]) == 0

    mesh = trimesh.load(str(out))
    assert mesh.is_watertight
    exact = (100 * 60 - 2 * math.pi * 6.35**2) * 8
    assert abs(mesh.volume - exact) / exact < 1e-4


def test_tolerance_actually_changes_the_mesh(plate_dxf, tmp_path):
    trimesh = pytest.importorskip("trimesh")
    coarse, fine = tmp_path / "coarse.stl", tmp_path / "fine.stl"
    main(["extrude", "--input", str(plate_dxf), "--layer", "OUTLINE", "--depth", "8",
          "--output", str(coarse), "--tolerance", "1e-2"])
    main(["extrude", "--input", str(plate_dxf), "--layer", "OUTLINE", "--depth", "8",
          "--output", str(fine), "--tolerance", "1e-5"])
    assert len(trimesh.load(str(fine)).faces) > len(trimesh.load(str(coarse)).faces)


def test_multiview_takes_the_exact_path_for_a_rectangular_front(plate_dxf, front_dxf, tmp_path, capsys):
    out = tmp_path / "part.step"
    assert main(["multiview", "--top", str(plate_dxf), "--top-layer", "OUTLINE",
                 "--front", str(front_dxf), "--output", str(out)]) == 0

    captured = capsys.readouterr().out
    assert "exact-prismatic" in captured
    assert validate_step(out.read_text()) == []
    assert summarize_step(out.read_text())["CYLINDRICAL_SURFACE"] == 2


def test_inspect_reports_exact_curve_types(plate_dxf, capsys):
    assert main(["inspect", "--input", str(plate_dxf), "--layer", "OUTLINE", "-v"]) == 0
    output = capsys.readouterr().out
    assert "closed-form" in output  # area came from the exact integral
    assert "arc r=6.35" in output  # the hole is reported as an arc, with its radius
    assert "holes=2" in output


def test_inspect_reports_when_nothing_closes(tmp_path, capsys):
    doc = ezdxf.new()
    doc.modelspace().add_line((0, 0), (10, 0))
    path = tmp_path / "open.dxf"
    doc.saveas(path)

    assert main(["inspect", "--input", str(path)]) == 1
    assert "did not close" in capsys.readouterr().out


def test_unsupported_output_format_is_rejected(plate_dxf, tmp_path, capsys):
    assert main(["extrude", "--input", str(plate_dxf), "--layer", "OUTLINE",
                 "--depth", "8", "--output", str(tmp_path / "x.xyz")]) == 1
    assert "unsupported output" in capsys.readouterr().err


def test_missing_file_is_reported(tmp_path, capsys):
    assert main(["inspect", "--input", str(tmp_path / "nope.dxf")]) == 1
    assert "error:" in capsys.readouterr().err


def test_stepped_reconstruction_refuses_to_pretend_to_be_exact(tmp_path, capsys):
    """A stepped approximation must not be written out as exact CAD."""
    doc = ezdxf.new()
    doc.modelspace().add_lwpolyline([(0, 0), (100, 0), (100, 60), (0, 60)], close=True)
    top = tmp_path / "top2.dxf"
    doc.saveas(top)

    doc2 = ezdxf.new()
    doc2.modelspace().add_lwpolyline(
        [(0, 0), (100, 0), (100, 5), (70, 5), (70, 12), (30, 12), (30, 5), (0, 5)], close=True
    )
    front = tmp_path / "stepped.dxf"
    doc2.saveas(front)

    assert main(["multiview", "--top", str(top), "--front", str(front),
                 "--output", str(tmp_path / "out.step")]) == 1
    assert "cannot be written as STEP" in capsys.readouterr().err


def test_stepped_reconstruction_writes_a_mesh(tmp_path):
    pytest.importorskip("trimesh")
    doc = ezdxf.new()
    doc.modelspace().add_lwpolyline([(0, 0), (100, 0), (100, 60), (0, 60)], close=True)
    top = tmp_path / "top3.dxf"
    doc.saveas(top)

    doc2 = ezdxf.new()
    doc2.modelspace().add_lwpolyline(
        [(0, 0), (100, 0), (100, 5), (70, 5), (70, 12), (30, 12), (30, 5), (0, 5)], close=True
    )
    front = tmp_path / "stepped2.dxf"
    doc2.saveas(front)

    out = tmp_path / "stepped.stl"
    assert main(["multiview", "--top", str(top), "--front", str(front), "--output", str(out)]) == 0
    assert out.exists()
