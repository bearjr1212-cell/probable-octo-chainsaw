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


# --------------------------------------------------------------------------
# Intake QA at the command line
# --------------------------------------------------------------------------


def qa_plate(tmp_path, name, *, x1=100.0, radius=3.175, dim_text=None, insunits=4):
    doc = ezdxf.new(setup=True)
    doc.header["$INSUNITS"] = insunits
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (x1, 0), (x1, 60), (0, 60)], close=True)
    msp.add_circle((25, 30), radius)
    kwargs = dict(center=(25, 30), radius=radius, angle=45)
    if dim_text is not None:
        kwargs["text"] = dim_text
    msp.add_diameter_dim(**kwargs).render()
    path = tmp_path / name
    doc.saveas(path)
    return path


def test_check_passes_a_clean_drawing(tmp_path, capsys):
    path = qa_plate(tmp_path, "clean.dxf")
    assert main(["check", "--input", str(path)]) == 0
    out = capsys.readouterr().out
    assert "READY TO CUT" in out
    assert "mm" in out  # the units it decided on, and why


def test_check_exits_two_when_the_drawing_contradicts_itself(tmp_path, capsys):
    """A distinct exit code, so an intake script can tell "bad drawing"
    from "the tool fell over"."""
    path = qa_plate(tmp_path, "liar.dxf", radius=3.15, dim_text="12.70")
    assert main(["check", "--input", str(path)]) == 2
    assert "DIMENSION_MISMATCH" in capsys.readouterr().out


def test_check_emits_json_on_request(tmp_path, capsys):
    import json

    path = qa_plate(tmp_path, "json.dxf")
    assert main(["check", "--input", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cuttable"] is True
    assert payload["metrics"]["holes"] == 1


def test_check_can_skip_the_audit(tmp_path, capsys):
    path = qa_plate(tmp_path, "skip.dxf", radius=3.15, dim_text="12.70")
    assert main(["check", "--input", str(path), "--no-audit"]) == 0
    assert "DIMENSION_MISMATCH" not in capsys.readouterr().out


def test_check_reports_a_missing_file_rather_than_crashing(tmp_path, capsys):
    assert main(["check", "--input", str(tmp_path / "nope.dxf")]) == 2
    assert "NOT CUTTABLE" in capsys.readouterr().out


def test_diff_names_the_feature_that_changed(tmp_path, capsys):
    before = qa_plate(tmp_path, "revB.dxf", radius=3.175)
    after = qa_plate(tmp_path, "revC.dxf", radius=3.25)
    assert main(["diff", "--before", str(before), "--after", str(after)]) == 2
    out = capsys.readouterr().out
    assert "revB → revC" in out
    assert "Ø6.35 → Ø6.5" in out
    assert "changed" in out


def test_diff_of_identical_files_is_clean(tmp_path, capsys):
    before = qa_plate(tmp_path, "a.dxf")
    after = qa_plate(tmp_path, "b.dxf")
    assert main(["diff", "--before", str(before), "--after", str(after)]) == 0
    assert "identical" in capsys.readouterr().out


def test_diff_flags_two_files_claiming_the_same_revision(tmp_path, capsys):
    before = qa_plate(tmp_path, "customer.dxf", radius=3.175)
    after = qa_plate(tmp_path, "engineering.dxf", radius=3.25)
    assert main(["diff", "--before", str(before), "--after", str(after),
                 "--revision-before", "C", "--revision-after", "C"]) == 2
    assert "REVISION_UNDOCUMENTED" in capsys.readouterr().out


def test_diff_json_carries_both_the_changes_and_the_report(tmp_path, capsys):
    import json

    before = qa_plate(tmp_path, "j1.dxf", radius=3.175)
    after = qa_plate(tmp_path, "j2.dxf", radius=3.25)
    main(["diff", "--before", str(before), "--after", str(after), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["identical"] is False
    assert payload["counts"]["changed"] == 1
    assert payload["report"]["defects"]


def test_diff_reports_an_unreadable_file_as_a_failure(tmp_path, capsys):
    assert main(["diff", "--before", str(tmp_path / "a.dxf"),
                 "--after", str(tmp_path / "b.dxf")]) == 1


def test_certify_reports_a_vector_part_as_exact(tmp_path, capsys):
    path = qa_plate(tmp_path, "cert.dxf")
    assert main(["certify", "--input", str(path), "--depth", "8"]) == 0
    out = capsys.readouterr().out
    assert "[vector]" in out
    assert "exact throughout" in out
    assert "fingerprint" in out


def test_certify_json_carries_the_fingerprint_and_every_basis(tmp_path, capsys):
    import json

    path = qa_plate(tmp_path, "certjson.dxf")
    assert main(["certify", "--input", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["fingerprint"]) == 64
    assert payload["exact"] is True
    assert {m["basis"] for m in payload["measurements"]} <= {"exact", "bounded", "estimated"}


def test_quote_costs_a_part_on_a_named_process(tmp_path, capsys):
    path = qa_plate(tmp_path, "quote.dxf")
    assert main(["quote", "--input", str(path), "--thickness", "6"]) == 0
    out = capsys.readouterr().out
    assert "cut length" in out
    assert "pierces" in out
    assert "READY TO CUT" in out


def test_quote_exits_two_when_the_machine_cannot_make_the_part(tmp_path, capsys):
    """Ø5 through 6 mm plate is a laser hole and not a plasma one."""
    path = qa_plate(tmp_path, "plasma.dxf", radius=2.5)
    assert main(["quote", "--input", str(path), "--thickness", "6",
                 "--process", "plasma"]) == 2
    assert "HOLE_TOO_SMALL_FOR_THICKNESS" in capsys.readouterr().out


def test_quote_compare_ranks_producible_before_cheap(tmp_path, capsys):
    path = qa_plate(tmp_path, "compare.dxf")
    assert main(["quote", "--input", str(path), "--thickness", "6", "--compare"]) == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()][1:]
    assert "OK" in lines[0]
    assert any("blocker" in l for l in lines)


def test_check_can_also_judge_manufacturability(tmp_path, capsys):
    path = qa_plate(tmp_path, "mfg.dxf", radius=2.5)
    assert main(["check", "--input", str(path), "--no-audit",
                 "--process", "plasma", "--thickness", "6"]) == 2
    assert "HOLE_TOO_SMALL_FOR_THICKNESS" in capsys.readouterr().out


def test_check_without_a_process_says_nothing_about_manufacturability(tmp_path, capsys):
    path = qa_plate(tmp_path, "nomfg.dxf")
    assert main(["check", "--input", str(path), "--no-audit"]) == 0
    assert "HOLE_TOO_SMALL" not in capsys.readouterr().out
