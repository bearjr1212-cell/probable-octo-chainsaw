"""DWG input, via an external converter invoked at arm's length.

DWG is the format most shops actually send, and it is closed. The two ways
in are LibreDWG and the ODA File Converter, and *how* they are used matters
as much as *that* they are used:

**LibreDWG is GPL-3.0.** Linking it -- as a C library or through Python
bindings -- makes the linking program a derivative work and propagates the
GPL to it. Running ``dwg2dxf`` as a **separate process** does not: the two
programs communicate through files at arm's length, which the FSF's own
guidance treats as separate works. So this module shells out and never
links, and blueprint23d keeps whatever licence its author chooses. The
practical cost is that LibreDWG becomes a runtime dependency the user
installs themselves, the way a tool might require ImageMagick.

The ODA File Converter is free-of-charge but proprietary and cannot be
redistributed; it is supported as an alternative backend for the same
reason and in the same way.

Conversion is lossy in ways worth knowing about, so it is *reported* rather
than assumed away. LibreDWG round trips geometry faithfully -- measured
here, arc radii survive bit-for-bit -- but drops layer names, which
silently breaks any layer-filtered workflow. That is emitted as a
:data:`~blueprint23d.diagnostics.Code.CONVERSION_METADATA_LOST` defect
instead of returning an empty result and letting the caller guess.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from ..brep import Face2D
from ..curves import Curve2D
from ..diagnostics import Code, Report, Severity

#: Conversion is bounded: a malicious or corrupt DWG must not hang a queue.
DEFAULT_TIMEOUT_SECONDS = 120

#: Refuse absurd inputs before handing them to a C program.
DEFAULT_MAX_BYTES = 512 * 1024 * 1024


@dataclass
class ConversionResult:
    dxf_path: Optional[Path]
    converter: Optional[str]
    report: Report = field(default_factory=Report)

    @property
    def ok(self) -> bool:
        return self.dxf_path is not None


class DwgConverter(ABC):
    """A program that can turn a DWG into a DXF."""

    name: str = "converter"
    licence: str = "unknown"

    @abstractmethod
    def executable(self) -> Optional[str]:
        """Path to the binary, or None when it is not installed."""

    @abstractmethod
    def command(self, source: Path, destination: Path) -> Sequence[str]:
        """Argument vector to run."""

    def available(self) -> bool:
        return self.executable() is not None

    def version(self) -> Optional[str]:
        binary = self.executable()
        if binary is None:
            return None
        try:
            done = subprocess.run(
                [binary, "--version"], capture_output=True, text=True, timeout=15
            )
            return (done.stdout or done.stderr).strip().splitlines()[0]
        except Exception:
            return None

    def convert(
        self,
        source: Path,
        destination: Path,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> Tuple[bool, str]:
        binary = self.executable()
        if binary is None:
            return False, f"{self.name} is not installed"
        try:
            done = subprocess.run(
                list(self.command(source, destination)),
                capture_output=True,
                text=True,
                timeout=timeout,
                # Run in a scratch directory: some converters write
                # alongside the input or into the working directory.
                cwd=str(destination.parent),
            )
        except subprocess.TimeoutExpired:
            return False, f"{self.name} timed out after {timeout}s"
        except OSError as exc:
            return False, f"{self.name} could not be run: {exc}"

        output = "\n".join(part for part in (done.stdout, done.stderr) if part).strip()
        if not destination.exists() or destination.stat().st_size == 0:
            return False, output or f"{self.name} produced no output"
        return True, output


class LibreDwgConverter(DwgConverter):
    """GNU LibreDWG's ``dwg2dxf``, run as a separate process.

    Invoked, never linked -- see the module docstring for why that
    distinction decides the licence of the calling program.
    """

    name = "LibreDWG"
    licence = "GPL-3.0 (invoked as a separate process, not linked)"

    def executable(self) -> Optional[str]:
        return shutil.which("dwg2dxf")

    def command(self, source: Path, destination: Path) -> Sequence[str]:
        return [self.executable() or "dwg2dxf", "-y", "-o", str(destination), str(source)]


class OdaConverter(DwgConverter):
    """Open Design Alliance File Converter.

    Free of charge, proprietary, not redistributable. Its command line is
    directory-oriented rather than file-oriented, so a whole scratch
    directory is handed over and the result collected afterwards.
    """

    name = "ODA File Converter"
    licence = "proprietary, free of charge, not redistributable"

    def executable(self) -> Optional[str]:
        for candidate in ("ODAFileConverter", "ODAFileConverter.exe"):
            found = shutil.which(candidate)
            if found:
                return found
        return None

    def command(self, source: Path, destination: Path) -> Sequence[str]:
        # input_dir output_dir version type recurse audit
        return [
            self.executable() or "ODAFileConverter",
            str(source.parent),
            str(destination.parent),
            "ACAD2018",
            "DXF",
            "0",
            "1",
            source.name,
        ]


#: Tried in order; the first installed one wins.
CONVERTERS: List[DwgConverter] = [LibreDwgConverter(), OdaConverter()]


def available_converters() -> List[DwgConverter]:
    return [c for c in CONVERTERS if c.available()]


def convert_to_dxf(
    path: Union[str, Path],
    workdir: Optional[Path] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> ConversionResult:
    """Convert a DWG to DXF with whichever backend is installed."""
    source = Path(path)
    report = Report(source=str(source))

    if not source.exists():
        report.add(Code.FILE_UNREADABLE, Severity.CRITICAL, f"{source} does not exist")
        return ConversionResult(None, None, report)

    size = source.stat().st_size
    if size > max_bytes:
        report.add(
            Code.FILE_TOO_LARGE,
            Severity.CRITICAL,
            f"file is {size / 1e6:.1f} MB, over the {max_bytes / 1e6:.0f} MB limit",
            bytes=float(size),
        )
        return ConversionResult(None, None, report)

    installed = available_converters()
    if not installed:
        report.add(
            Code.CONVERTER_UNAVAILABLE,
            Severity.CRITICAL,
            "no DWG converter found on PATH (looked for dwg2dxf, ODAFileConverter)",
        )
        return ConversionResult(None, None, report)

    owns_workdir = workdir is None
    workdir = Path(workdir or tempfile.mkdtemp(prefix="blueprint23d-dwg-"))
    workdir.mkdir(parents=True, exist_ok=True)
    destination = workdir / (source.stem + ".dxf")

    last_message = ""
    for converter in installed:
        ok, message = converter.convert(source, destination, timeout=timeout)
        if ok:
            if message:
                # Converter chatter is informational, not failure -- but it
                # is where "handle collision" style corruption shows up, so
                # it is preserved rather than swallowed.
                for line in _interesting_lines(message):
                    report.add(
                        Code.CONVERSION_WARNING,
                        Severity.INFO,
                        f"{converter.name}: {line}",
                    )
            return ConversionResult(destination, converter.name, report)
        last_message = message

    report.add(
        Code.CONVERSION_FAILED,
        Severity.CRITICAL,
        f"every available converter failed; last said: {last_message[:400]}",
    )
    if owns_workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    return ConversionResult(None, None, report)


def _interesting_lines(output: str, limit: int = 8) -> List[str]:
    """Keep converter output that signals a problem, drop the progress noise."""
    keep = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith(("reading ", "writing ")):
            continue
        if any(word in lowered for word in ("error", "warning", "invalid", "duplicate", "wrong", "corrupt")):
            keep.append(line)
        if len(keep) >= limit:
            break
    return keep


def load_faces(
    path: Union[str, Path],
    layer: Optional[str] = None,
    weld_tolerance: float = 1e-7,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    keep_dxf: Optional[Path] = None,
) -> Tuple[List[Face2D], List[List[Curve2D]], Report]:
    """Read a DWG into exact faces, reporting anything conversion cost.

    Also returns the converted intermediate when ``keep_dxf`` is given,
    which is worth having: if a customer disputes a result, the DXF the
    geometry was actually read from is the evidence.
    """
    from . import dxf_exact

    result = convert_to_dxf(path, timeout=timeout)
    report = result.report
    if not result.ok:
        return [], [], report

    dxf_path = result.dxf_path
    assert dxf_path is not None

    layers = _layers_present(dxf_path)
    lost_layers = layers <= {"", "0"}
    if lost_layers:
        report.add(
            Code.CONVERSION_METADATA_LOST,
            Severity.WARNING if layer is None else Severity.ERROR,
            f"{result.converter} did not preserve layer names; every entity came "
            f"back on layer {sorted(layers) or ['<none>']}",
        )

    effective_layer = layer
    if layer is not None and lost_layers:
        # Filtering on a name that no longer exists would silently return
        # nothing at all, which reads as "empty drawing" rather than "the
        # converter dropped your layers". Read everything instead and say so.
        effective_layer = None

    faces, open_chains = dxf_exact.load_faces(
        dxf_path, layer=effective_layer, weld_tolerance=weld_tolerance
    )

    if not faces:
        report.add(
            Code.NO_CLOSED_PROFILE,
            Severity.ERROR,
            "no closed profile was found after conversion",
        )
    if open_chains:
        report.add(
            Code.OPEN_CONTOUR,
            Severity.WARNING,
            f"{len(open_chains)} chain(s) did not close",
        )

    report.metrics["converter"] = result.converter
    report.metrics["converted_dxf"] = str(dxf_path)

    if keep_dxf is not None:
        keep_dxf = Path(keep_dxf)
        keep_dxf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dxf_path, keep_dxf)
        report.metrics["converted_dxf"] = str(keep_dxf)

    return faces, open_chains, report


def _layers_present(dxf_path: Path) -> set:
    import ezdxf

    try:
        doc = ezdxf.readfile(str(dxf_path))
    except Exception:
        return set()
    return {entity.dxf.layer for entity in doc.modelspace() if entity.dxf.hasattr("layer")}


def load_face(
    path: Union[str, Path],
    layer: Optional[str] = None,
    weld_tolerance: float = 1e-7,
) -> Face2D:
    """The largest closed profile in a DWG."""
    faces, _, report = load_faces(path, layer=layer, weld_tolerance=weld_tolerance)
    if not faces:
        blocking = report.at_least(Severity.ERROR)
        detail = blocking[0].message if blocking else "no closed profile found"
        raise ValueError(f"{path}: {detail}")
    return max(faces, key=lambda f: f.area())


def describe_backends() -> str:
    """What converters are installed, for diagnostics and support."""
    lines = []
    for converter in CONVERTERS:
        if converter.available():
            lines.append(f"  {converter.name}: {converter.version() or 'installed'}  [{converter.licence}]")
        else:
            lines.append(f"  {converter.name}: not installed")
    return "\n".join(lines)
