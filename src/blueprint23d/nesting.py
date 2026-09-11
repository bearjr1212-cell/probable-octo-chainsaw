"""How many of these come off a sheet, and how much of the sheet is scrap.

On a production job the material is frequently the larger half of the
price, and the number that decides it is not on the drawing: it is how
many parts fit on a sheet. A quote built on cut time alone gets this
wrong in the expensive direction, because the sheet is bought whole
whether or not the parts use it.

Full nesting is a hard combinatorial problem and this module does not
pretend to solve it. What it does is bracket the answer honestly:

* a **grid estimate** -- an achievable lower bound. However many this
  says, that many really do fit, because the layout it counts is one you
  could actually run;
* an **area bound** -- an upper bound nothing can beat, since parts cannot
  overlap and each consumes at least its own area;
* the **gap between them**, which is the headroom a real nesting package
  has to work in, and which is large exactly when the part is concave.

The grid estimate is better than rows of bounding boxes, because bounding
boxes are a terrible model of a triangle. Rows are allowed to *interlock*:
the next row up may be shifted sideways and, optionally, turned end for
end, and the row pitch is then whatever the two profiles actually permit
rather than the height of the box. Two triangles tile a rectangle, and
this finds that -- the pitch collapses from the triangle's full height to
the gap, doubling the parts per sheet.

The profile model is deliberately conservative: each part is treated as
solid between the lowest and highest material in every column across it,
so a C-shape is packed as though its mouth were filled. That can only
*under*-count, never over-count, which keeps the lower bound a lower
bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .brep import Face2D, Loop2D

Point2 = Tuple[float, float]

#: Columns used to sample a part's top and bottom profile. More is more
#: accurate and the cost is linear; 96 resolves a millimetre on a 100 mm
#: part, which is finer than any nesting gap worth having.
DEFAULT_COLUMNS = 96

#: Sideways row shifts tried when looking for an interlock.
DEFAULT_SHIFTS = 48


# --------------------------------------------------------------------------
# The part's silhouette, column by column
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Silhouette:
    """A part reduced to what nesting cares about: its skyline and its floor.

    ``top[i]`` and ``bottom[i]`` are the highest and lowest material in
    column ``i``, relative to the part's own bounding box. Columns with no
    material at all are ``None`` in both.
    """

    width: float
    height: float
    columns: Tuple[Optional[float], ...]  # top
    floors: Tuple[Optional[float], ...]  # bottom

    @property
    def column_width(self) -> float:
        return self.width / len(self.columns)

    def flipped(self) -> "Silhouette":
        """The part turned end for end -- rotated 180°, not mirrored."""
        n = len(self.columns)
        top = tuple(
            (self.height - self.floors[n - 1 - i]) if self.floors[n - 1 - i] is not None else None
            for i in range(n)
        )
        bottom = tuple(
            (self.height - self.columns[n - 1 - i]) if self.columns[n - 1 - i] is not None else None
            for i in range(n)
        )
        return Silhouette(self.width, self.height, top, bottom)

    def turned(self) -> "Silhouette":
        """The part rotated 90°. Rebuilt from scratch: a skyline does not
        transpose."""
        raise NotImplementedError("use silhouette(face, turned=True)")


def _densify(points: Sequence[Point2], step: float) -> List[Point2]:
    """Insert points so no two consecutive samples are further apart than
    ``step`` -- otherwise a long edge contributes to no column but its two
    ends, and the skyline has holes in it."""
    out: List[Point2] = []
    n = len(points)
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        out.append(a)
        distance = math.dist(a, b)
        if distance > step:
            pieces = int(distance / step) + 1
            for k in range(1, pieces):
                t = k / pieces
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def silhouette(
    face: Face2D,
    columns: int = DEFAULT_COLUMNS,
    turned: bool = False,
    tolerance: float = 0.05,
) -> Silhouette:
    """Sample a part's top and bottom profile across its width.

    ``turned`` rotates the part 90° first, which is the other orientation
    a nester will try.
    """
    points = face.outer.tessellate(tolerance)
    if turned:
        points = [(-y, x) for x, y in points]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, y0 = min(xs), min(ys)
    width = max(xs) - x0
    height = max(ys) - y0
    if width <= 0.0 or height <= 0.0:
        return Silhouette(width, height, (None,) * columns, (None,) * columns)

    step = width / columns
    dense = _densify([(x - x0, y - y0) for x, y in points], step / 2.0)

    top: List[Optional[float]] = [None] * columns
    bottom: List[Optional[float]] = [None] * columns
    for x, y in dense:
        index = min(columns - 1, max(0, int(x / step)))
        if top[index] is None or y > top[index]:
            top[index] = y
        if bottom[index] is None or y < bottom[index]:
            bottom[index] = y

    return Silhouette(width, height, tuple(top), tuple(bottom))


def required_pitch(
    lower: Silhouette, upper: Silhouette, shift: int, gap: float
) -> float:
    """Vertical spacing two rows need, with the upper row shifted right.

    ``shift`` is in columns. Returns ``inf`` if the rows never overlap in
    any column, which means the shift has moved the upper part clear of
    the lower one entirely and the answer is not a row pitch at all.
    """
    n = len(lower.columns)
    worst = -math.inf
    overlapped = False
    for i in range(n):
        j = i - shift
        if j < 0 or j >= n:
            continue
        low = lower.columns[i]
        high = upper.floors[j]
        if low is None or high is None:
            continue
        overlapped = True
        worst = max(worst, low - high)
    if not overlapped:
        return math.inf
    return worst + gap


# --------------------------------------------------------------------------
# The estimate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """One achievable way of filling a sheet.

    Rows alternate between two orientations, and the two gaps between them
    are allowed to differ. That matters: a right triangle and its
    end-for-end twin form a rectangle with almost nothing between them,
    and then the *next* pair has to clear the whole height. Forcing one
    uniform pitch would throw away half the sheet on exactly the parts
    where nesting has the most to give.
    """

    per_sheet: int
    rows: int
    #: Gap from a row to the one above it, and from that one to the next.
    #: Equal for a plain grid; very unequal when parts pair up.
    pitch_up: float
    pitch_back: float
    shift: float  # sideways offset of alternate rows, in drawing units
    flipped: bool
    turned: bool
    first_row: int
    other_row: int

    @property
    def pitch(self) -> float:
        """Average rise per row -- what the sheet actually costs per row."""
        return (self.pitch_up + self.pitch_back) / 2.0

    def describe(self) -> str:
        how = []
        if self.turned:
            how.append("part turned 90°")
        if self.flipped:
            how.append("alternate rows turned end for end")
        if self.shift > 0.0:
            how.append(f"alternate rows shifted {self.shift:.3g}")
        detail = ", ".join(how) if how else "plain grid"
        pitch = (
            f"{self.pitch_up:.4g} pitch"
            if abs(self.pitch_up - self.pitch_back) < 1e-9
            else f"pitch {self.pitch_up:.4g} then {self.pitch_back:.4g}"
        )
        return f"{self.per_sheet} per sheet in {self.rows} rows at {pitch} ({detail})"


@dataclass(frozen=True)
class NestingEstimate:
    """What a sheet holds, bracketed between what is achievable and what is
    conceivable."""

    layout: Layout
    part_area: float
    sheet: Tuple[float, float]
    usable: Tuple[float, float]
    gap: float
    margin: float
    #: Upper bound on any nesting whatsoever: parts cannot overlap.
    area_bound: int
    #: Part area over the area of its own bounding box. Low means concave,
    #: which is exactly when a real nester beats a grid.
    compactness: float

    @property
    def per_sheet(self) -> int:
        return self.layout.per_sheet

    @property
    def utilisation(self) -> float:
        """Fraction of the whole sheet that leaves as parts."""
        total = self.sheet[0] * self.sheet[1]
        return self.per_sheet * self.part_area / total if total else 0.0

    @property
    def headroom(self) -> int:
        """How many more a perfect nester could conceivably place."""
        return max(0, self.area_bound - self.per_sheet)

    def sheets_for(self, quantity: int) -> int:
        if self.per_sheet <= 0:
            return 0
        return math.ceil(quantity / self.per_sheet)

    def material_per_part(self, sheet_cost: float) -> float:
        """Sheet price divided by what the sheet yields -- the honest number,
        because the offcut is bought too."""
        if self.per_sheet <= 0:
            return math.inf
        return sheet_cost / self.per_sheet

    def describe(self) -> str:
        lines = [
            f"sheet {self.sheet[0]:g} × {self.sheet[1]:g}"
            + (f", {self.margin:g} margin" if self.margin else "")
            + (f", {self.gap:g} between parts" if self.gap else ""),
            f"  {self.layout.describe()}",
            f"  utilisation  {self.utilisation * 100:5.1f} %",
            f"  area bound   {self.area_bound} -- no nesting can beat this",
        ]
        if self.headroom:
            lines.append(
                f"  headroom     {self.headroom} more, if a nester can use the "
                f"{(1.0 - self.compactness) * 100:.0f}% of the bounding box this part does not fill"
            )
        else:
            lines.append("  headroom     none; the grid already reaches the area bound")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_sheet": self.per_sheet,
            "area_bound": self.area_bound,
            "headroom": self.headroom,
            "utilisation": self.utilisation,
            "compactness": self.compactness,
            "part_area": self.part_area,
            "sheet": list(self.sheet),
            "usable": list(self.usable),
            "gap": self.gap,
            "margin": self.margin,
            "layout": {
                "rows": self.layout.rows,
                "pitch": self.layout.pitch,
                "pitch_up": self.layout.pitch_up,
                "pitch_back": self.layout.pitch_back,
                "shift": self.layout.shift,
                "flipped": self.layout.flipped,
                "turned": self.layout.turned,
            },
        }


def _row_positions(
    usable_height: float, part_height: float, pitch_up: float, pitch_back: float
) -> int:
    """How many rows fit, stepping alternately by the two pitches."""
    if part_height > usable_height + 1e-9 or min(pitch_up, pitch_back) <= 0.0:
        return 0
    rows = 1
    y = 0.0
    while True:
        y += pitch_up if rows % 2 == 1 else pitch_back
        if y + part_height > usable_height + 1e-9:
            return rows
        rows += 1


def _per_row(usable_width: float, part_width: float, gap: float, offset: float = 0.0) -> int:
    available = usable_width - offset
    if part_width > available:
        return 0
    return 1 + int((available - part_width + 1e-9) / (part_width + gap))


def _pair_pitch(
    lower: Silhouette,
    upper: Silhouette,
    shift_columns: int,
    span_columns: int,
    gap: float,
) -> float:
    """Pitch two rows need, counting every part in the row, not just one.

    A row is a periodic train of parts ``span_columns`` apart. Checking
    only the part directly at ``shift`` misses its neighbours: shift a row
    of discs by nearly a full pitch and the disc above sits almost on top
    of the *next* one along, which is how an estimate ends up claiming
    more area in parts than the sheet has. Only the neighbour on each side
    can reach, since the spacing is at least the part's own width.
    """
    worst = -math.inf
    found = False
    for k in (-1, 0, 1):
        pitch = required_pitch(lower, upper, shift_columns + k * span_columns, gap)
        if math.isfinite(pitch):
            worst = max(worst, pitch)
            found = True
    return worst if found else math.inf


def _best_layout(
    profile: Silhouette,
    usable: Tuple[float, float],
    gap: float,
    turned: bool,
    shifts: int,
) -> Layout:
    """The most parts this orientation can be made to yield."""
    width, height = profile.width, profile.height
    best = Layout(0, 0, math.inf, math.inf, 0.0, False, turned, 0, 0)

    candidates: List[Tuple[Silhouette, bool]] = [(profile, False), (profile.flipped(), True)]
    columns = len(profile.columns)
    column_width = profile.column_width

    for upper, flipped in candidates:
        # Rows alternate between this orientation at offset 0 and the
        # candidate at offset `shift`, so a row is two pitches from the
        # next one like it. Both pairings have to clear, and so does the
        # like-to-like pairing across the gap between them -- which is the
        # constraint that stops a pitch from collapsing to the gap and
        # stacking a hundred parts through each other.
        like = required_pitch(profile, profile, 0, gap)
        unlike = required_pitch(upper, upper, 0, gap)
        floor_pitch = max(
            (p / 2.0 for p in (like, unlike) if math.isfinite(p)), default=0.0
        )

        span_columns = max(1, int(round((width + gap) / column_width)))
        for step in range(0, shifts + 1):
            shift_columns = int(round(step * columns / max(1, shifts)))
            forward = _pair_pitch(profile, upper, shift_columns, span_columns, gap)
            backward = _pair_pitch(upper, profile, -shift_columns, span_columns, gap)
            if not math.isfinite(forward) and not math.isfinite(backward):
                continue
            # Going up into the alternate row costs `up`; coming back down
            # into the next like row costs `back`, and the pair together
            # has to clear both like-to-like constraints.
            up = forward if math.isfinite(forward) else 0.0
            back = backward if math.isfinite(backward) else 0.0
            back = max(back, 2.0 * floor_pitch - up)
            if up <= 0.0 or back <= 0.0:
                continue
            shift = shift_columns * column_width

            rows = _row_positions(usable[1], height, up, back)
            if rows <= 0:
                continue
            first = _per_row(usable[0], width, gap)
            other = _per_row(usable[0], width, gap, offset=shift) if shift else first
            total = math.ceil(rows / 2) * first + (rows // 2) * other
            if total > best.per_sheet:
                best = Layout(total, rows, up, back, shift, flipped, turned, first, other)

    return best


def estimate(
    face: Face2D,
    sheet: Tuple[float, float],
    gap: float = 5.0,
    margin: float = 10.0,
    allow_turning: bool = True,
    columns: int = DEFAULT_COLUMNS,
    shifts: int = DEFAULT_SHIFTS,
) -> NestingEstimate:
    """Estimate the yield of one sheet.

    ``gap`` is the clear space left between parts (kerf plus whatever the
    process needs), ``margin`` the unusable border of the sheet.
    """
    usable = (sheet[0] - 2.0 * margin, sheet[1] - 2.0 * margin)
    part_area = face.area()

    layouts = [_best_layout(silhouette(face, columns), usable, gap, False, shifts)]
    if allow_turning:
        layouts.append(
            _best_layout(silhouette(face, columns, turned=True), usable, gap, True, shifts)
        )
    layout = max(layouts, key=lambda item: item.per_sheet)

    usable_area = max(0.0, usable[0]) * max(0.0, usable[1])
    area_bound = int(usable_area // part_area) if part_area > 0 else 0

    x0, y0, x1, y1 = face.bounds()
    box = (x1 - x0) * (y1 - y0)
    compactness = part_area / box if box else 0.0

    return NestingEstimate(
        layout=layout,
        part_area=part_area,
        sheet=sheet,
        usable=usable,
        gap=gap,
        margin=margin,
        area_bound=area_bound,
        compactness=compactness,
    )


def estimate_for(
    face: Face2D,
    proc,
    gap: Optional[float] = None,
    margin: float = 10.0,
    **kwargs: Any,
) -> NestingEstimate:
    """Estimate against a :class:`~blueprint23d.processes.Process`'s sheet.

    The default gap is twice the kerf plus a millimetre: one kerf belongs
    to each part's own cut, and a millimetre of slag or slat clearance is
    the least any shop leaves.
    """
    if gap is None:
        gap = 2.0 * proc.kerf + 1.0
    return estimate(face, proc.sheet, gap=gap, margin=margin, **kwargs)
