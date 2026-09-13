"""Cost and quality metrics for comparing grid-building strategies.

Every function here produces a value with a clear provenance: **measured**
directly, **derived** from other measured values, or explicitly unavailable.
The benchmark script (`scripts/run_benchmark.py`) is the only place these get
assembled into a report, and it carries the label for every column forward
into `results/benchmark.csv` rather than letting a reader guess which numbers
were actually run and which were computed after the fact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .grids.common import CellTable
from .preprocess import PreprocessedPoints


def median_build_time_s(build_fn: Callable[[], object], repeats: int = 3) -> float:
    """Median wall-clock time of calling ``build_fn()``, in seconds.

    A median over repeats damps one-off scheduler noise without hiding a
    genuinely slow build the way a minimum would. ``build_fn`` should perform
    only the grid construction — loading and preprocessing are timed
    separately elsewhere, so "build time" means build time and nothing else.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        build_fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def dense_equivalent_cells(extent_m: float, cell_size_m: float) -> int:
    """Cells a dense array would need to cover a ``2*extent_m`` square.

    This is a **derived** figure: no dense array is ever built anywhere in
    this project. It exists to make a sparse `CellTable`'s cell count and
    memory legible by comparison — "how much would the naive approach cost at
    this resolution" — not as a measurement of anything that ran.
    """
    if extent_m <= 0 or cell_size_m <= 0:
        raise ValueError("extent_m and cell_size_m must be positive")
    side_cells = int(np.ceil(2 * extent_m / cell_size_m))
    return side_cells * side_cells


@dataclass(frozen=True)
class BandQuality:
    """Elevation and semantic quality within one distance band.

    Both figures are measured directly against the raw points a grid was
    built from — no synthetic ground truth is used, since the question is how
    much the grid distorts what the sensor actually reported.

    Attributes:
        r_min, r_max: Band boundaries in metres, half-open ``[r_min, r_max)``.
        n_points: Points in this band whose cell survived to be scored.
        n_unmatched: Points in this band with no surviving cell — either
            outside every zone (adaptive only) or in a cell dropped by
            ``min_points_per_cell``. Rare under the shipped default config.
        elevation_rmse_m: Root-mean-square ``|point.z - cell.z_ref|`` over the
            matched points in this band. ``NaN`` if ``n_points == 0``.
        semantic_agreement: Fraction of matched points whose own class equals
            their cell's majority class. ``NaN`` if ``n_points == 0``.
    """

    r_min: float
    r_max: float
    n_points: int
    n_unmatched: int
    elevation_rmse_m: float
    semantic_agreement: float

    @property
    def label(self) -> str:
        return f"{self.r_min:g}-{self.r_max:g}m"


def quality_by_band(
    points: PreprocessedPoints,
    table: CellTable,
    point_row: np.ndarray,
    bands_m: tuple[float, ...],
) -> list[BandQuality]:
    """Elevation RMSE and semantic agreement, one :class:`BandQuality` per band.

    ``point_row[i]`` must be the row in ``table`` that point ``i`` of
    ``points`` was aggregated into, or -1 — exactly what
    ``build_uniform_map_with_point_cells`` / ``build_adaptive_map_with_point_cells``
    return alongside the table itself. ``bands_m`` gives the band edges, e.g.
    ``(0.0, 10.0, 30.0, 60.0, 100.0)`` for four bands; the same tuple must be
    used for every method being compared so the results line up.
    """
    if point_row.shape[0] != points.x.shape[0]:
        raise ValueError(
            f"point_row has {point_row.shape[0]} entries but points has "
            f"{points.x.shape[0]}"
        )
    if len(bands_m) < 2:
        raise ValueError("bands_m needs at least two edges to form one band")

    matched = point_row >= 0
    elev_error = np.full(points.x.shape[0], np.nan, dtype=np.float64)
    sem_match = np.zeros(points.x.shape[0], dtype=bool)
    if matched.any():
        rows = point_row[matched]
        elev_error[matched] = np.abs(
            points.z[matched].astype(np.float64) - table.z_ref[rows].astype(np.float64)
        )
        sem_match[matched] = points.sem[matched] == table.sem_class[rows]

    results = []
    for r_min, r_max in zip(bands_m, bands_m[1:]):
        in_band = (points.radius >= r_min) & (points.radius < r_max)
        band_matched = in_band & matched
        n_points = int(band_matched.sum())
        n_unmatched = int((in_band & ~matched).sum())
        if n_points:
            rmse = float(np.sqrt(np.mean(elev_error[band_matched] ** 2)))
            agreement = float(np.mean(sem_match[band_matched]))
        else:
            rmse = float("nan")
            agreement = float("nan")
        results.append(
            BandQuality(
                r_min=r_min,
                r_max=r_max,
                n_points=n_points,
                n_unmatched=n_unmatched,
                elevation_rmse_m=rmse,
                semantic_agreement=agreement,
            )
        )
    return results
