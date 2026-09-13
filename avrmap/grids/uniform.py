"""The uniform-resolution 2.5D grid: one cell size everywhere.

This is the baseline the adaptive grid is measured against. Three variants are
used in the benchmark (see the technical plan): ``uniform_fine`` at the
adaptive grid's finest cell size, ``uniform_matched`` at a size chosen so its
cell count is close to the adaptive grid's, and ``uniform_coarse`` at the
coarsest zone size. All three are just this one function called with a
different ``cell_m``.
"""

from __future__ import annotations

import numpy as np

from ..config import GridConfig
from ..preprocess import PreprocessedPoints
from .common import CellTable, aggregate_cells


def build_uniform_map_with_point_cells(
    points: PreprocessedPoints,
    cell_m: float,
    min_points_per_cell: int = 1,
    elevation_stat: str = "p95",
) -> tuple[CellTable, np.ndarray]:
    """Like :func:`build_uniform_map`, also returning the point-to-cell map.

    Zone id is always 0: a uniform grid has exactly one zone by definition,
    so every point in ``points`` is a candidate for some cell (there is no
    "uncovered" concept for a uniform grid, unlike the adaptive grid's zone
    ladder). A point still maps to -1 if its cell was dropped by
    ``min_points_per_cell``.
    """
    return aggregate_cells(
        x=points.x,
        y=points.y,
        z=points.z,
        sem=points.sem,
        cell_size=cell_m,
        zone_id=0,
        min_points_per_cell=min_points_per_cell,
        elevation_stat=elevation_stat,
        return_point_cells=True,
    )


def build_uniform_map(
    points: PreprocessedPoints,
    cell_m: float,
    min_points_per_cell: int = 1,
    elevation_stat: str = "p95",
) -> CellTable:
    """Grid already-preprocessed points at one uniform cell size."""
    table, _ = build_uniform_map_with_point_cells(
        points, cell_m, min_points_per_cell, elevation_stat
    )
    return table


def build_uniform_map_from_config(
    points: PreprocessedPoints, grid_cfg: GridConfig
) -> CellTable:
    """Convenience wrapper using ``grid.uniform_cell_m`` from the config."""
    return build_uniform_map(
        points,
        cell_m=grid_cfg.uniform_cell_m,
        min_points_per_cell=grid_cfg.min_points_per_cell,
        elevation_stat=grid_cfg.elevation_stat,
    )


def find_matching_uniform_cell_size(
    points: PreprocessedPoints,
    target_cells: int,
    min_points_per_cell: int = 1,
    elevation_stat: str = "p95",
    lo_m: float = 0.01,
    hi_m: float = 5.0,
    iterations: int = 12,
) -> tuple[float, int]:
    """Search for a uniform cell size whose occupied-cell count is close to
    ``target_cells``: the ``uniform_matched`` baseline from the benchmark plan.

    For a fixed point cloud, occupied-cell count is a non-increasing function
    of cell size (a bigger cell can only merge occupied cells together, never
    split one), which is exactly what makes bisection a valid way to search
    for a target count. The match found is approximate, and both the size and
    the cell count it actually produced are returned rather than assuming the
    search landed exactly on target — a caller reporting `uniform_matched`
    results should report the achieved cell count alongside it.

    Runs up to ``iterations + 2`` full grid builds, so this is meaningfully
    more expensive than building a single uniform grid; it is meant for
    benchmark runs, not interactive use.
    """
    if target_cells <= 0:
        raise ValueError("target_cells must be positive")
    if lo_m <= 0 or hi_m <= lo_m:
        raise ValueError("require 0 < lo_m < hi_m")

    def cells_at(size: float) -> int:
        return len(build_uniform_map(points, size, min_points_per_cell, elevation_stat))

    best_size, best_cells = lo_m, cells_at(lo_m)
    hi_cells = cells_at(hi_m)
    if abs(hi_cells - target_cells) < abs(best_cells - target_cells):
        best_size, best_cells = hi_m, hi_cells

    lo, hi = lo_m, hi_m
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        cells = cells_at(mid)
        if abs(cells - target_cells) < abs(best_cells - target_cells):
            best_size, best_cells = mid, cells
        if cells > target_cells:
            lo = mid  # too many cells: cells are too fine, go coarser
        else:
            hi = mid  # too few cells (or exact): stop overshooting coarser
    return best_size, best_cells
