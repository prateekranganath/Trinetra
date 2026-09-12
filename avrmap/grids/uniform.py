"""The uniform-resolution 2.5D grid: one cell size everywhere.

This is the baseline the adaptive grid is measured against. Three variants are
used in the benchmark (see the technical plan): ``uniform_fine`` at the
adaptive grid's finest cell size, ``uniform_matched`` at a size chosen so its
cell count is close to the adaptive grid's, and ``uniform_coarse`` at the
coarsest zone size. All three are just this one function called with a
different ``cell_m``.
"""

from __future__ import annotations

from ..config import GridConfig
from ..preprocess import PreprocessedPoints
from .common import CellTable, aggregate_cells


def build_uniform_map(
    points: PreprocessedPoints,
    cell_m: float,
    min_points_per_cell: int = 1,
    elevation_stat: str = "p95",
) -> CellTable:
    """Grid already-preprocessed points at one uniform cell size.

    Zone id is always 0: a uniform grid has exactly one zone by definition.
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
    )


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
