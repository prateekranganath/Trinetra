"""The adaptive (foveated) 2.5D grid: fine cells near the ego, coarse cells far away.

Distance zones are purely radial (see :mod:`avrmap.geometry`: the map frame is
gravity-aligned and only translated to the ego position, never rotated), so
building the adaptive grid needs nothing beyond each point's already-computed
planar radius. Each zone is gridded independently at its own cell size through
the same :func:`~avrmap.grids.common.aggregate_cells` kernel the uniform grid
uses, and the per-zone tables are concatenated. Cells from different zones
never collide, because :func:`avrmap.config.load_config` requires zones to be
contiguous and half-open (``[r_min, r_max)``), so every point in range belongs
to exactly one zone.
"""

from __future__ import annotations

import numpy as np

from ..config import GridConfig, ZoneConfig
from ..preprocess import PreprocessedPoints
from .common import CellTable, aggregate_cells


def build_adaptive_map_with_point_cells(
    points: PreprocessedPoints,
    zones: tuple[ZoneConfig, ...],
    min_points_per_cell: int = 1,
    elevation_stat: str = "p95",
) -> tuple[CellTable, np.ndarray]:
    """Like :func:`build_adaptive_map`, also returning the point-to-cell map.

    ``point_row`` has one entry per point in ``points`` (every zone's
    original indices are scattered back into this single array), mapping to a
    row in the concatenated table, or -1 if the point either fell outside
    every zone or landed in a cell dropped by ``min_points_per_cell`` — both
    are folded into the same -1, since from the standpoint of "does this
    point have a cell to be scored against" they are the same situation. Use
    :func:`uncovered_point_count` to separate the two if that distinction
    matters.
    """
    n = points.x.shape[0]
    point_row = np.full(n, -1, dtype=np.int64)
    tables: list[CellTable] = []
    row_offset = 0
    for zone_id, zone in enumerate(zones):
        idx = np.flatnonzero(zone.contains(points.radius))
        if idx.size == 0:
            continue
        table, sub_row = aggregate_cells(
            x=points.x[idx],
            y=points.y[idx],
            z=points.z[idx],
            sem=points.sem[idx],
            cell_size=zone.cell_m,
            zone_id=zone_id,
            min_points_per_cell=min_points_per_cell,
            elevation_stat=elevation_stat,
            return_point_cells=True,
        )
        valid = sub_row >= 0
        point_row[idx[valid]] = sub_row[valid] + row_offset
        tables.append(table)
        row_offset += len(table)
    return CellTable.concat(tables), point_row


def build_adaptive_map(
    points: PreprocessedPoints,
    zones: tuple[ZoneConfig, ...],
    min_points_per_cell: int = 1,
    elevation_stat: str = "p95",
) -> CellTable:
    """Grid already-preprocessed points on a per-zone cell size ladder.

    ``zone_id`` on the output is each zone's position in ``zones``, so it can
    be traced back to the config that produced it (for the zone-size or
    distance-band views in the dashboard, and for the per-zone benchmark
    breakdown).

    A point whose radius falls outside every zone (for example beyond the
    outermost zone's ``r_max``, if that is configured smaller than
    ``preprocess.max_range_m``) is silently excluded from the result, exactly
    as a uniform grid excludes points beyond its own range crop. This is a
    valid configuration, not an error, but it means the adaptive and uniform
    cell counts are only comparable when the zones' combined reach matches the
    range the uniform grid was built over.
    """
    table, _ = build_adaptive_map_with_point_cells(
        points, zones, min_points_per_cell, elevation_stat
    )
    return table


def build_adaptive_map_from_config(
    points: PreprocessedPoints, grid_cfg: GridConfig
) -> CellTable:
    """Convenience wrapper using the zone ladder from the config."""
    return build_adaptive_map(
        points,
        zones=grid_cfg.zones,
        min_points_per_cell=grid_cfg.min_points_per_cell,
        elevation_stat=grid_cfg.elevation_stat,
    )


def uncovered_point_count(points: PreprocessedPoints, zones: tuple[ZoneConfig, ...]) -> int:
    """How many points fall outside every zone.

    Zero for the shipped default config, where the zone ladder's outer edge
    matches ``preprocess.max_range_m`` exactly. Exposed so the benchmark and
    dashboard can report this honestly instead of silently dropping points if
    someone narrows the zone ladder without narrowing the range crop to match.
    """
    if points.radius.size == 0 or not zones:
        return int(points.radius.size)
    covered = zones[0].contains(points.radius)
    for zone in zones[1:]:
        covered |= zone.contains(points.radius)
    return int((~covered).sum())
