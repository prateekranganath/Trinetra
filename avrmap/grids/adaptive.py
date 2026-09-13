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

from ..config import GridConfig, ZoneConfig
from ..preprocess import PreprocessedPoints
from .common import CellTable, aggregate_cells


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
    tables: list[CellTable] = []
    for zone_id, zone in enumerate(zones):
        mask = zone.contains(points.radius)
        if not mask.any():
            continue
        tables.append(
            aggregate_cells(
                x=points.x[mask],
                y=points.y[mask],
                z=points.z[mask],
                sem=points.sem[mask],
                cell_size=zone.cell_m,
                zone_id=zone_id,
                min_points_per_cell=min_points_per_cell,
                elevation_stat=elevation_stat,
            )
        )
    return CellTable.concat(tables)


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
