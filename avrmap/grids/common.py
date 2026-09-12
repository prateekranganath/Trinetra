"""The shared sparse cell representation and its aggregation kernel.

A dense 2D array cannot hold mixed resolutions, and even a single-resolution
dense array over a 100 m radius at 0.1 m cells would be 4 million cells of
which only a few percent are occupied. Both the uniform and the adaptive grid
therefore share one representation, :class:`CellTable`: parallel NumPy arrays
with **one row per occupied cell**. Using identical fields and dtypes for both
grid types is what makes the memory and cell-count comparison in the benchmark
honest — the row count is the only thing that differs.

:func:`aggregate_cells` is the one place points turn into cells. It is called
once by the uniform grid (a single zone covering the whole range) and once per
zone by the adaptive grid, so a bug fixed here is fixed in both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VALID_ELEVATION_STATS = ("p95", "max", "mean", "min")

# CellTable fields, as (name, dtype) pairs. Used to build empty tables and to
# keep concat() from silently dropping a field someone adds later.
_FIELDS: tuple[tuple[str, str], ...] = (
    ("cx", "float32"),
    ("cy", "float32"),
    ("size", "float32"),
    ("zone", "uint8"),
    ("z_ref", "float32"),
    ("z_min", "float32"),
    ("z_max", "float32"),
    ("n_points", "uint32"),
    ("sem_class", "uint8"),
    ("sem_conf", "float32"),
)


@dataclass(frozen=True)
class CellTable:
    """One row per occupied grid cell.

    Attributes:
        cx, cy: Cell centre, metres from the ego vehicle, map-frame axes.
        size: Cell edge length in metres.
        zone: Distance-zone index the cell belongs to (0 for a uniform grid).
        z_ref: Representative elevation, per the configured statistic.
        z_min, z_max: Height extent of points aggregated into the cell.
        n_points: Point count aggregated into the cell.
        sem_class: Majority-vote semantic class id.
        sem_conf: Majority fraction, in [0, 1].
    """

    cx: np.ndarray
    cy: np.ndarray
    size: np.ndarray
    zone: np.ndarray
    z_ref: np.ndarray
    z_min: np.ndarray
    z_max: np.ndarray
    n_points: np.ndarray
    sem_class: np.ndarray
    sem_conf: np.ndarray

    def __post_init__(self):
        n = len(self.cx)
        for name, _ in _FIELDS:
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"CellTable field lengths disagree: cx has {n}, {name} has "
                    f"{len(getattr(self, name))}"
                )

    def __len__(self) -> int:
        return int(self.cx.shape[0])

    @property
    def n_cells(self) -> int:
        return len(self)

    @property
    def nbytes(self) -> int:
        """Total bytes held by every field array. This is the measured memory
        figure used in the benchmark, as opposed to a dense-array estimate."""
        return sum(getattr(self, name).nbytes for name, _ in _FIELDS)

    @classmethod
    def empty(cls) -> "CellTable":
        return cls(**{name: np.zeros(0, dtype=dtype) for name, dtype in _FIELDS})

    @classmethod
    def concat(cls, tables: list["CellTable"]) -> "CellTable":
        """Stack cell tables from multiple zones (or frames) into one.

        Used by the adaptive grid to combine its per-zone tables. Cells from
        different zones never collide because zones occupy disjoint radii, so
        no re-aggregation across tables is needed here.
        """
        tables = [t for t in tables if len(t) > 0]
        if not tables:
            return cls.empty()
        return cls(
            **{
                name: np.concatenate([getattr(t, name) for t in tables]).astype(
                    dtype, copy=False
                )
                for name, dtype in _FIELDS
            }
        )


def aggregate_cells(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sem: np.ndarray,
    cell_size: float,
    zone_id: int = 0,
    min_points_per_cell: int = 1,
    elevation_stat: str = "p95",
) -> CellTable:
    """Bin points into square cells of ``cell_size`` and aggregate each cell.

    This is the single aggregation kernel shared by the uniform and adaptive
    grids. Called once per zone: the uniform grid uses one call covering its
    whole range with ``zone_id=0``; the adaptive grid calls it once per zone
    with that zone's own cell size and id, then concatenates the results.

    Elevation aggregation. Points are grouped by cell via one stable sort, then
    ``np.add.reduceat`` / segment boundaries give mean, min and max in a single
    pass. The 95th-percentile default is a nearest-rank statistic computed from
    a second sort ordered by ``(cell, z)``, which is far more resistant to a
    single spurious high return than the max.

    Semantic majority vote. Points are grouped by the combined key
    ``cell * 256 + class`` (both int64, safely apart), which yields per-cell,
    per-class counts from one more ``np.unique``. A monotone score,
    ``count * 256 - class_id``, is then reduced per cell: since 256 exceeds the
    largest possible class id, a strictly higher count always wins, and among
    equal counts the lower class id wins, deterministically. No Python loop
    touches the points.

    Args:
        x, y, z: (N,) float arrays in the map frame.
        sem: (N,) integer semantic class per point.
        cell_size: Edge length of a cell, in metres. Must be positive.
        zone_id: Recorded on every output cell, for provenance in a
            multi-zone (adaptive) grid.
        min_points_per_cell: Cells with fewer points than this are dropped as
            unreliable.
        elevation_stat: One of ``"p95"``, ``"max"``, ``"mean"``, ``"min"``,
            selecting what ``z_ref`` reports.

    Returns:
        A :class:`CellTable` with one row per surviving occupied cell, sorted
        by cell key (i.e. by ix then iy).
    """
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive, got {cell_size}")
    if elevation_stat not in VALID_ELEVATION_STATS:
        raise ValueError(
            f"elevation_stat must be one of {VALID_ELEVATION_STATS}, got "
            f"'{elevation_stat}'"
        )
    n = x.shape[0]
    if n == 0:
        return CellTable.empty()

    ix = np.floor(x / cell_size).astype(np.int64)
    iy = np.floor(y / cell_size).astype(np.int64)

    # Key cells by their position within the actual bounding box of visited
    # cells, not a fixed-size grid. This needs no assumption about how far
    # points can range and can never collide: it is exactly row-major indexing
    # over the occupied footprint.
    lo_x, lo_y = int(ix.min()), int(iy.min())
    span_y = int(iy.max()) - lo_y + 1
    key = (ix - lo_x) * span_y + (iy - lo_y)

    # --- elevation: sort by (cell, z) so both percentile and min/max/mean
    # fall out of one pass ------------------------------------------------
    z64 = z.astype(np.float64, copy=False)  # avoid float32 cancellation in sums
    elev_order = np.lexsort((z64, key))
    skey = key[elev_order]
    sz = z64[elev_order]
    cells, start_idx, counts = np.unique(skey, return_index=True, return_counts=True)

    z_min = sz[start_idx].astype(np.float32)
    z_max = sz[start_idx + counts - 1].astype(np.float32)
    z_sum = np.add.reduceat(sz, start_idx)
    z_mean = (z_sum / counts).astype(np.float32)
    if elevation_stat == "max":
        z_ref = z_max
    elif elevation_stat == "min":
        z_ref = z_min
    elif elevation_stat == "mean":
        z_ref = z_mean
    else:  # p95, nearest-rank: the smallest value at or above the 95th
        rank = np.clip(np.ceil(0.95 * counts).astype(np.int64) - 1, 0, counts - 1)
        z_ref = sz[start_idx + rank].astype(np.float32)

    # --- semantic majority vote -------------------------------------------
    combo = key.astype(np.int64) * 256 + sem.astype(np.int64)
    sem_order = np.argsort(combo, kind="stable")
    uniq_combo, combo_counts = np.unique(combo[sem_order], return_counts=True)
    combo_cells = uniq_combo // 256
    combo_classes = (uniq_combo % 256).astype(np.uint8)
    # Strictly monotone in (count, -class_id): a higher count always wins, and
    # among equal counts the lower class id wins. See the docstring.
    score = combo_counts.astype(np.int64) * 256 - combo_classes.astype(np.int64)
    _, group_start = np.unique(combo_cells, return_index=True)
    group_sizes = np.diff(np.append(group_start, len(combo_cells)))
    best_score = np.maximum.reduceat(score, group_start)
    is_best = score == np.repeat(best_score, group_sizes)
    # score is injective within a group by construction, so exactly one row
    # per group matches; this both selects and orders results by cell id.
    best_idx = np.flatnonzero(is_best)
    sem_class = combo_classes[best_idx]
    sem_top_count = combo_counts[best_idx]

    if not np.array_equal(combo_cells[best_idx], cells):
        raise AssertionError(
            "internal error: semantic and elevation aggregation disagree on "
            "which cells are occupied"
        )

    n_points = counts.astype(np.uint32)
    sem_conf = (sem_top_count / counts).astype(np.float32)

    # --- cell centres, decoded from the same key used to group points ------
    ix_of_cell = cells // span_y + lo_x
    iy_of_cell = cells % span_y + lo_y
    cx = ((ix_of_cell.astype(np.float64) + 0.5) * cell_size).astype(np.float32)
    cy = ((iy_of_cell.astype(np.float64) + 0.5) * cell_size).astype(np.float32)

    keep = n_points >= min_points_per_cell
    n_kept = int(keep.sum())
    return CellTable(
        cx=cx[keep],
        cy=cy[keep],
        size=np.full(n_kept, cell_size, dtype=np.float32),
        zone=np.full(n_kept, zone_id, dtype=np.uint8),
        z_ref=z_ref[keep],
        z_min=z_min[keep],
        z_max=z_max[keep],
        n_points=n_points[keep],
        sem_class=sem_class[keep],
        sem_conf=sem_conf[keep],
    )
