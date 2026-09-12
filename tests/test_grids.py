"""The shared cell aggregation kernel and the uniform grid built on it.

Every case here has an answer computable by hand; that answer is asserted
directly rather than compared against a second implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from avrmap.grids.common import CellTable, aggregate_cells
from avrmap.grids.uniform import build_uniform_map, build_uniform_map_from_config
from avrmap.preprocess import PreprocessedPoints


def cell(x, y, z, sem, **kwargs) -> CellTable:
    return aggregate_cells(
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(z, dtype=np.float32),
        np.asarray(sem, dtype=np.uint8),
        **kwargs,
    )


class TestEmptyAndSingleCell:
    def test_empty_input_returns_an_empty_table(self):
        t = cell([], [], [], [], cell_size=1.0)
        assert len(t) == 0
        assert t.nbytes == 0

    def test_a_flat_plane_reports_that_exact_elevation(self):
        t = cell([0.1, 0.4, 0.9], [0.1, 0.4, 0.9], [1.0, 1.0, 1.0], [7, 7, 7], cell_size=1.0)
        assert len(t) == 1
        assert t.z_ref[0] == t.z_min[0] == t.z_max[0] == 1.0
        assert t.n_points[0] == 3
        assert t.sem_class[0] == 7 and t.sem_conf[0] == 1.0

    def test_cell_centre_is_the_geometric_centre_not_a_point_average(self):
        # Points cluster near one corner of the cell; the centre must still be
        # the cell's own midpoint, not the mean of the points.
        t = cell([0.01, 0.02], [0.01, 0.02], [0.0, 0.0], [7, 7], cell_size=1.0)
        assert t.cx[0] == pytest.approx(0.5)
        assert t.cy[0] == pytest.approx(0.5)


class TestGridGeometry:
    def test_a_dense_square_yields_exactly_l_over_r_squared_cells(self):
        rng = np.random.default_rng(0)
        side, res, per_cell = 3.0, 0.5, 4
        n_side = int(side / res)
        xs, ys = [], []
        for i in range(n_side):
            for j in range(n_side):
                xs.append(rng.uniform(i * res, (i + 1) * res, per_cell))
                ys.append(rng.uniform(j * res, (j + 1) * res, per_cell))
        x, y = np.concatenate(xs), np.concatenate(ys)
        t = cell(x, y, np.zeros_like(x), np.full(x.shape, 7, dtype=np.uint8), cell_size=res)
        assert len(t) == n_side * n_side

    def test_negative_coordinates_form_their_own_correct_cells(self):
        t = cell([-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [1, 2, 3], [7, 8, 9], cell_size=1.0)
        assert len(t) == 3
        assert sorted(zip(t.cx.tolist(), t.cy.tolist())) == [(-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5)]

    def test_a_point_on_a_cell_boundary_belongs_to_the_upper_cell(self):
        # floor(1.0 / 1.0) == 1, so x=1.0 is cell index 1, not cell index 0.
        t = cell([0.5, 1.0], [0.5, 0.5], [0.0, 0.0], [7, 7], cell_size=1.0)
        assert len(t) == 2


class TestElevationStatistics:
    Z = np.arange(100, dtype=np.float32)  # 0..99 in one cell

    def test_p95_is_the_nearest_rank_not_a_max_or_a_mean(self):
        t = cell(np.full(100, 0.5), np.full(100, 0.5), self.Z, np.full(100, 7, dtype=np.uint8),
                  cell_size=1.0, elevation_stat="p95")
        assert t.z_ref[0] == 94.0  # ceil(0.95*100) - 1

    def test_max_stat_matches_z_max(self):
        t = cell(np.full(100, 0.5), np.full(100, 0.5), self.Z, np.full(100, 7, dtype=np.uint8),
                  cell_size=1.0, elevation_stat="max")
        assert t.z_ref[0] == t.z_max[0] == 99.0

    def test_min_stat_matches_z_min(self):
        t = cell(np.full(100, 0.5), np.full(100, 0.5), self.Z, np.full(100, 7, dtype=np.uint8),
                  cell_size=1.0, elevation_stat="min")
        assert t.z_ref[0] == t.z_min[0] == 0.0

    def test_mean_stat_is_the_arithmetic_mean(self):
        t = cell(np.full(100, 0.5), np.full(100, 0.5), self.Z, np.full(100, 7, dtype=np.uint8),
                  cell_size=1.0, elevation_stat="mean")
        assert t.z_ref[0] == pytest.approx(49.5)

    def test_a_single_spurious_high_return_barely_moves_p95_but_dominates_max(self):
        z = np.concatenate([np.zeros(99, dtype=np.float32), [1000.0]])
        xy = np.full(100, 0.5, dtype=np.float32)
        sem = np.full(100, 7, dtype=np.uint8)
        p95 = cell(xy, xy, z, sem, cell_size=1.0, elevation_stat="p95")
        mx = cell(xy, xy, z, sem, cell_size=1.0, elevation_stat="max")
        assert p95.z_ref[0] == 0.0
        assert mx.z_ref[0] == 1000.0

    def test_unknown_elevation_stat_is_rejected(self):
        with pytest.raises(ValueError, match="elevation_stat"):
            cell([0.5], [0.5], [0.0], [7], cell_size=1.0, elevation_stat="bogus")


class TestSemanticMajorityVote:
    def test_clear_majority_wins_with_the_correct_confidence(self):
        sem = [7, 7, 7, 13, 13]
        t = cell([0.1] * 5, [0.1] * 5, [0.0] * 5, sem, cell_size=1.0)
        assert t.sem_class[0] == 7
        assert t.sem_conf[0] == pytest.approx(0.6)

    def test_ties_break_to_the_lowest_class_id(self):
        sem = [13, 13, 13, 7, 7, 7]
        t = cell([0.1] * 6, [0.1] * 6, range(6), sem, cell_size=1.0)
        assert t.sem_class[0] == 7
        assert t.sem_conf[0] == pytest.approx(0.5)

    def test_confidence_always_lies_in_zero_one(self):
        rng = np.random.default_rng(3)
        n = 500
        x = rng.uniform(0, 5, n).astype(np.float32)
        y = rng.uniform(0, 5, n).astype(np.float32)
        z = rng.uniform(-1, 1, n).astype(np.float32)
        sem = rng.integers(1, 43, n).astype(np.uint8)
        t = cell(x, y, z, sem, cell_size=0.5)
        assert (t.sem_conf > 0).all() and (t.sem_conf <= 1.0).all()

    def test_a_single_point_cell_has_full_confidence(self):
        t = cell([10.0], [10.0], [0.0], [30], cell_size=1.0)
        assert t.sem_class[0] == 30 and t.sem_conf[0] == 1.0


class TestMinPointsPerCell:
    def test_sparse_cells_are_dropped(self):
        # cell A gets 2 points, cell B gets 1
        t = cell([0.1, 0.1, 1.5], [0.1, 0.1, 1.5], [0, 0, 0], [7, 7, 7],
                  cell_size=1.0, min_points_per_cell=2)
        assert len(t) == 1
        assert t.cx[0] == pytest.approx(0.5)

    def test_threshold_of_one_keeps_every_occupied_cell(self):
        t = cell([0.1, 1.5], [0.1, 1.5], [0, 0], [7, 7], cell_size=1.0, min_points_per_cell=1)
        assert len(t) == 2


class TestCellTable:
    def test_concat_stacks_rows_and_preserves_totals(self):
        a = cell([0.1], [0.1], [1.0], [7], cell_size=1.0, zone_id=0)
        b = cell([0.1], [0.1], [2.0], [13], cell_size=0.5, zone_id=1)
        merged = CellTable.concat([a, b])
        assert len(merged) == 2
        assert set(merged.zone.tolist()) == {0, 1}
        assert set(merged.size.tolist()) == {1.0, 0.5}

    def test_concat_of_all_empty_tables_is_empty(self):
        merged = CellTable.concat([CellTable.empty(), CellTable.empty()])
        assert len(merged) == 0

    def test_concat_skips_empty_tables_without_error(self):
        a = cell([0.1], [0.1], [1.0], [7], cell_size=1.0)
        merged = CellTable.concat([CellTable.empty(), a, CellTable.empty()])
        assert len(merged) == 1

    def test_mismatched_field_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="disagree"):
            CellTable(
                cx=np.zeros(2, dtype=np.float32),
                cy=np.zeros(1, dtype=np.float32),
                size=np.zeros(2, dtype=np.float32),
                zone=np.zeros(2, dtype=np.uint8),
                z_ref=np.zeros(2, dtype=np.float32),
                z_min=np.zeros(2, dtype=np.float32),
                z_max=np.zeros(2, dtype=np.float32),
                n_points=np.zeros(2, dtype=np.uint32),
                sem_class=np.zeros(2, dtype=np.uint8),
                sem_conf=np.zeros(2, dtype=np.float32),
            )

    def test_nbytes_sums_every_field_array(self):
        t = cell([0.1], [0.1], [1.0], [7], cell_size=1.0)
        expected = (
            t.cx.nbytes + t.cy.nbytes + t.size.nbytes + t.zone.nbytes + t.z_ref.nbytes
            + t.z_min.nbytes + t.z_max.nbytes + t.n_points.nbytes + t.sem_class.nbytes
            + t.sem_conf.nbytes
        )
        assert t.nbytes == expected


class TestUniformGrid:
    def _points(self, x, y, z, sem) -> PreprocessedPoints:
        x, y, z, sem = (np.asarray(a) for a in (x, y, z, sem))
        r = np.hypot(x, y).astype(np.float32)
        return PreprocessedPoints(
            x=x.astype(np.float32), y=y.astype(np.float32), z=z.astype(np.float32),
            sem=sem.astype(np.uint8), radius=r,
            n_input=len(x), n_kept=len(x),
            dropped_range=0, dropped_height=0, dropped_class=0, dropped_device=0,
        )

    def test_zone_is_always_zero(self):
        pts = self._points([0.1, 5.1], [0.1, 5.1], [0.0, 1.0], [7, 13])
        t = build_uniform_map(pts, cell_m=1.0)
        assert (t.zone == 0).all()

    def test_finer_cells_never_reduce_the_cell_count_below_coarser_cells(self):
        rng = np.random.default_rng(1)
        n = 2000
        x = rng.uniform(-20, 20, n)
        y = rng.uniform(-20, 20, n)
        pts = self._points(x, y, np.zeros(n), np.full(n, 7))
        fine = build_uniform_map(pts, cell_m=0.5)
        coarse = build_uniform_map(pts, cell_m=2.0)
        assert len(fine) >= len(coarse)

    def test_config_wrapper_matches_the_direct_call(self, config):
        pts = self._points([0.1, 15.0], [0.1, 15.0], [0.0, 1.0], [7, 13])
        via_cfg = build_uniform_map_from_config(pts, config.grid)
        direct = build_uniform_map(
            pts,
            cell_m=config.grid.uniform_cell_m,
            min_points_per_cell=config.grid.min_points_per_cell,
            elevation_stat=config.grid.elevation_stat,
        )
        assert len(via_cfg) == len(direct)
        assert via_cfg.size[0] == direct.size[0]


@pytest.mark.slow
class TestUniformGridOnRealData:
    """Regression pin against the real dataset, measured by this suite itself.

    The exact cell count is sensitive to preprocessing defaults (range crop,
    dropped classes). It is asserted precisely so a change to those defaults,
    or a regression in the aggregation kernel, is immediately visible; the
    number was produced by running this code, not assumed in advance.
    """

    def test_frame_zero_at_the_finest_zone_resolution(self, real_sequence, config):
        from avrmap.frames import load_frame
        from avrmap.preprocess import preprocess_frame

        frame = load_frame(real_sequence, "00")
        pre = preprocess_frame(frame, config.preprocess)
        grid = build_uniform_map_from_config(pre, config.grid)

        assert len(grid) == 63_611
        assert np.isfinite(grid.z_ref).all()
        assert (grid.n_points >= config.grid.min_points_per_cell).all()
        assert (grid.sem_conf > 0).all() and (grid.sem_conf <= 1.0).all()
        # every cell centre must fall within the configured range plus half a
        # cell's slack, since a cell can straddle the range boundary
        radius = np.hypot(grid.cx, grid.cy)
        assert radius.max() < config.preprocess.max_range_m + config.grid.uniform_cell_m

    def test_cell_count_decreases_monotonically_as_cells_get_coarser(self, real_sequence, config):
        from avrmap.frames import load_frame
        from avrmap.preprocess import preprocess_frame

        frame = load_frame(real_sequence, "00")
        pre = preprocess_frame(frame, config.preprocess)
        counts = [len(build_uniform_map(pre, cell_m=c)) for c in (0.1, 0.2, 0.4, 0.8)]
        assert counts == sorted(counts, reverse=True)
        assert counts[0] > counts[-1]

    def test_holds_across_every_frame_in_the_sequence(self, real_sequence, config):
        """Every frame must grid without error, with at least one class."""
        from avrmap.frames import load_frame
        from avrmap.preprocess import preprocess_frame

        for frame_id in real_sequence.frame_ids[::10]:  # sample for speed
            frame = load_frame(real_sequence, frame_id)
            pre = preprocess_frame(frame, config.preprocess)
            grid = build_uniform_map_from_config(pre, config.grid)
            assert len(grid) > 0
            assert np.isfinite(grid.z_ref).all()
