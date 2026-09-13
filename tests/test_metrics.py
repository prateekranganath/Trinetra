"""Cost and quality metrics: timing, dense-equivalent memory, and per-band
elevation/semantic scoring against raw points.

Every case has an answer computable by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from avrmap.grids.common import CellTable
from avrmap.metrics import BandQuality, dense_equivalent_cells, median_build_time_s, quality_by_band
from avrmap.preprocess import PreprocessedPoints


class TestMedianBuildTime:
    def test_calls_build_fn_exactly_repeats_times(self):
        calls = []
        median_build_time_s(lambda: calls.append(1), repeats=5)
        assert len(calls) == 5

    def test_returns_a_non_negative_float(self):
        t = median_build_time_s(lambda: sum(range(1000)), repeats=3)
        assert isinstance(t, float) and t >= 0.0

    def test_rejects_fewer_than_one_repeat(self):
        with pytest.raises(ValueError, match="repeats"):
            median_build_time_s(lambda: None, repeats=0)

    def test_is_the_median_not_the_mean(self):
        # One slow call among fast ones should barely move a median but would
        # move a mean noticeably; simulate with a counter-driven sleep proxy.
        import time

        durations = iter([0.0, 0.0, 0.2, 0.0, 0.0])

        def fn():
            time.sleep(next(durations))

        t = median_build_time_s(fn, repeats=5)
        assert t < 0.05  # median of four ~0s calls and one 0.2s call is ~0


class TestDenseEquivalentCells:
    def test_matches_the_documented_four_million_figure(self):
        # 100 m extent, 0.1 m cells -> a 2000x2000 dense array
        assert dense_equivalent_cells(100.0, 0.1) == 4_000_000

    def test_scales_as_the_inverse_square_of_cell_size(self):
        base = dense_equivalent_cells(50.0, 1.0)
        finer = dense_equivalent_cells(50.0, 0.5)
        assert finer == base * 4

    def test_rejects_non_positive_inputs(self):
        with pytest.raises(ValueError):
            dense_equivalent_cells(0.0, 0.1)
        with pytest.raises(ValueError):
            dense_equivalent_cells(10.0, 0.0)


def make_points(x, y, z, sem) -> PreprocessedPoints:
    x, y, z, sem = (np.asarray(a, dtype=np.float32) for a in (x, y, z, sem))
    r = np.hypot(x, y).astype(np.float32)
    return PreprocessedPoints(
        x=x, y=y, z=z, sem=sem.astype(np.uint8), radius=r,
        n_input=len(x), n_kept=len(x),
        dropped_range=0, dropped_height=0, dropped_class=0, dropped_device=0,
    )


def two_cell_table() -> CellTable:
    # cell 0: z_ref=1.0, class=7 ; cell 1: z_ref=5.0, class=13
    return CellTable(
        cx=np.array([0.5, 10.5], dtype=np.float32),
        cy=np.array([0.5, 10.5], dtype=np.float32),
        size=np.array([1.0, 1.0], dtype=np.float32),
        zone=np.array([0, 1], dtype=np.uint8),
        z_ref=np.array([1.0, 5.0], dtype=np.float32),
        z_min=np.array([1.0, 5.0], dtype=np.float32),
        z_max=np.array([1.0, 5.0], dtype=np.float32),
        n_points=np.array([2, 2], dtype=np.uint32),
        sem_class=np.array([7, 13], dtype=np.uint8),
        sem_conf=np.array([1.0, 1.0], dtype=np.float32),
    )


class TestQualityByBand:
    BANDS = (0.0, 10.0, 30.0)

    def test_rmse_and_agreement_match_a_hand_computation(self):
        # radius 1,1 -> band 0-10 (cell 0); radius 15,15 -> band 10-30 (cell 1)
        points = make_points(
            x=[1, 1, 15, 15], y=[0, 0, 0, 0],
            z=[1.0, 3.0, 5.0, 8.0],   # errors vs z_ref: [0, 2] and [0, 3]
            sem=[7, 13, 13, 7],        # matches: [True, False] and [True, False]
        )
        table = two_cell_table()
        point_row = np.array([0, 0, 1, 1], dtype=np.int64)
        result = quality_by_band(points, table, point_row, self.BANDS)

        assert result[0].elevation_rmse_m == pytest.approx(np.sqrt(2), abs=1e-9)
        assert result[0].semantic_agreement == pytest.approx(0.5)
        assert result[0].n_points == 2 and result[0].n_unmatched == 0

        assert result[1].elevation_rmse_m == pytest.approx(np.sqrt(4.5), abs=1e-9)
        assert result[1].semantic_agreement == pytest.approx(0.5)

    def test_perfect_reconstruction_gives_zero_rmse_and_full_agreement(self):
        points = make_points(x=[1, 1], y=[0, 0], z=[1.0, 1.0], sem=[7, 7])
        table = two_cell_table()
        point_row = np.array([0, 0], dtype=np.int64)
        result = quality_by_band(points, table, point_row, self.BANDS)
        assert result[0].elevation_rmse_m == 0.0
        assert result[0].semantic_agreement == 1.0

    def test_unmatched_points_are_excluded_and_counted_separately(self):
        points = make_points(x=[1, 1], y=[0, 0], z=[1.0, 99.0], sem=[7, 41])
        table = two_cell_table()
        point_row = np.array([0, -1], dtype=np.int64)  # second point has no cell
        result = quality_by_band(points, table, point_row, self.BANDS)
        assert result[0].n_points == 1
        assert result[0].n_unmatched == 1
        assert result[0].elevation_rmse_m == 0.0  # only the matched point counts

    def test_a_band_with_no_points_reports_nan_not_zero(self):
        points = make_points(x=[1], y=[0], z=[1.0], sem=[7])
        table = two_cell_table()
        point_row = np.array([0], dtype=np.int64)
        result = quality_by_band(points, table, point_row, self.BANDS)
        assert result[0].n_points == 1
        assert result[1].n_points == 0
        assert np.isnan(result[1].elevation_rmse_m)
        assert np.isnan(result[1].semantic_agreement)

    def test_all_points_unmatched_is_nan_not_an_error(self):
        points = make_points(x=[1, 1], y=[0, 0], z=[1.0, 2.0], sem=[7, 13])
        table = two_cell_table()
        point_row = np.full(2, -1, dtype=np.int64)
        result = quality_by_band(points, table, point_row, self.BANDS)
        assert result[0].n_unmatched == 2
        assert np.isnan(result[0].elevation_rmse_m)

    def test_band_edges_are_half_open(self):
        # radius exactly 10.0 belongs to the second band, not the first
        points = make_points(x=[10.0], y=[0.0], z=[5.0], sem=[13])
        table = two_cell_table()
        point_row = np.array([1], dtype=np.int64)
        result = quality_by_band(points, table, point_row, self.BANDS)
        assert result[0].n_points == 0
        assert result[1].n_points == 1

    def test_mismatched_point_row_length_is_rejected(self):
        points = make_points(x=[1, 2], y=[0, 0], z=[1.0, 2.0], sem=[7, 7])
        table = two_cell_table()
        with pytest.raises(ValueError, match="point_row"):
            quality_by_band(points, table, np.array([0], dtype=np.int64), self.BANDS)

    def test_rejects_fewer_than_two_band_edges(self):
        points = make_points(x=[1], y=[0], z=[1.0], sem=[7])
        table = two_cell_table()
        with pytest.raises(ValueError, match="bands_m"):
            quality_by_band(points, table, np.array([0], dtype=np.int64), (0.0,))

    def test_band_label_is_human_readable(self):
        bq = BandQuality(r_min=0.0, r_max=10.0, n_points=1, n_unmatched=0,
                          elevation_rmse_m=0.0, semantic_agreement=1.0)
        assert bq.label == "0-10m"
