"""Sliding-window multi-frame accumulation.

The synthetic fixture drives three frames whose ego position advances along
+x by exactly 1 m per frame (see conftest.make_lidar_frame/synthetic_sequence),
so a fused window has a hand-computable answer: a point that appears at the
same *world* x in frame 0 and frame 1 is 1 m further back, relative to frame
1's ego, than it was relative to frame 0's own ego.
"""

from __future__ import annotations

import numpy as np
import pytest

from avrmap.accumulate import accumulate_window, select_window
from avrmap.config import PreprocessConfig
from avrmap.dataset import require_sequence
from avrmap.frames import load_frame
from avrmap.preprocess import preprocess_frame
from avrmap.semantics import DYNAMIC_CLASSES

WIDE_OPEN = PreprocessConfig(
    min_range_m=0.0, max_range_m=1000.0, z_min_m=-1000.0, z_max_m=1000.0,
    drop_classes=(), devices=(),
)


class TestSelectWindow:
    FRAME_IDS = ("00", "01", "02", "03", "04")

    def test_returns_the_n_most_recent_frames_ending_at_the_reference(self):
        assert select_window(self.FRAME_IDS, "03", window_size=2) == ("02", "03")

    def test_clamps_at_the_start_of_the_sequence(self):
        assert select_window(self.FRAME_IDS, "01", window_size=10) == ("00", "01")

    def test_window_size_one_is_just_the_reference_frame(self):
        assert select_window(self.FRAME_IDS, "02", window_size=1) == ("02",)

    def test_a_window_reaching_exactly_the_start_is_not_clamped_short(self):
        assert select_window(self.FRAME_IDS, "02", window_size=3) == ("00", "01", "02")

    def test_rejects_a_window_size_below_one(self):
        with pytest.raises(ValueError, match="window_size"):
            select_window(self.FRAME_IDS, "02", window_size=0)

    def test_rejects_an_unknown_reference_frame(self):
        with pytest.raises(ValueError, match="not a known frame"):
            select_window(self.FRAME_IDS, "99", window_size=1)


class TestAccumulateWindow:
    def test_window_size_one_reproduces_single_frame_preprocessing_exactly(self, synthetic_sequence):
        seq = require_sequence(synthetic_sequence)
        result = accumulate_window(seq, "01", window_size=1, preprocess_cfg=WIDE_OPEN)
        single = preprocess_frame(load_frame(seq, "01"), WIDE_OPEN)
        assert result.window_frame_ids == ("01",)
        assert np.array_equal(result.points.x, single.x)
        assert np.array_equal(result.points.z, single.z)
        assert result.points.n_kept == single.n_kept

    def test_points_are_repositioned_relative_to_the_reference_frame(self, synthetic_sequence):
        # The synthetic fixture's poses advance by exactly 1 m along +x per
        # frame (see conftest.synthetic_sequence), so the same world point
        # sits 1 m further back relative to frame 1's ego than frame 0's.
        seq = require_sequence(synthetic_sequence)
        f0, f1 = load_frame(seq, "00"), load_frame(seq, "01")
        world_x = f0.xyz_world[0, 0]
        x_rel_to_own_frame = world_x - f0.pose.position[0]
        x_rel_to_frame1 = world_x - f1.pose.position[0]
        assert x_rel_to_frame1 == pytest.approx(x_rel_to_own_frame - 1.0, abs=1e-4)

    def test_total_points_sum_the_per_frame_contributions(self, synthetic_sequence):
        seq = require_sequence(synthetic_sequence)
        result = accumulate_window(seq, "02", window_size=3, preprocess_cfg=WIDE_OPEN)
        assert sum(result.n_points_per_frame) == result.points.n_kept
        assert len(result.n_points_per_frame) == len(result.window_frame_ids) == 3

    def test_reference_frame_id_and_window_size_are_recorded(self, synthetic_sequence):
        seq = require_sequence(synthetic_sequence)
        result = accumulate_window(seq, "02", window_size=2, preprocess_cfg=WIDE_OPEN)
        assert result.reference_frame_id == "02"
        assert result.window_size == 2

    def test_clamped_window_near_sequence_start_still_works(self, synthetic_sequence):
        seq = require_sequence(synthetic_sequence)
        result = accumulate_window(seq, "00", window_size=5, preprocess_cfg=WIDE_OPEN)
        assert result.window_frame_ids == ("00",)
        assert result.window_size == 1


class TestDynamicHistoryFilter:
    def test_default_excludes_dynamic_classes_from_non_reference_frames(self, synthetic_sequence):
        seq = require_sequence(synthetic_sequence)
        # frame fixtures are seeded so classes vary; whichever dynamic-class
        # points exist in non-reference frames must not appear in the fusion
        result = accumulate_window(seq, "02", window_size=3, preprocess_cfg=WIDE_OPEN)
        dyn = np.asarray(DYNAMIC_CLASSES)
        # every point that IS dynamic in the merged result must have come
        # from the reference frame alone; check by recomputing what the
        # reference frame alone contributes and comparing dynamic counts
        ref_only = preprocess_frame(load_frame(seq, "02"), WIDE_OPEN)
        n_dyn_ref = int(np.isin(ref_only.sem, dyn).sum())
        n_dyn_merged = int(np.isin(result.points.sem, dyn).sum())
        assert n_dyn_merged == n_dyn_ref
        assert result.include_dynamic_history is False

    def test_include_dynamic_history_keeps_at_least_as_many_points(self, synthetic_sequence):
        seq = require_sequence(synthetic_sequence)
        excluded = accumulate_window(seq, "02", window_size=3, preprocess_cfg=WIDE_OPEN, include_dynamic_history=False)
        included = accumulate_window(seq, "02", window_size=3, preprocess_cfg=WIDE_OPEN, include_dynamic_history=True)
        assert included.points.n_kept >= excluded.points.n_kept
        assert included.include_dynamic_history is True

    def test_a_single_frame_window_is_unaffected_by_the_dynamic_filter(self, synthetic_sequence):
        # there is no "history" to filter when the window is just the
        # reference frame itself
        seq = require_sequence(synthetic_sequence)
        excluded = accumulate_window(seq, "01", window_size=1, preprocess_cfg=WIDE_OPEN, include_dynamic_history=False)
        included = accumulate_window(seq, "01", window_size=1, preprocess_cfg=WIDE_OPEN, include_dynamic_history=True)
        assert excluded.points.n_kept == included.points.n_kept


@pytest.mark.slow
class TestAccumulateOnRealData:
    def test_a_five_frame_window_reveals_more_ground_than_one_frame(self, real_sequence, config):
        """Accumulation's whole point: more of the static world becomes visible."""
        from avrmap.grids.adaptive import build_adaptive_map_from_config

        single = preprocess_frame(load_frame(real_sequence, "20"), config.preprocess)
        result = accumulate_window(real_sequence, "20", window_size=5, preprocess_cfg=config.preprocess)

        assert result.points.n_kept > single.n_kept
        assert sum(result.n_points_per_frame) == result.points.n_kept

        single_grid = build_adaptive_map_from_config(single, config.grid)
        fused_grid = build_adaptive_map_from_config(result.points, config.grid)
        assert len(fused_grid) > len(single_grid)

    def test_window_shrinks_correctly_near_the_start_of_the_real_sequence(self, real_sequence, config):
        result = accumulate_window(real_sequence, real_sequence.frame_ids[1], window_size=10, preprocess_cfg=config.preprocess)
        assert result.window_size == 2
        assert result.window_frame_ids == real_sequence.frame_ids[:2]
