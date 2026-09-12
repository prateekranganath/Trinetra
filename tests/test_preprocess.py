"""Range, height, class and device filtering.

Each filter is tested in isolation against synthetic points with known
positions/classes/devices, then combined to confirm overlaps are handled by a
plain AND rather than double-subtracting.
"""

from __future__ import annotations

import numpy as np
import pytest

from avrmap.config import PreprocessConfig
from avrmap.dataset import require_sequence
from avrmap.frames import Frame, load_frame
from avrmap.geometry import Pose
from avrmap.preprocess import preprocess_frame


def make_frame(x, y, z, sem, device=None) -> Frame:
    n = len(x)
    return Frame(
        seq_id="synthetic",
        frame_id="00",
        xyz_world=np.column_stack([x, y, z]).astype(np.float32),
        intensity=np.zeros(n, dtype=np.uint8),
        device=(np.zeros(n, dtype=np.uint8) if device is None else np.asarray(device, dtype=np.uint8)),
        point_time=np.zeros(n, dtype=np.float64),
        semantic=np.asarray(sem, dtype=np.uint8),
        pose=Pose(position=np.zeros(3), quaternion=np.array([1.0, 0, 0, 0])),
        timestamp=0.0,
    )


DEFAULT = PreprocessConfig()


class TestRangeFilter:
    def test_keeps_points_inside_the_half_open_interval(self):
        frame = make_frame([0, 5, 10, 99.9, 100], [0, 0, 0, 0, 0], [0] * 5, [7] * 5)
        cfg = PreprocessConfig(min_range_m=0.0, max_range_m=100.0, z_min_m=-100, z_max_m=100)
        out = preprocess_frame(frame, cfg)
        # radius 100 is excluded (upper bound is exclusive), radius 0 is included
        assert list(out.radius) == pytest.approx([0, 5, 10, 99.9])
        assert out.dropped_range == 1

    def test_inner_radius_excludes_points_too_close(self):
        frame = make_frame([0, 2, 5], [0, 0, 0], [0, 0, 0], [7, 7, 7])
        cfg = PreprocessConfig(min_range_m=1.0, max_range_m=100.0, z_min_m=-100, z_max_m=100)
        out = preprocess_frame(frame, cfg)
        assert out.n_kept == 2
        assert out.dropped_range == 1


class TestHeightFilter:
    def test_keeps_the_closed_interval(self):
        frame = make_frame([1, 1, 1, 1], [0, 0, 0, 0], [-10.0, -9.9, 30.0, 30.1], [7] * 4)
        cfg = PreprocessConfig(min_range_m=0, max_range_m=100, z_min_m=-10.0, z_max_m=30.0)
        out = preprocess_frame(frame, cfg)
        # both endpoints are inclusive
        assert out.n_kept == 3
        assert out.dropped_height == 1


class TestClassFilter:
    def test_drops_only_the_listed_classes(self):
        frame = make_frame([1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 7, 13])
        cfg = PreprocessConfig(
            min_range_m=0, max_range_m=100, z_min_m=-100, z_max_m=100, drop_classes=(1, 2)
        )
        out = preprocess_frame(frame, cfg)
        assert list(out.sem) == [7, 13]
        assert out.dropped_class == 2

    def test_empty_drop_list_keeps_every_class(self):
        frame = make_frame([1, 1], [0, 0], [0, 0], [1, 2])
        cfg = PreprocessConfig(min_range_m=0, max_range_m=100, z_min_m=-100, z_max_m=100, drop_classes=())
        out = preprocess_frame(frame, cfg)
        assert out.n_kept == 2
        assert out.dropped_class == 0


class TestDeviceFilter:
    def test_keeps_only_the_listed_devices(self):
        frame = make_frame([1, 1, 1], [0, 0, 0], [0, 0, 0], [7, 7, 7], device=[0, 1, 2])
        cfg = PreprocessConfig(min_range_m=0, max_range_m=100, z_min_m=-100, z_max_m=100, devices=(0, 1))
        out = preprocess_frame(frame, cfg)
        assert out.n_kept == 2
        assert out.dropped_device == 1

    def test_empty_device_list_keeps_every_sensor(self):
        frame = make_frame([1, 1], [0, 0], [0, 0], [7, 7], device=[0, 9])
        cfg = PreprocessConfig(min_range_m=0, max_range_m=100, z_min_m=-100, z_max_m=100, devices=())
        out = preprocess_frame(frame, cfg)
        assert out.n_kept == 2
        assert out.dropped_device == 0


class TestCombinedFiltering:
    def test_a_point_failing_two_filters_is_counted_in_both_but_dropped_once(self):
        # This point is both out of range and the wrong class; a single other
        # point is kept, so n_kept must be 1, not 1 - (double-subtracted).
        frame = make_frame(
            x=[1.0, 500.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            sem=[7, 1],
            device=[0, 0],
        )
        cfg = PreprocessConfig(
            min_range_m=0, max_range_m=100, z_min_m=-100, z_max_m=100, drop_classes=(1,)
        )
        out = preprocess_frame(frame, cfg)
        assert out.n_kept == 1
        assert out.n_dropped == 1
        assert out.dropped_range == 1
        assert out.dropped_class == 1

    def test_output_arrays_stay_aligned_by_point(self):
        frame = make_frame([1, 2, 3], [0, 0, 0], [0.1, 0.2, 0.3], [7, 13, 41])
        cfg = PreprocessConfig(min_range_m=0, max_range_m=100, z_min_m=-100, z_max_m=100)
        out = preprocess_frame(frame, cfg)
        assert list(out.z) == pytest.approx([0.1, 0.2, 0.3])
        assert list(out.sem) == [7, 13, 41]

    def test_keep_fraction_and_counts_are_consistent(self):
        frame = make_frame([1, 200, 1], [0, 0, 0], [0, 0, 0], [7, 7, 7])
        cfg = PreprocessConfig(min_range_m=0, max_range_m=100, z_min_m=-100, z_max_m=100)
        out = preprocess_frame(frame, cfg)
        assert out.n_input == 3
        assert out.n_kept == 2
        assert out.keep_fraction == pytest.approx(2 / 3)

    def test_preprocessing_uses_the_map_frame_not_world(self):
        """Radius and z must be computed relative to the ego, not the origin."""
        frame = make_frame([10.0], [0.0], [1.0], [7])
        object.__setattr__(
            frame, "pose", Pose(position=np.array([10.0, 0.0, 0.0]), quaternion=np.array([1.0, 0, 0, 0]))
        )
        cfg = PreprocessConfig(min_range_m=0, max_range_m=100, z_min_m=-100, z_max_m=100)
        out = preprocess_frame(frame, cfg)
        assert out.radius[0] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.slow
class TestRealFrame:
    def test_preprocessing_the_real_frame_zero(self, real_sequence):
        from avrmap.config import load_config, default_config_path

        cfg = load_config(default_config_path())
        frame = load_frame(real_sequence, "00")
        out = preprocess_frame(frame, cfg.preprocess)
        assert out.n_input == frame.n_points
        assert 0 < out.n_kept <= out.n_input
        assert out.dropped_class > 0  # frame 00 contains noise classes 2 and 4
        assert np.isfinite(out.x).all() and np.isfinite(out.z).all()
