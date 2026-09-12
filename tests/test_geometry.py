"""Poses and coordinate frames.

The last class here is the important one. It pins down the heading convention
against the real data, so that if anyone "fixes" the transpose in
``geometry.world_to_ego`` the suite says so immediately.
"""

from __future__ import annotations

import numpy as np
import pytest

from avrmap.frames import load_frame
from avrmap.geometry import (
    Pose,
    ego_to_world,
    load_poses,
    path_length,
    planar_radius,
    quat_to_matrix,
    world_to_ego,
    world_to_map,
)


def yaw_pose(yaw_rad: float, position=(0.0, 0.0, 0.0)) -> Pose:
    """Pose of a vehicle whose forward axis points along world bearing ``yaw_rad``.

    The negated half-angle is not a typo and is the whole point of this helper.
    PandaSet stores the *world-to-ego* rotation, so a vehicle heading +90 degrees
    is recorded as a -90 degree quaternion. Building the fixture the intuitive
    way would quietly encode the opposite convention from the real files and
    make every test below agree with a wrong implementation.
    """
    half = -yaw_rad / 2.0
    return Pose(
        position=np.asarray(position, dtype=np.float64),
        quaternion=np.array([np.cos(half), 0.0, 0.0, np.sin(half)]),
    )


class TestQuaternion:
    def test_identity_quaternion_gives_the_identity_matrix(self):
        assert quat_to_matrix([1, 0, 0, 0]) == pytest.approx(np.eye(3))

    def test_matrix_is_orthonormal_with_unit_determinant(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            rot = quat_to_matrix(rng.normal(size=4))
            assert rot @ rot.T == pytest.approx(np.eye(3), abs=1e-12)
            assert np.linalg.det(rot) == pytest.approx(1.0)

    def test_non_unit_quaternions_are_normalised(self):
        assert quat_to_matrix([2, 0, 0, 0]) == pytest.approx(np.eye(3))

    def test_zero_quaternion_is_rejected(self):
        with pytest.raises(ValueError, match="non-zero"):
            quat_to_matrix([0, 0, 0, 0])

    def test_quarter_turn_about_z_maps_x_onto_y(self):
        rot = quat_to_matrix([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        assert rot @ np.array([1.0, 0, 0]) == pytest.approx([0, 1, 0], abs=1e-12)

    def test_stored_quaternion_is_the_world_to_ego_rotation(self):
        """Documents the convention that world_to_ego relies on.

        A vehicle heading 90 degrees left stores a -90 degree quaternion, and
        that matrix applied to the world direction of travel yields ego +x.
        """
        pose = yaw_pose(np.pi / 2)
        heading_world = np.array([0.0, 1.0, 0.0])
        assert pose.rotation @ heading_world == pytest.approx([1, 0, 0], abs=1e-12)


class TestFrames:
    def test_map_frame_only_translates(self):
        pose = yaw_pose(0.7, position=(5.0, -3.0, 1.0))
        pts = np.array([[1.0, 2.0, 3.0], [-4.0, 0.0, 0.5]])
        mapped = world_to_map(pts, pose)
        assert mapped == pytest.approx(pts - np.array([5.0, -3.0, 1.0]), abs=1e-5)

    def test_map_frame_preserves_vertical_differences(self):
        """Elevation must survive the map transform untouched."""
        pose = yaw_pose(1.3, position=(10.0, 20.0, 0.0))
        pts = np.array([[0.0, 0.0, 2.0], [30.0, -5.0, 2.0]])
        mapped = world_to_map(pts, pose)
        assert mapped[0, 2] == pytest.approx(mapped[1, 2])

    def test_ego_round_trip_returns_the_original_points(self):
        pose = Pose(
            position=np.array([3.0, -7.0, 1.5]),
            quaternion=np.array([0.9218, 0.0098, 0.0244, -0.3867]),
        )
        rng = np.random.default_rng(1)
        pts = rng.uniform(-100, 100, size=(500, 3))
        assert ego_to_world(world_to_ego(pts, pose), pose) == pytest.approx(pts, abs=1e-9)

    def test_ego_rotation_matches_a_known_yaw(self):
        """A vehicle heading 90 degrees left sees world +y as straight ahead."""
        pose = yaw_pose(np.pi / 2)
        ego = world_to_ego(np.array([[0.0, 1.0, 0.0]]), pose)
        assert ego[0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)

    def test_yaw_property_recovers_the_heading(self):
        for yaw in (-2.0, -0.4, 0.0, 1.1, 3.0):
            assert yaw_pose(yaw).yaw_rad == pytest.approx(yaw, abs=1e-9)

    def test_planar_radius_ignores_elevation(self):
        pts = np.array([[3.0, 4.0, 0.0], [3.0, 4.0, 99.0]])
        assert planar_radius(pts) == pytest.approx([5.0, 5.0])


class TestPoseFile:
    def test_poses_load_in_order(self, synthetic_sequence):
        poses = load_poses(synthetic_sequence / "lidar" / "poses.json")
        assert len(poses) == 3
        assert [p.position[0] for p in poses] == [0.0, 1.0, 2.0]

    def test_path_length_sums_the_steps(self, synthetic_sequence):
        poses = load_poses(synthetic_sequence / "lidar" / "poses.json")
        assert path_length(poses) == pytest.approx(2.0)

    def test_path_length_of_a_single_pose_is_zero(self, synthetic_sequence):
        poses = load_poses(synthetic_sequence / "lidar" / "poses.json")
        assert path_length(poses[:1]) == 0.0

    def test_malformed_pose_entry_is_an_error(self, tmp_path):
        bad = tmp_path / "poses.json"
        bad.write_text('[{"position": {"x": 0}}]', encoding="utf-8")
        with pytest.raises(ValueError, match="malformed"):
            load_poses(bad)


@pytest.mark.slow
class TestConventionAgainstRealData:
    """Regression guards for the empirically determined heading convention.

    Both checks are physical facts about the vehicle, not restatements of the
    code: the forward-facing sensor points where the car is going, and the car
    drives forwards.
    """

    def test_forward_sensor_forms_a_cone_around_positive_x(self, real_sequence):
        frame = load_frame(real_sequence, real_sequence.frame_ids[0])
        ego = world_to_ego(frame.xyz_world, frame.pose)
        forward = frame.device == 1
        assert forward.sum() > 1000, "expected a populated forward-facing sensor"
        azimuth = np.degrees(np.arctan2(ego[forward, 1], ego[forward, 0]))
        assert abs(azimuth.mean()) < 15.0
        assert abs(np.percentile(azimuth, 5)) < 45.0
        assert abs(np.percentile(azimuth, 95)) < 45.0

    def test_the_vehicle_drives_towards_positive_x(self, real_sequence):
        poses = load_poses(real_sequence.poses_path)
        steps = np.array(
            [
                (b.position - a.position) @ a.rotation.T
                for a, b in zip(poses, poses[1:])
            ]
        )
        forward, lateral = steps[:, 0], steps[:, 1]
        assert (forward > 0).all(), "the sequence contains no reversing"
        assert np.abs(lateral).max() < np.abs(forward).min()

    def test_the_world_frame_is_more_gravity_aligned_than_the_ego_frame(
        self, real_sequence
    ):
        """Justifies building the map on world axes rather than body axes."""
        frame = load_frame(real_sequence, real_sequence.frame_ids[0])
        road = np.isin(frame.semantic, [7, 8, 9, 10])
        assert road.sum() > 1000

        def tilt_degrees(points: np.ndarray) -> float:
            near = road & (planar_radius(frame.xyz_map()) < 60.0)
            pts = points[near]
            design = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
            coef, *_ = np.linalg.lstsq(design, pts[:, 2], rcond=None)
            return float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))

        world_tilt = tilt_degrees(frame.xyz_map())
        ego_tilt = tilt_degrees(world_to_ego(frame.xyz_world, frame.pose))
        assert world_tilt < 1.0
        assert ego_tilt > world_tilt
