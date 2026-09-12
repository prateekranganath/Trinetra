"""Poses and coordinate frames.

Three frames matter in this project.

**World.** The frame the PandaSet ``lidar/*.pkl`` files are already stored in. Its
origin sits at the ego position of frame 0. Measured on this sequence, it is
gravity-aligned: a plane fitted to road-class points within 60 m is tilted 0.30
degrees.

**Map.** World axes translated so the origin follows the ego vehicle. No rotation
is applied, so map Z is still gravity-aligned and elevation stays meaningful.
This is the frame the 2.5D grids are built in. Because the distance zones are
radial, they need only the ego position, so the heading never enters the core
pipeline.

**Ego.** World rotated into the vehicle body frame, x forward. Used only for
optional heading-aligned views. Note that the ground is tilted 3.06 degrees in
this frame, which is exactly why the map frame does not use it.

The heading convention was determined empirically rather than assumed. Reading
the stored quaternion as (w, x, y, z) into a standard rotation matrix ``R``, the
transform that puts the vehicle's forward direction on +x is ``R @ (p - T)``, not
its transpose. Two independent checks agree: the forward-facing sensor (device 1)
forms a cone centred on +x, and the frame-to-frame motion vector points along +x.
``tests/test_geometry.py`` asserts both against the real data so the convention
cannot silently regress.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Pose:
    """Ego pose for one frame.

    Attributes:
        position: (3,) float64 ego origin in world coordinates.
        quaternion: (4,) float64 heading as (w, x, y, z), exactly as stored.
    """

    position: np.ndarray
    quaternion: np.ndarray

    @property
    def rotation(self) -> np.ndarray:
        """(3, 3) matrix mapping world directions to ego directions."""
        return quat_to_matrix(self.quaternion)

    @property
    def yaw_rad(self) -> float:
        """Vehicle heading in the world XY plane, radians, +x of ego in world."""
        forward_world = ego_to_world_direction(np.array([[1.0, 0.0, 0.0]]), self)[0]
        return float(np.arctan2(forward_world[1], forward_world[0]))


def quat_to_matrix(quat) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion to a 3x3 rotation matrix.

    The quaternion is normalised first, so slightly non-unit input from JSON is
    tolerated.
    """
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError(f"quaternion must be finite and non-zero, got {quat!r}")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_poses(path: str | Path) -> list[Pose]:
    """Read ``lidar/poses.json`` into one Pose per frame."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} should hold a non-empty list of poses")
    poses = []
    for i, item in enumerate(raw):
        try:
            pos = item["position"]
            head = item["heading"]
            poses.append(
                Pose(
                    position=np.array(
                        [pos["x"], pos["y"], pos["z"]], dtype=np.float64
                    ),
                    quaternion=np.array(
                        [head["w"], head["x"], head["y"], head["z"]], dtype=np.float64
                    ),
                )
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path} entry {i} is malformed: {exc}") from exc
    return poses


def load_timestamps(path: str | Path) -> np.ndarray:
    """Read a ``timestamps.json`` list of epoch seconds."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    stamps = np.asarray(raw, dtype=np.float64)
    if stamps.ndim != 1 or stamps.size == 0:
        raise ValueError(f"{path} should hold a non-empty flat list of timestamps")
    return stamps


def world_to_map(points_xyz: np.ndarray, pose: Pose) -> np.ndarray:
    """Translate world points so the ego position is the origin.

    No rotation is applied, so the result stays gravity-aligned. This is the
    frame the 2.5D grids are built in.
    """
    pts = np.asarray(points_xyz)
    return (pts - pose.position.astype(pts.dtype, copy=False)).astype(
        np.float32, copy=False
    )


def world_to_ego(points_xyz: np.ndarray, pose: Pose) -> np.ndarray:
    """Rotate and translate world points into the vehicle body frame, x forward."""
    pts = np.asarray(points_xyz, dtype=np.float64)
    return (pts - pose.position) @ pose.rotation.T


def ego_to_world(points_ego: np.ndarray, pose: Pose) -> np.ndarray:
    """Inverse of :func:`world_to_ego`."""
    pts = np.asarray(points_ego, dtype=np.float64)
    return pts @ pose.rotation + pose.position


def ego_to_world_direction(directions: np.ndarray, pose: Pose) -> np.ndarray:
    """Rotate ego-frame directions into world, ignoring translation."""
    return np.asarray(directions, dtype=np.float64) @ pose.rotation


def planar_radius(points_xyz: np.ndarray) -> np.ndarray:
    """Horizontal distance from the origin, which is the ego in the map frame.

    Distance zones are defined on this radius rather than on 3D range, because a
    2.5D grid cell is a footprint on the ground plane.
    """
    pts = np.asarray(points_xyz)
    return np.hypot(pts[:, 0], pts[:, 1])


def path_length(poses) -> float:
    """Total distance travelled along a sequence of poses, in metres."""
    pos = np.array([p.position for p in poses], dtype=np.float64)
    if len(pos) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
