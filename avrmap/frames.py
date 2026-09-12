"""Loading one LiDAR frame together with its semantic labels.

Frames leave this module as plain typed NumPy arrays. pandas is used only to
unpickle; nothing downstream depends on it. Coordinates become float32 and
labels uint8, which halves memory against the stored float64 and costs nothing
in accuracy at centimetre scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .dataset import SequenceIndex, _read_pickle
from .geometry import Pose, load_poses, load_timestamps, planar_radius, world_to_map


@dataclass(frozen=True)
class Frame:
    """One LiDAR sweep with per-point semantic labels.

    Attributes:
        seq_id: Sequence this frame came from.
        frame_id: Zero-padded frame stem, for example ``"07"``.
        xyz_world: (N, 3) float32 point coordinates in the world frame.
        intensity: (N,) uint8 return intensity, 0-255 as stored.
        device: (N,) uint8 sensor id. 0 is the spinning sensor, 1 forward-facing.
        point_time: (N,) float64 per-point epoch seconds.
        semantic: (N,) uint8 PandaSet class id.
        pose: Ego pose for this frame.
        timestamp: Frame epoch seconds from ``timestamps.json``.
    """

    seq_id: str
    frame_id: str
    xyz_world: np.ndarray
    intensity: np.ndarray
    device: np.ndarray
    point_time: np.ndarray
    semantic: np.ndarray
    pose: Pose
    timestamp: float

    @property
    def n_points(self) -> int:
        return int(self.xyz_world.shape[0])

    def xyz_map(self) -> np.ndarray:
        """(N, 3) float32 coordinates in the ego-centred, gravity-aligned frame."""
        return world_to_map(self.xyz_world, self.pose)

    def radius_map(self) -> np.ndarray:
        """(N,) float32 horizontal distance from the ego vehicle."""
        return planar_radius(self.xyz_map()).astype(np.float32, copy=False)


@lru_cache(maxsize=8)
def _load_arrays(lidar_path: str, semseg_path: str):
    """Unpickle and convert. Cached on the resolved paths, which are hashable."""
    lidar = _read_pickle(Path(lidar_path))
    semseg = _read_pickle(Path(semseg_path))
    if len(lidar) != len(semseg):
        raise ValueError(
            f"{Path(lidar_path).name}: {len(lidar)} points but {len(semseg)} labels. "
            "Run scripts/validate_dataset.py to locate the bad frames."
        )
    xyz = np.ascontiguousarray(lidar[["x", "y", "z"]].to_numpy(), dtype=np.float32)
    intensity = lidar["i"].to_numpy().astype(np.uint8, copy=False)
    device = lidar["d"].to_numpy().astype(np.uint8, copy=False)
    point_time = lidar["t"].to_numpy().astype(np.float64, copy=False)
    labels = semseg["class"].to_numpy()
    if labels.min() < 0 or labels.max() > 255:
        raise ValueError(
            f"{Path(semseg_path).name}: class ids outside 0-255 cannot be stored "
            f"as uint8 (min {labels.min()}, max {labels.max()})"
        )
    return xyz, intensity, device, point_time, labels.astype(np.uint8, copy=False)


@lru_cache(maxsize=4)
def _load_sequence_meta(poses_path: str, timestamps_path: str):
    return load_poses(poses_path), load_timestamps(timestamps_path)


def load_frame(seq: SequenceIndex, frame_id: str) -> Frame:
    """Load one frame of ``seq`` by its id.

    Repeated calls for the same frame hit an LRU cache, which matters for
    dashboard playback where the user steps back and forth.
    """
    index = seq.frame_index(frame_id)
    poses, stamps = _load_sequence_meta(
        str(seq.poses_path.resolve()), str(seq.timestamps_path.resolve())
    )
    if index >= len(poses):
        raise ValueError(
            f"frame '{frame_id}' is at position {index} but only {len(poses)} "
            f"poses are available in {seq.poses_path.name}"
        )
    xyz, intensity, device, point_time, semantic = _load_arrays(
        str(seq.lidar_path(frame_id).resolve()), str(seq.semseg_path(frame_id).resolve())
    )
    return Frame(
        seq_id=seq.seq_id,
        frame_id=frame_id,
        xyz_world=xyz,
        intensity=intensity,
        device=device,
        point_time=point_time,
        semantic=semantic,
        pose=poses[index],
        timestamp=float(stamps[index]) if index < stamps.size else float("nan"),
    )


def clear_cache() -> None:
    """Drop cached frames and sequence metadata."""
    _load_arrays.cache_clear()
    _load_sequence_meta.cache_clear()
