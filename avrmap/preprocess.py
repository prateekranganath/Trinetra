"""Point-cloud preprocessing: crop by range and height, drop noise classes and
unwanted sensors.

Everything here operates on the **map frame** (world axes translated to an
ego position, see :mod:`avrmap.geometry`), because that is the frame the
grids are built in. Preprocessing never rotates points, so elevation stays
exactly what it was in the source pickle.

Filters are computed as independent boolean masks before being combined, so a
caller can report *why* points were dropped, not just how many. The four
per-criterion counts are **not a partition**: a point can fail more than one
filter, and it is then counted under every filter it fails. Only ``n_kept``
and the combined keep mask reflect the actual result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PreprocessConfig
from .frames import Frame
from .geometry import Pose, world_to_map


@dataclass(frozen=True)
class PreprocessedPoints:
    """Points surviving preprocessing, in the map frame.

    Attributes:
        x, y, z: (M,) float32 coordinates, ego-centred and gravity-aligned.
        sem: (M,) uint8 semantic class.
        radius: (M,) float32 horizontal distance from the ego vehicle.
        n_input: Points in the source frame, before any filtering.
        n_kept: Points surviving every filter, i.e. ``len(x)``.
        dropped_range: Points outside ``[min_range_m, max_range_m)``.
        dropped_height: Points outside ``[z_min_m, z_max_m]``.
        dropped_class: Points whose class is in ``drop_classes``.
        dropped_device: Points whose sensor id is not in ``devices``.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    sem: np.ndarray
    radius: np.ndarray
    n_input: int
    n_kept: int
    dropped_range: int
    dropped_height: int
    dropped_class: int
    dropped_device: int

    @property
    def n_dropped(self) -> int:
        return self.n_input - self.n_kept

    @property
    def keep_fraction(self) -> float:
        return self.n_kept / self.n_input if self.n_input else 0.0


def preprocess_frame(
    frame: Frame, cfg: PreprocessConfig, reference_pose: Pose | None = None
) -> PreprocessedPoints:
    """Crop and filter one frame's points according to ``cfg``.

    Order does not matter for the result, since all filters combine with a
    single logical AND, but each is still reported independently.

    ``reference_pose`` positions the frame's points relative to a *different*
    ego pose than its own — every frame's ``xyz_world`` already lives in the
    same gravity-aligned world frame (see :mod:`avrmap.geometry`), so this is
    exactly one subtraction, not a new coordinate convention. Defaults to the
    frame's own pose, i.e. ordinary single-frame preprocessing.
    :mod:`avrmap.accumulate` is the one caller that passes something else, to
    place an older frame's points correctly relative to where the ego *is
    now* rather than where it was when that frame was captured.
    """
    pose = reference_pose if reference_pose is not None else frame.pose
    xyz = world_to_map(frame.xyz_world, pose)
    radius = np.hypot(xyz[:, 0], xyz[:, 1]).astype(np.float32, copy=False)
    sem = frame.semantic
    device = frame.device
    n = xyz.shape[0]

    range_mask = (radius >= cfg.min_range_m) & (radius < cfg.max_range_m)
    height_mask = (xyz[:, 2] >= cfg.z_min_m) & (xyz[:, 2] <= cfg.z_max_m)

    if cfg.drop_classes:
        drop_arr = np.asarray(cfg.drop_classes, dtype=sem.dtype)
        class_mask = ~np.isin(sem, drop_arr)
    else:
        class_mask = np.ones(n, dtype=bool)

    if cfg.devices:
        keep_devices = np.asarray(cfg.devices, dtype=device.dtype)
        device_mask = np.isin(device, keep_devices)
    else:
        device_mask = np.ones(n, dtype=bool)

    keep = range_mask & height_mask & class_mask & device_mask

    return PreprocessedPoints(
        x=np.ascontiguousarray(xyz[keep, 0]),
        y=np.ascontiguousarray(xyz[keep, 1]),
        z=np.ascontiguousarray(xyz[keep, 2]),
        sem=np.ascontiguousarray(sem[keep]),
        radius=np.ascontiguousarray(radius[keep]),
        n_input=n,
        n_kept=int(keep.sum()),
        dropped_range=int((~range_mask).sum()),
        dropped_height=int((~height_mask).sum()),
        dropped_class=int((~class_mask).sum()),
        dropped_device=int((~device_mask).sum()),
    )
