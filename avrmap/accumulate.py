"""Sliding-window multi-frame accumulation.

Building a persistent 2.5D map from more than one sweep needs a single
consistent reference position: every frame's LiDAR points already live in
the same gravity-aligned *world* frame (see :mod:`avrmap.geometry`), so
accumulation needs only one change from single-frame preprocessing — a
frame's points are positioned relative to the *reference* frame's ego
position instead of its own, via :func:`avrmap.preprocess.preprocess_frame`'s
``reference_pose`` argument. That places an older frame's points correctly
relative to where the ego *is now*, and the existing uniform and adaptive
grid builders consume the fused result completely unchanged.

**Moving objects are the real problem with this idea, not a footnote.**
Accumulation is only valid because the static world hasn't moved between
frames — a building seen three frames ago is still exactly there. A car is
not: fusing its points across frames would smear it into a comet trail
pointing back along its own path, which is actively misleading rather than
merely imprecise. By default, points whose semantic class is in
:data:`avrmap.semantics.DYNAMIC_CLASSES` are therefore kept only from the
*reference* (most recent) frame — the fused grid becomes a persistent,
lower-noise map of the static world, with moving things shown only at their
current, unsmeared position. ``include_dynamic_history=True`` turns this off
to demonstrate the smearing artifact directly rather than hiding it; both are
one call to :func:`accumulate_window` apart, and the choice is recorded on
the returned :class:`AccumulationResult`, never silent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PreprocessConfig
from .dataset import SequenceIndex
from .frames import load_frame
from .preprocess import PreprocessedPoints, preprocess_frame
from .semantics import DYNAMIC_CLASSES


@dataclass(frozen=True)
class AccumulationResult:
    """Points fused from a sliding window of frames, plus what went into it.

    Attributes:
        points: Preprocessed points from every window frame, positioned
            relative to the reference frame's ego position — feed this
            straight into ``build_uniform_map`` / ``build_adaptive_map``
            exactly as a single frame's ``PreprocessedPoints`` would be.
        reference_frame_id: The frame the window is anchored on — "now", in
            mapping terms. Distance zones and "near field" are relative to
            this frame's ego position, not any other frame's.
        window_frame_ids: Every frame id that was read, oldest first,
            ending with ``reference_frame_id``. Shorter than requested near
            the start of the sequence, where there is no earlier history.
        include_dynamic_history: Whether dynamic-class points from
            non-reference frames were kept (smearing them across the window)
            or dropped (the default — see the module docstring).
        n_points_per_frame: Points each window frame actually contributed,
            aligned index-for-index with ``window_frame_ids``, after every
            filter including the dynamic-history one.
    """

    points: PreprocessedPoints
    reference_frame_id: str
    window_frame_ids: tuple[str, ...]
    include_dynamic_history: bool
    n_points_per_frame: tuple[int, ...]

    @property
    def window_size(self) -> int:
        return len(self.window_frame_ids)


def select_window(
    frame_ids: tuple[str, ...], reference_frame_id: str, window_size: int
) -> tuple[str, ...]:
    """The ``window_size`` most recent frame ids up to and including
    ``reference_frame_id``, oldest first.

    Clamped at the start of the sequence rather than raising: requesting a
    10-frame window anchored on the sequence's second frame correctly yields
    just 2 frames instead of failing, since there is no earlier history to
    give.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be at least 1, got {window_size}")
    if reference_frame_id not in frame_ids:
        raise ValueError(f"'{reference_frame_id}' is not a known frame id")
    idx = frame_ids.index(reference_frame_id)
    start = max(0, idx - window_size + 1)
    return frame_ids[start : idx + 1]


def _drop_dynamic(points: PreprocessedPoints) -> PreprocessedPoints:
    """A copy of ``points`` with every dynamic-class point removed, folded
    into ``dropped_class`` since a class is exactly why these are dropped."""
    keep = ~np.isin(points.sem, np.asarray(DYNAMIC_CLASSES, dtype=points.sem.dtype))
    if keep.all():
        return points
    return PreprocessedPoints(
        x=points.x[keep], y=points.y[keep], z=points.z[keep],
        sem=points.sem[keep], radius=points.radius[keep],
        n_input=points.n_input, n_kept=int(keep.sum()),
        dropped_range=points.dropped_range, dropped_height=points.dropped_height,
        dropped_class=points.dropped_class + int((~keep).sum()),
        dropped_device=points.dropped_device,
    )


def _concat(parts: list[PreprocessedPoints]) -> PreprocessedPoints:
    return PreprocessedPoints(
        x=np.concatenate([p.x for p in parts]),
        y=np.concatenate([p.y for p in parts]),
        z=np.concatenate([p.z for p in parts]),
        sem=np.concatenate([p.sem for p in parts]),
        radius=np.concatenate([p.radius for p in parts]),
        n_input=sum(p.n_input for p in parts),
        n_kept=sum(p.n_kept for p in parts),
        dropped_range=sum(p.dropped_range for p in parts),
        dropped_height=sum(p.dropped_height for p in parts),
        dropped_class=sum(p.dropped_class for p in parts),
        dropped_device=sum(p.dropped_device for p in parts),
    )


def accumulate_window(
    seq: SequenceIndex,
    reference_frame_id: str,
    window_size: int,
    preprocess_cfg: PreprocessConfig,
    include_dynamic_history: bool = False,
) -> AccumulationResult:
    """Fuse up to ``window_size`` consecutive frames ending at
    ``reference_frame_id`` into one set of preprocessed points, positioned
    relative to the reference frame's ego position.

    A ``window_size`` of 1 degenerates to ordinary single-frame
    preprocessing (with ``reference_frame_id`` as that one frame), which is a
    useful sanity check: it must reproduce exactly what
    ``preprocess_frame(frame, cfg)`` gives for that frame alone.
    """
    window = select_window(seq.frame_ids, reference_frame_id, window_size)
    reference_pose = load_frame(seq, reference_frame_id).pose

    parts: list[PreprocessedPoints] = []
    counts: list[int] = []
    for frame_id in window:
        frame = load_frame(seq, frame_id)
        pre = preprocess_frame(frame, preprocess_cfg, reference_pose=reference_pose)
        if not include_dynamic_history and frame_id != reference_frame_id:
            pre = _drop_dynamic(pre)
        parts.append(pre)
        counts.append(pre.n_kept)

    return AccumulationResult(
        points=_concat(parts),
        reference_frame_id=reference_frame_id,
        window_frame_ids=window,
        include_dynamic_history=include_dynamic_history,
        n_points_per_frame=tuple(counts),
    )
