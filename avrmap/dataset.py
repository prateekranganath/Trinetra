"""Sequence discovery, indexing and validation.

Sequences are found by *structure*, never by folder name. Any directory holding
both ``lidar/*.pkl`` and ``annotations/semseg/*.pkl`` counts as one. That keeps
discovery working whether the sequence sits at the project root, inside a
``data/`` folder, or as ``data/001/``, and it survives the folder being renamed.

Nothing in this module ever writes inside a sequence directory.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import Pose, load_poses, load_timestamps, path_length
from .semantics import UNLABELED_CLASS

LIDAR_DIRNAME = "lidar"
SEMSEG_RELPATH = Path("annotations") / "semseg"
CLASSES_FILENAME = "classes.json"
POSES_FILENAME = "poses.json"
TIMESTAMPS_FILENAME = "timestamps.json"


class DatasetError(RuntimeError):
    """Raised when the dataset layout is unusable."""


@dataclass(frozen=True)
class SequenceIndex:
    """Resolved paths for one PandaSet sequence.

    ``frame_ids`` holds the zero-padded stems shared by the LiDAR and semseg
    files, for example ``"00"``. Only frames present in *both* directories are
    listed; anything unpaired is reported by :func:`validate_sequence`.
    """

    seq_id: str
    root: Path
    frame_ids: tuple[str, ...]
    lidar_only: tuple[str, ...] = ()
    semseg_only: tuple[str, ...] = ()

    @property
    def lidar_dir(self) -> Path:
        return self.root / LIDAR_DIRNAME

    @property
    def semseg_dir(self) -> Path:
        return self.root / SEMSEG_RELPATH

    @property
    def classes_path(self) -> Path:
        return self.semseg_dir / CLASSES_FILENAME

    @property
    def poses_path(self) -> Path:
        return self.lidar_dir / POSES_FILENAME

    @property
    def timestamps_path(self) -> Path:
        return self.lidar_dir / TIMESTAMPS_FILENAME

    def lidar_path(self, frame_id: str) -> Path:
        return self.lidar_dir / f"{frame_id}.pkl"

    def semseg_path(self, frame_id: str) -> Path:
        return self.semseg_dir / f"{frame_id}.pkl"

    def frame_index(self, frame_id: str) -> int:
        """Position of a frame id in the sequence, used to look up its pose."""
        try:
            return self.frame_ids.index(frame_id)
        except ValueError:
            raise DatasetError(
                f"frame '{frame_id}' is not part of sequence '{self.seq_id}'"
            ) from None

    def __len__(self) -> int:
        return len(self.frame_ids)


def _frame_stems(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {p.stem for p in directory.glob("*.pkl")}


def _is_sequence_dir(candidate: Path) -> bool:
    return bool(_frame_stems(candidate / LIDAR_DIRNAME)) and bool(
        _frame_stems(candidate / SEMSEG_RELPATH)
    )


def _index_sequence(root: Path, seq_id: str) -> SequenceIndex:
    lidar = _frame_stems(root / LIDAR_DIRNAME)
    semseg = _frame_stems(root / SEMSEG_RELPATH)
    paired = sorted(lidar & semseg)
    return SequenceIndex(
        seq_id=seq_id,
        root=root,
        frame_ids=tuple(paired),
        lidar_only=tuple(sorted(lidar - semseg)),
        semseg_only=tuple(sorted(semseg - lidar)),
    )


def discover_sequences(root: str | Path, max_depth: int = 2) -> list[SequenceIndex]:
    """Find every sequence at or below ``root``, breadth-first up to ``max_depth``.

    Returns them sorted by id. An empty list is a normal result, not an error;
    the caller decides how to report it.
    """
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(f"dataset root is not a directory: {root}")

    found: list[SequenceIndex] = []
    level = [root]
    for depth in range(max_depth + 1):
        next_level: list[Path] = []
        for candidate in level:
            if _is_sequence_dir(candidate):
                seq_id = candidate.name if depth > 0 else candidate.resolve().name
                found.append(_index_sequence(candidate, seq_id))
                continue  # a sequence is never nested inside another
            if depth < max_depth:
                next_level.extend(
                    p for p in sorted(candidate.iterdir()) if p.is_dir()
                )
        level = next_level
        if not level:
            break
    return sorted(found, key=lambda s: s.seq_id)


def require_sequence(
    root: str | Path, max_depth: int = 2, seq_id: str | None = None
) -> SequenceIndex:
    """Return one sequence, with an actionable error when that is impossible."""
    sequences = discover_sequences(root, max_depth)
    if not sequences:
        raise DatasetError(
            f"no sequence found under {root}. A sequence directory must contain "
            f"both {LIDAR_DIRNAME}/*.pkl and {SEMSEG_RELPATH.as_posix()}/*.pkl."
        )
    if seq_id is None:
        return sequences[0]
    for seq in sequences:
        if seq.seq_id == seq_id:
            return seq
    names = ", ".join(s.seq_id for s in sequences)
    raise DatasetError(f"sequence '{seq_id}' not found. Available: {names}")


def load_classes(path: str | Path) -> dict[int, str]:
    """Read ``classes.json`` into an int-keyed mapping of class id to name."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"semantic class map not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict) or not raw:
        raise DatasetError(f"{path} should hold a non-empty object")
    return {int(k): str(v) for k, v in raw.items()}


def _read_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


@dataclass
class FrameCheck:
    """Per-frame result of :func:`validate_sequence`."""

    frame_id: str
    n_lidar: int
    n_semseg: int
    index_match: bool
    classes: tuple[int, ...]
    n_unlabeled: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class ValidationReport:
    """Everything the validation CLI prints, as data rather than text."""

    seq_id: str
    root: Path
    n_frames_checked: int
    frames: list[FrameCheck] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    classes_declared: dict[int, str] = field(default_factory=dict)
    classes_present: tuple[int, ...] = ()
    n_poses: int = 0
    n_timestamps: int = 0
    frame_interval_s: float | None = None
    path_length_m: float | None = None
    poses: list[Pose] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and all(f.ok for f in self.frames)

    @property
    def n_frames_ok(self) -> int:
        return sum(1 for f in self.frames if f.ok)

    @property
    def point_counts(self) -> np.ndarray:
        return np.array([f.n_lidar for f in self.frames], dtype=np.int64)

    @property
    def n_unlabeled(self) -> int:
        return sum(f.n_unlabeled for f in self.frames)


LIDAR_COLUMNS = ("x", "y", "z", "i", "t", "d")
SEMSEG_COLUMN = "class"


def validate_sequence(
    seq: SequenceIndex, frame_ids: list[str] | None = None
) -> ValidationReport:
    """Check that every LiDAR frame has a usable, correctly paired semseg frame.

    Validation reads the real files rather than trusting the directory listing:
    row counts, index equality and column names are all confirmed per frame.
    """
    frame_ids = list(seq.frame_ids) if frame_ids is None else list(frame_ids)
    report = ValidationReport(
        seq_id=seq.seq_id, root=seq.root, n_frames_checked=len(frame_ids)
    )

    if seq.lidar_only:
        report.errors.append(
            f"{len(seq.lidar_only)} LiDAR frame(s) have no semseg counterpart: "
            f"{', '.join(seq.lidar_only[:5])}"
        )
    if seq.semseg_only:
        report.errors.append(
            f"{len(seq.semseg_only)} semseg frame(s) have no LiDAR counterpart: "
            f"{', '.join(seq.semseg_only[:5])}"
        )
    if not frame_ids:
        report.errors.append("the sequence contains no paired frames")
        return report

    try:
        report.classes_declared = load_classes(seq.classes_path)
    except DatasetError as exc:
        report.errors.append(str(exc))

    try:
        report.poses = load_poses(seq.poses_path)
        report.n_poses = len(report.poses)
        report.path_length_m = path_length(report.poses)
    except (OSError, ValueError) as exc:
        report.errors.append(f"could not read {seq.poses_path.name}: {exc}")

    try:
        stamps = load_timestamps(seq.timestamps_path)
        report.n_timestamps = int(stamps.size)
        if stamps.size > 1:
            report.frame_interval_s = float(np.median(np.diff(stamps)))
    except (OSError, ValueError) as exc:
        report.errors.append(f"could not read {seq.timestamps_path.name}: {exc}")

    if report.n_poses and report.n_poses != len(seq.frame_ids):
        report.errors.append(
            f"{seq.poses_path.name} holds {report.n_poses} poses but the sequence "
            f"has {len(seq.frame_ids)} paired frames"
        )
    if report.n_timestamps and report.n_timestamps != len(seq.frame_ids):
        report.errors.append(
            f"{seq.timestamps_path.name} holds {report.n_timestamps} entries but "
            f"the sequence has {len(seq.frame_ids)} paired frames"
        )

    seen_classes: set[int] = set()
    for frame_id in frame_ids:
        check = _validate_frame(seq, frame_id)
        seen_classes.update(check.classes)
        report.frames.append(check)
    report.classes_present = tuple(sorted(seen_classes))

    if report.classes_declared:
        undeclared = sorted(seen_classes - set(report.classes_declared))
        if UNLABELED_CLASS in undeclared:
            undeclared.remove(UNLABELED_CLASS)
            n_frames = sum(1 for f in report.frames if f.n_unlabeled)
            report.warnings.append(
                f"{report.n_unlabeled} point(s) in {n_frames} frame(s) carry class "
                f"{UNLABELED_CLASS}, which {CLASSES_FILENAME} does not declare. "
                "PandaSet uses it for unlabelled returns; preprocessing drops it "
                "by default."
            )
        if undeclared:
            report.errors.append(
                f"labels not present in {CLASSES_FILENAME}: {undeclared}"
            )
    return report


def _validate_frame(seq: SequenceIndex, frame_id: str) -> FrameCheck:
    lidar_path = seq.lidar_path(frame_id)
    semseg_path = seq.semseg_path(frame_id)
    check = FrameCheck(
        frame_id=frame_id, n_lidar=0, n_semseg=0, index_match=False, classes=()
    )
    try:
        lidar = _read_pickle(lidar_path)
        semseg = _read_pickle(semseg_path)
    except Exception as exc:  # unpickling can raise almost anything
        check.errors.append(f"could not unpickle: {type(exc).__name__}: {exc}")
        return check

    missing = [c for c in LIDAR_COLUMNS if c not in getattr(lidar, "columns", [])]
    if missing:
        check.errors.append(f"LiDAR frame is missing column(s) {missing}")
    if SEMSEG_COLUMN not in getattr(semseg, "columns", []):
        check.errors.append(f"semseg frame is missing the '{SEMSEG_COLUMN}' column")
    if check.errors:
        return check

    check.n_lidar = len(lidar)
    check.n_semseg = len(semseg)
    if check.n_lidar != check.n_semseg:
        check.errors.append(
            f"row count mismatch: {check.n_lidar} LiDAR points vs "
            f"{check.n_semseg} labels"
        )
        return check

    check.index_match = bool(np.array_equal(lidar.index.to_numpy(), semseg.index.to_numpy()))
    if not check.index_match:
        check.errors.append("LiDAR and semseg indices differ")

    labels = semseg[SEMSEG_COLUMN].to_numpy()
    check.classes = tuple(int(c) for c in np.unique(labels))
    check.n_unlabeled = int((labels == UNLABELED_CLASS).sum())

    finite = np.isfinite(lidar[["x", "y", "z"]].to_numpy()).all()
    if not finite:
        check.errors.append("LiDAR frame contains non-finite coordinates")
    return check
