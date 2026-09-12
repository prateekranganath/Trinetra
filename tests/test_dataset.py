"""Discovery, indexing and validation.

Failure modes are provoked on a synthetic tree. The real dataset is only ever
read, never modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avrmap.dataset import (
    DatasetError,
    discover_sequences,
    load_classes,
    require_sequence,
    validate_sequence,
)
from avrmap.frames import Frame, load_frame
from avrmap.semantics import UNLABELED_CLASS
from conftest import make_lidar_frame, make_semseg_frame, write_pickle


class TestDiscovery:
    def test_finds_sequence_at_the_root_itself(self, synthetic_sequence: Path):
        found = discover_sequences(synthetic_sequence)
        assert [s.seq_id for s in found] == [synthetic_sequence.name]
        assert found[0].frame_ids == ("00", "01", "02")

    def test_finds_sequence_nested_one_level_down(self, synthetic_sequence: Path):
        found = discover_sequences(synthetic_sequence.parent)
        assert len(found) == 1
        assert found[0].seq_id == "seq_a"

    def test_identifies_sequences_by_structure_not_by_name(
        self, synthetic_sequence: Path
    ):
        renamed = synthetic_sequence.rename(synthetic_sequence.parent / "renamed_007")
        found = discover_sequences(renamed.parent)
        assert [s.seq_id for s in found] == ["renamed_007"]

    def test_ignores_directories_without_both_halves(self, tmp_path: Path):
        (tmp_path / "not_a_sequence" / "lidar").mkdir(parents=True)
        write_pickle(
            tmp_path / "not_a_sequence" / "lidar" / "00.pkl", make_lidar_frame(10)
        )
        assert discover_sequences(tmp_path) == []

    def test_respects_the_depth_limit(self, synthetic_sequence: Path):
        parent = synthetic_sequence.parent  # holds seq_a and nothing else
        assert discover_sequences(parent, max_depth=0) == []
        assert len(discover_sequences(parent, max_depth=1)) == 1

    def test_missing_root_is_an_error(self, tmp_path: Path):
        with pytest.raises(DatasetError, match="not a directory"):
            discover_sequences(tmp_path / "nope")

    def test_require_sequence_names_the_alternatives(self, synthetic_sequence: Path):
        with pytest.raises(DatasetError, match="Available: seq_a"):
            require_sequence(synthetic_sequence.parent, seq_id="seq_b")


class TestPairing:
    def test_all_frames_pair_cleanly(self, synthetic_sequence: Path):
        seq = require_sequence(synthetic_sequence)
        report = validate_sequence(seq)
        assert report.ok
        assert report.n_frames_ok == 3
        assert report.classes_present == (5, 7, 13, 41)

    def test_unpaired_lidar_frame_is_reported(self, synthetic_sequence: Path):
        write_pickle(synthetic_sequence / "lidar" / "03.pkl", make_lidar_frame(50))
        seq = require_sequence(synthetic_sequence)
        assert seq.frame_ids == ("00", "01", "02")  # the orphan is not usable
        assert seq.lidar_only == ("03",)
        report = validate_sequence(seq)
        assert not report.ok
        assert any("no semseg counterpart" in e for e in report.errors)

    def test_row_count_mismatch_is_caught(self, synthetic_sequence: Path):
        write_pickle(
            synthetic_sequence / "annotations" / "semseg" / "01.pkl",
            make_semseg_frame(7),
        )
        report = validate_sequence(require_sequence(synthetic_sequence))
        assert not report.ok
        bad = [f for f in report.frames if f.frame_id == "01"][0]
        assert any("row count mismatch" in e for e in bad.errors)

    def test_missing_column_is_caught(self, synthetic_sequence: Path):
        frame = make_lidar_frame(120).drop(columns=["d"])
        write_pickle(synthetic_sequence / "lidar" / "00.pkl", frame)
        report = validate_sequence(require_sequence(synthetic_sequence))
        bad = [f for f in report.frames if f.frame_id == "00"][0]
        assert any("missing column" in e for e in bad.errors)

    def test_label_outside_the_declared_set_is_caught(self, synthetic_sequence: Path):
        frame = make_semseg_frame(120)
        frame.loc[0, "class"] = 99
        write_pickle(
            synthetic_sequence / "annotations" / "semseg" / "00.pkl", frame
        )
        report = validate_sequence(require_sequence(synthetic_sequence))
        assert any("not present in classes.json" in e for e in report.errors)

    def test_pose_count_mismatch_is_caught(self, synthetic_sequence: Path):
        poses_path = synthetic_sequence / "lidar" / "poses.json"
        poses = json.loads(poses_path.read_text())
        poses_path.write_text(json.dumps(poses[:2]), encoding="utf-8")
        report = validate_sequence(require_sequence(synthetic_sequence))
        assert any("poses but the sequence has" in e for e in report.errors)


class TestClasses:
    def test_class_map_is_int_keyed(self, synthetic_sequence: Path):
        seq = require_sequence(synthetic_sequence)
        classes = load_classes(seq.classes_path)
        assert classes[7] == "Road"
        assert all(isinstance(k, int) for k in classes)

    def test_missing_class_map_is_an_error(self, tmp_path: Path):
        with pytest.raises(DatasetError, match="not found"):
            load_classes(tmp_path / "classes.json")


class TestFrameLoading:
    def test_frame_arrays_have_the_intended_dtypes(self, synthetic_sequence: Path):
        seq = require_sequence(synthetic_sequence)
        frame = load_frame(seq, "01")
        assert isinstance(frame, Frame)
        assert frame.n_points == 130
        assert frame.xyz_world.dtype == "float32"
        assert frame.semantic.dtype == "uint8"
        assert frame.device.dtype == "uint8"
        assert frame.semantic.shape == (frame.n_points,)

    def test_map_frame_is_centred_on_the_ego(self, synthetic_sequence: Path):
        seq = require_sequence(synthetic_sequence)
        frame = load_frame(seq, "02")  # pose position is (2, 0, 0)
        shift = frame.xyz_world[:, 0] - frame.xyz_map()[:, 0]
        assert shift == pytest.approx(2.0, abs=1e-4)

    def test_unknown_frame_id_is_an_error(self, synthetic_sequence: Path):
        seq = require_sequence(synthetic_sequence)
        with pytest.raises(DatasetError, match="not part of sequence"):
            load_frame(seq, "99")


@pytest.mark.slow
class TestRealDataset:
    def test_every_frame_pairs_and_validates(self, real_sequence):
        report = validate_sequence(real_sequence)
        assert report.ok, report.errors
        assert report.n_frames_ok == len(real_sequence)
        assert all(f.index_match for f in report.frames)

    def test_unlabelled_points_warn_rather_than_fail(self, real_sequence):
        """Class 0 is undeclared but legitimate, so it must not fail validation."""
        report = validate_sequence(real_sequence)
        if report.n_unlabeled:
            assert any("class 0" in w for w in report.warnings)
            assert report.ok

    def test_poses_and_timestamps_cover_every_frame(self, real_sequence):
        report = validate_sequence(real_sequence, frame_ids=[real_sequence.frame_ids[0]])
        assert report.n_poses == len(real_sequence)
        assert report.n_timestamps == len(real_sequence)

    def test_every_label_is_declared_apart_from_the_unlabelled_sentinel(
        self, real_sequence
    ):
        """Checks all 80 frames, since class 0 appears in only one of them."""
        report = validate_sequence(real_sequence)
        undeclared = set(report.classes_present) - set(report.classes_declared)
        assert undeclared <= {UNLABELED_CLASS}
