"""Validate the PandaSet sequence before anything else is built on top of it.

Every number printed is read from the files at run time. Nothing here is a
constant copied from an earlier inspection.

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --config configs/default.yaml --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from avrmap.config import ConfigError, default_config_path, load_config
from avrmap.dataset import (
    DatasetError,
    ValidationReport,
    discover_sequences,
    validate_sequence,
)
from avrmap.frames import load_frame
from avrmap.geometry import planar_radius, world_to_ego, world_to_map
from avrmap.semantics import class_name, group_of


def _rule(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _report_frames(report: ValidationReport) -> None:
    _rule("Frame pairing")
    total = report.n_frames_checked
    print(f"  frames checked          {total}")
    print(f"  LiDAR/semseg pairs OK   {report.n_frames_ok}/{total}")
    bad = [f for f in report.frames if not f.ok]
    if bad:
        print(f"  frames with problems    {len(bad)}")
        for check in bad[:10]:
            for msg in check.errors:
                print(f"    frame {check.frame_id}: {msg}")
        if len(bad) > 10:
            print(f"    ... and {len(bad) - 10} more")
    else:
        print("  row counts and indices match on every checked frame")

    counts = report.point_counts
    if counts.size:
        print(
            f"  points per frame        min {counts.min():,}  "
            f"mean {int(counts.mean()):,}  max {counts.max():,}"
        )
        print(f"  points in checked set   {counts.sum():,}")


def _report_sequence_meta(report: ValidationReport) -> None:
    _rule("Sequence metadata")
    print(f"  poses                   {report.n_poses}")
    print(f"  timestamps              {report.n_timestamps}")
    if report.frame_interval_s is not None:
        rate = 1.0 / report.frame_interval_s if report.frame_interval_s else float("nan")
        print(
            f"  frame interval          {report.frame_interval_s:.3f} s "
            f"({rate:.1f} Hz)"
        )
    if report.path_length_m is not None:
        print(f"  ego path length         {report.path_length_m:.1f} m")


def _report_classes(report: ValidationReport) -> None:
    _rule("Semantic labels")
    declared = report.classes_declared
    present = report.classes_present
    print(f"  classes declared        {len(declared)}")
    print(f"  classes present         {len(present)}")
    for cid in present:
        print(f"    {cid:>3}  {class_name(declared, cid):<38} [{group_of(cid)}]")
    unused = sorted(set(declared) - set(present))
    if unused:
        print(f"  declared but unused     {len(unused)}: {unused}")


def _report_geometry(report: ValidationReport, seq) -> None:
    """Print frame extents and the coordinate-frame self-check."""
    _rule("Coordinate frames (first checked frame)")
    frame_id = report.frames[0].frame_id if report.frames else seq.frame_ids[0]
    frame = load_frame(seq, frame_id)
    world = frame.xyz_world
    print(f"  frame                   {frame_id}  ({frame.n_points:,} points)")
    for name, pts in (
        ("world", world),
        ("map (ego-centred)", world_to_map(world, frame.pose)),
        ("ego (body frame)", world_to_ego(world, frame.pose)),
    ):
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        print(
            f"  {name:<22}  x [{lo[0]:8.1f},{hi[0]:8.1f}]  "
            f"y [{lo[1]:8.1f},{hi[1]:8.1f}]  z [{lo[2]:7.1f},{hi[2]:7.1f}]"
        )

    radius = planar_radius(world_to_map(world, frame.pose))
    print()
    for limit in (10.0, 30.0, 60.0, 100.0):
        inside = float((radius < limit).mean()) * 100.0
        print(f"  within {limit:6.1f} m          {inside:5.1f}% of points")

    _rule("Coordinate-frame self-check")
    ego = world_to_ego(world, frame.pose)
    forward = frame.device == 1
    if forward.any():
        azimuth = np.degrees(np.arctan2(ego[forward, 1], ego[forward, 0]))
        mean_az = float(np.mean(azimuth))
        verdict = "PASS" if abs(mean_az) < 15.0 else "FAIL"
        print(
            f"  forward sensor azimuth  {mean_az:+.1f} deg from +x  "
            f"[{verdict}: expected near 0]"
        )
    else:
        print("  forward sensor azimuth  unavailable: no device-1 returns in this frame")

    idx = seq.frame_index(frame_id)
    if idx + 1 < len(report.poses):
        step_world = report.poses[idx + 1].position - report.poses[idx].position
        step_ego = step_world @ report.poses[idx].rotation.T
        verdict = "PASS" if step_ego[0] > abs(step_ego[1]) else "FAIL"
        print(
            f"  ego motion to next      "
            f"[{step_ego[0]:+.3f}, {step_ego[1]:+.3f}, {step_ego[2]:+.3f}] m  "
            f"[{verdict}: +x should dominate]"
        )
    else:
        print("  ego motion to next      unavailable: this is the last frame")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        default=str(default_config_path()),
        help="path to the YAML configuration (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--sequence", default=None, help="sequence id to validate (default: all found)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="check only the first, middle and last frame instead of every frame",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"config   {cfg.source_path}")
    print(f"root     {cfg.dataset.root}")

    try:
        sequences = discover_sequences(cfg.dataset.root, cfg.dataset.max_search_depth)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.sequence is not None:
        sequences = [s for s in sequences if s.seq_id == args.sequence]
        if not sequences:
            print(f"ERROR: sequence '{args.sequence}' not found", file=sys.stderr)
            return 2

    _rule("Discovery")
    if not sequences:
        print(
            f"  no sequence found under {cfg.dataset.root}\n"
            "  a sequence directory must hold lidar/*.pkl and "
            "annotations/semseg/*.pkl",
            file=sys.stderr,
        )
        return 2
    print(f"  sequences found         {len(sequences)}")
    for seq in sequences:
        print(
            f"    {seq.seq_id:<12} {len(seq)} paired frames  "
            f"({seq.root})"
        )

    exit_code = 0
    for seq in sequences:
        frame_ids = None
        if args.quick and len(seq) >= 3:
            frame_ids = [seq.frame_ids[0], seq.frame_ids[len(seq) // 2], seq.frame_ids[-1]]

        print(f"\n{'=' * 62}\nSequence '{seq.seq_id}'\n{'=' * 62}")
        report = validate_sequence(seq, frame_ids)
        _report_frames(report)
        _report_sequence_meta(report)
        _report_classes(report)
        if report.frames and report.frames[0].ok and report.poses:
            _report_geometry(report, seq)

        _rule("Result")
        for msg in report.warnings:
            print(f"  WARNING: {msg}")
        for msg in report.errors:
            print(f"  ERROR:   {msg}")
        if report.ok:
            print(f"  VALID: {report.n_frames_ok}/{report.n_frames_checked} frames usable")
        else:
            print("  INVALID: see the errors above")
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
