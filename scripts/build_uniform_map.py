"""Build and summarize a uniform 2.5D map for one frame.

A manual sanity-check entry point for the preprocessing and uniform-grid
pipeline, in the same spirit as ``validate_dataset.py``: every number printed
comes from actually running the code, not from an earlier measurement.

Usage:
    python scripts/build_uniform_map.py
    python scripts/build_uniform_map.py --frame 42 --cell-m 0.2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from avrmap.config import ConfigError, default_config_path, load_config
from avrmap.dataset import DatasetError, load_classes, require_sequence
from avrmap.frames import load_frame
from avrmap.grids.uniform import build_uniform_map
from avrmap.preprocess import preprocess_frame
from avrmap.semantics import class_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--sequence", default=None, help="sequence id (default: first found)")
    parser.add_argument("--frame", default=None, help="frame id, e.g. 00 (default: first frame)")
    parser.add_argument(
        "--cell-m",
        type=float,
        default=None,
        help="override grid.uniform_cell_m from the config",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        seq = require_sequence(cfg.dataset.root, cfg.dataset.max_search_depth, args.sequence)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    frame_id = args.frame if args.frame is not None else seq.frame_ids[0]
    cell_m = args.cell_m if args.cell_m is not None else cfg.grid.uniform_cell_m

    t0 = time.perf_counter()
    frame = load_frame(seq, frame_id)
    load_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    pre = preprocess_frame(frame, cfg.preprocess)
    pre_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    grid = build_uniform_map(
        pre, cell_m=cell_m,
        min_points_per_cell=cfg.grid.min_points_per_cell,
        elevation_stat=cfg.grid.elevation_stat,
    )
    grid_ms = (time.perf_counter() - t0) * 1000

    classes = load_classes(seq.classes_path)

    print(f"sequence          {seq.seq_id}")
    print(f"frame             {frame_id}")
    print(f"cell size         {cell_m} m")
    print()
    print(f"load              {frame.n_points:>10,} points   {load_ms:6.1f} ms")
    print(
        f"preprocess        {pre.n_kept:>10,} kept     {pre_ms:6.1f} ms   "
        f"({pre.n_dropped:,} dropped: {pre.dropped_range:,} range, "
        f"{pre.dropped_height:,} height, {pre.dropped_class:,} class, "
        f"{pre.dropped_device:,} device)"
    )
    print(f"grid              {len(grid):>10,} cells    {grid_ms:6.1f} ms")
    print()

    if len(grid):
        print(f"z_ref range       [{grid.z_ref.min():.2f}, {grid.z_ref.max():.2f}] m")
        print(f"points per cell   min {grid.n_points.min()}  mean {grid.n_points.mean():.1f}  max {grid.n_points.max()}")
        print(f"mean confidence   {grid.sem_conf.mean():.3f}")
        print(f"table memory      {grid.nbytes / 1024:.1f} KiB")
        print(f"dense-equivalent  {int((2 * cfg.preprocess.max_range_m / cell_m) ** 2):,} cells "
              "[derived, not the actual storage]")
        print()
        print("cells by semantic class:")
        counts = np.bincount(grid.sem_class, minlength=256)
        for cid in np.flatnonzero(counts):
            print(f"  {int(cid):>3}  {class_name(classes, int(cid)):<38} {int(counts[cid]):>8,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
