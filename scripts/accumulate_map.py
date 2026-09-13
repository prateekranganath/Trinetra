"""Fuse a sliding window of frames into one map and summarize the result.

Demonstrates why accumulation matters: more of the static world becomes
visible than any single sweep shows, because occlusions and gaps differ
frame to frame. Every number is measured by actually building the fused
grid, not assumed.

Usage:
    python scripts/accumulate_map.py --frame 20 --window 5
    python scripts/accumulate_map.py --frame 20 --window 5 --include-dynamic-history
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from avrmap.accumulate import accumulate_window
from avrmap.config import ConfigError, default_config_path, load_config
from avrmap.dataset import DatasetError, require_sequence
from avrmap.frames import load_frame
from avrmap.grids.adaptive import build_adaptive_map_from_config
from avrmap.grids.uniform import build_uniform_map
from avrmap.preprocess import preprocess_frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--sequence", default=None, help="sequence id (default: first found)")
    parser.add_argument("--frame", default=None, help="reference frame id (default: a mid-sequence frame)")
    parser.add_argument("--window", type=int, default=5, help="number of frames to fuse")
    parser.add_argument(
        "--include-dynamic-history", action="store_true",
        help="keep moving-object points from every frame instead of only the reference frame "
             "(demonstrates the smearing artifact rather than hiding it)",
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

    frame_id = args.frame if args.frame is not None else seq.frame_ids[len(seq) // 2]

    t0 = time.perf_counter()
    result = accumulate_window(
        seq, frame_id, args.window, cfg.preprocess,
        include_dynamic_history=args.include_dynamic_history,
    )
    fuse_ms = (time.perf_counter() - t0) * 1000

    single = preprocess_frame(load_frame(seq, frame_id), cfg.preprocess)

    t0 = time.perf_counter()
    fused_adaptive = build_adaptive_map_from_config(result.points, cfg.grid)
    adaptive_ms = (time.perf_counter() - t0) * 1000
    single_adaptive = build_adaptive_map_from_config(single, cfg.grid)
    fused_uniform = build_uniform_map(
        result.points, cfg.grid.finest_cell_m, cfg.grid.min_points_per_cell, cfg.grid.elevation_stat
    )

    print(f"sequence               {seq.seq_id}")
    print(f"reference frame        {frame_id}")
    print(f"window requested       {args.window} frames")
    print(f"window actually used   {result.window_frame_ids}  ({result.window_size} frames)")
    print(f"dynamic-class history  {'kept (smearing shown)' if result.include_dynamic_history else 'excluded (default)'}")
    print()
    print("points contributed per frame (oldest first):")
    for fid, n in zip(result.window_frame_ids, result.n_points_per_frame):
        marker = " <- reference" if fid == frame_id else ""
        print(f"  {fid}: {n:>8,}{marker}")
    print()
    print(f"single-frame points     {single.n_kept:>10,}")
    print(f"fused points             {result.points.n_kept:>10,}   ({fuse_ms:.0f} ms to fuse)")
    print()
    print(f"single-frame adaptive   {len(single_adaptive):>10,} cells")
    print(f"fused adaptive          {len(fused_adaptive):>10,} cells   ({adaptive_ms:.0f} ms to build)   "
          f"[+{(len(fused_adaptive)/len(single_adaptive)-1)*100:.0f}% more coverage]" if len(single_adaptive) else "")
    print(f"fused uniform_fine      {len(fused_uniform):>10,} cells")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
