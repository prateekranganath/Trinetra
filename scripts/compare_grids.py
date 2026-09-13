"""Compare the adaptive grid against a uniform grid at the same finest resolution.

This is the practical demonstration of the project's central claim: the
adaptive grid keeps full near-field resolution while using far fewer cells
overall than a uniform grid fine enough to match it. Every number here is
measured by actually building both grids for the chosen frame, not looked up
from an earlier run. Milestone 4 turns this into a proper multi-baseline,
multi-frame benchmark; this script is the one-frame sanity check that comes
before that.

Usage:
    python scripts/compare_grids.py
    python scripts/compare_grids.py --frame 42
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from avrmap.config import ConfigError, default_config_path, load_config
from avrmap.dataset import DatasetError, require_sequence
from avrmap.frames import load_frame
from avrmap.grids.adaptive import build_adaptive_map_from_config, uncovered_point_count
from avrmap.grids.uniform import build_uniform_map
from avrmap.preprocess import preprocess_frame


def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--sequence", default=None, help="sequence id (default: first found)")
    parser.add_argument("--frame", default=None, help="frame id, e.g. 00 (default: first frame)")
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
    frame = load_frame(seq, frame_id)
    pre = preprocess_frame(frame, cfg.preprocess)

    fine_cell = cfg.grid.finest_cell_m
    coarse_cell = cfg.grid.coarsest_cell_m
    uniform_fine, t_fine = _timed(
        build_uniform_map, pre, cell_m=fine_cell,
        min_points_per_cell=cfg.grid.min_points_per_cell,
        elevation_stat=cfg.grid.elevation_stat,
    )
    uniform_coarse, t_coarse = _timed(
        build_uniform_map, pre, cell_m=coarse_cell,
        min_points_per_cell=cfg.grid.min_points_per_cell,
        elevation_stat=cfg.grid.elevation_stat,
    )
    adaptive, t_adaptive = _timed(build_adaptive_map_from_config, pre, cfg.grid)
    uncovered = uncovered_point_count(pre, cfg.grid.zones)

    print(f"sequence          {seq.seq_id}")
    print(f"frame             {frame_id}   ({pre.n_kept:,} points after preprocessing)")
    if uncovered:
        print(f"WARNING           {uncovered:,} points fall outside every configured zone")
    print()
    print(f"{'grid':<28}{'cells':>10}{'time':>10}{'mean pts/cell':>16}")
    print(
        f"{'uniform_fine (' + str(fine_cell) + ' m)':<28}{len(uniform_fine):>10,}"
        f"{t_fine:>9.1f}ms{uniform_fine.n_points.mean():>16.1f}"
    )
    print(
        f"{'uniform_coarse (' + str(coarse_cell) + ' m)':<28}{len(uniform_coarse):>10,}"
        f"{t_coarse:>9.1f}ms{uniform_coarse.n_points.mean():>16.1f}"
    )
    print(
        f"{'adaptive (' + str(len(cfg.grid.zones)) + ' zones)':<28}{len(adaptive):>10,}"
        f"{t_adaptive:>9.1f}ms{adaptive.n_points.mean():>16.1f}"
    )
    print()

    reduction = 1 - len(adaptive) / len(uniform_fine) if len(uniform_fine) else float("nan")
    print(f"adaptive vs uniform_fine    {reduction * 100:5.1f}% fewer cells   [measured]")
    print(
        f"adaptive vs uniform_fine    {adaptive.nbytes / 1024:.1f} KiB vs "
        f"{uniform_fine.nbytes / 1024:.1f} KiB table memory   [measured]"
    )
    print()

    print("adaptive grid, per zone:")
    print(f"  {'zone':<6}{'range':<14}{'cell':>7}{'cells':>10}{'points':>12}{'mean pts/cell':>16}")
    for zid, zone in enumerate(cfg.grid.zones):
        mask = adaptive.zone == zid
        n_cells = int(mask.sum())
        n_pts = int(adaptive.n_points[mask].sum()) if n_cells else 0
        mean_ppc = adaptive.n_points[mask].mean() if n_cells else 0.0
        print(
            f"  {zid:<6}{zone.r_min:>5.0f}-{zone.r_max:<7.0f}{zone.cell_m:>6.2f}m"
            f"{n_cells:>10,}{n_pts:>12,}{mean_ppc:>16.1f}"
        )

    near = np.hypot(adaptive.cx, adaptive.cy) < cfg.grid.zones[0].r_max
    near_fine = np.hypot(uniform_fine.cx, uniform_fine.cy) < cfg.grid.zones[0].r_max
    print()
    print(
        f"near-field (< {cfg.grid.zones[0].r_max:.0f} m) cell count: "
        f"adaptive {int(near.sum()):,} vs uniform_fine {int(near_fine.sum()):,}   [measured, should match closely]"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
