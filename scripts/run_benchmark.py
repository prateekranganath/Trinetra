"""Benchmark uniform vs. adaptive gridding across the sequence.

Four methods are compared per frame:

  uniform_fine    uniform grid at the adaptive grid's finest cell size —
                  the primary cost baseline, since adaptive gives identical
                  near-field resolution, so any cell saving here is real.
  uniform_matched uniform grid whose cell size is searched (bisection, see
                  avrmap.grids.uniform.find_matching_uniform_cell_size) so
                  its cell count is close to the adaptive grid's — the
                  primary quality baseline: at an equal cell budget, does
                  adaptive spend its cells where they matter?
  uniform_coarse  uniform grid at the coarsest configured zone size — a
                  floor reference.
  adaptive        the foveated grid.

For every method, cost metrics (cells, build time, memory) are measured
directly, and quality metrics (elevation RMSE, semantic agreement) are scored
per distance band directly against the frame's own points — no synthetic
ground truth is used. Every number written is labelled in the technical plan
and in README.md as measured, derived, or unavailable; nothing here is
estimated or filled in after the fact.

Usage:
    python scripts/run_benchmark.py                       # all frames
    python scripts/run_benchmark.py --frames 10            # first 10 frames
    python scripts/run_benchmark.py --frames 00,10,20,79   # specific frames
    python scripts/run_benchmark.py --out results --repeats 5
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from avrmap.config import ConfigError, PipelineConfig, default_config_path, load_config
from avrmap.dataset import DatasetError, SequenceIndex, require_sequence
from avrmap.frames import load_frame
from avrmap.grids.adaptive import (
    build_adaptive_map_with_point_cells,
    uncovered_point_count,
)
from avrmap.grids.common import CellTable
from avrmap.grids.uniform import (
    build_uniform_map,
    build_uniform_map_with_point_cells,
    find_matching_uniform_cell_size,
)
from avrmap.metrics import dense_equivalent_cells, median_build_time_s, quality_by_band
from avrmap.preprocess import PreprocessedPoints, preprocess_frame

METHODS = ("uniform_fine", "uniform_matched", "uniform_coarse", "adaptive")


def _band_columns(bands_m: tuple[float, ...]) -> list[str]:
    return [f"{lo:g}_{hi:g}m" for lo, hi in zip(bands_m, bands_m[1:])]


def fieldnames(bands_m: tuple[float, ...]) -> list[str]:
    cols = [
        "frame_id", "method", "cell_size_m", "n_zones",
        "n_input_points", "n_kept_points",
        "n_cells", "build_time_ms", "table_kib", "dense_equivalent_cells",
        "n_unmatched_points", "n_uncovered_points",
    ]
    for band in _band_columns(bands_m):
        cols += [f"rmse_{band}", f"agree_{band}", f"n_{band}"]
    return cols


def _select_frames(seq: SequenceIndex, spec: str) -> list[str]:
    if spec == "all":
        return list(seq.frame_ids)
    if "," in spec:
        chosen = [f.strip() for f in spec.split(",")]
        unknown = [f for f in chosen if f not in seq.frame_ids]
        if unknown:
            raise DatasetError(f"frame id(s) not in sequence: {unknown}")
        return chosen
    n = int(spec)
    if n <= 0 or n > len(seq.frame_ids):
        raise DatasetError(f"--frames {n} is out of range for {len(seq.frame_ids)} frames")
    return list(seq.frame_ids[:n])


def _row(
    frame_id: str,
    method: str,
    cell_size_m: float,
    n_zones: int,
    pre: PreprocessedPoints,
    table: CellTable,
    build_time_s: float,
    dense_extent_m: float,
    n_uncovered: int,
    bands_m: tuple[float, ...],
    band_quality,
) -> dict:
    row = {
        "frame_id": frame_id,
        "method": method,
        "cell_size_m": round(cell_size_m, 4),
        "n_zones": n_zones,
        "n_input_points": pre.n_input,
        "n_kept_points": pre.n_kept,
        "n_cells": len(table),
        "build_time_ms": round(build_time_s * 1000, 3),
        "table_kib": round(table.nbytes / 1024, 2),
        "dense_equivalent_cells": dense_equivalent_cells(dense_extent_m, cell_size_m),
        "n_unmatched_points": sum(b.n_unmatched for b in band_quality),
        "n_uncovered_points": n_uncovered,
    }
    for band, bq in zip(_band_columns(bands_m), band_quality):
        row[f"rmse_{band}"] = "" if np.isnan(bq.elevation_rmse_m) else round(bq.elevation_rmse_m, 4)
        row[f"agree_{band}"] = "" if np.isnan(bq.semantic_agreement) else round(bq.semantic_agreement, 4)
        row[f"n_{band}"] = bq.n_points
    return row


def run_frame(frame_id: str, seq: SequenceIndex, cfg: PipelineConfig, matched_iterations: int) -> list[dict]:
    frame = load_frame(seq, frame_id)
    pre = preprocess_frame(frame, cfg.preprocess)
    bands = cfg.benchmark.bands_m
    repeats = cfg.benchmark.repeats
    max_range = cfg.preprocess.max_range_m
    rows = []

    # adaptive first: its cell count is the target uniform_matched searches for
    adaptive_table, adaptive_rows = build_adaptive_map_with_point_cells(
        pre, cfg.grid.zones, cfg.grid.min_points_per_cell, cfg.grid.elevation_stat
    )
    adaptive_time = median_build_time_s(
        lambda: build_adaptive_map_with_point_cells(
            pre, cfg.grid.zones, cfg.grid.min_points_per_cell, cfg.grid.elevation_stat
        ),
        repeats,
    )
    rows.append(
        _row(
            frame_id, "adaptive", cfg.grid.finest_cell_m, len(cfg.grid.zones), pre,
            adaptive_table, adaptive_time, max_range,
            uncovered_point_count(pre, cfg.grid.zones), bands,
            quality_by_band(pre, adaptive_table, adaptive_rows, bands),
        )
    )

    matched_size, matched_cells = find_matching_uniform_cell_size(
        pre, target_cells=len(adaptive_table),
        min_points_per_cell=cfg.grid.min_points_per_cell,
        elevation_stat=cfg.grid.elevation_stat,
        iterations=matched_iterations,
    )

    for method, cell_size in (
        ("uniform_fine", cfg.grid.finest_cell_m),
        ("uniform_matched", matched_size),
        ("uniform_coarse", cfg.grid.coarsest_cell_m),
    ):
        table, prow = build_uniform_map_with_point_cells(
            pre, cell_size, cfg.grid.min_points_per_cell, cfg.grid.elevation_stat
        )
        build_time = median_build_time_s(
            lambda cs=cell_size: build_uniform_map(
                pre, cs, cfg.grid.min_points_per_cell, cfg.grid.elevation_stat
            ),
            repeats,
        )
        rows.append(
            _row(
                frame_id, method, cell_size, 1, pre, table, build_time, max_range,
                0, bands, quality_by_band(pre, table, prow, bands),
            )
        )
    return rows


def summarize(rows: list[dict], bands_m: tuple[float, ...]) -> dict:
    """Aggregate per-frame rows into one figure per method.

    Every summary figure is a mean or ratio of already-measured per-frame
    values — still derived, never a fresh estimate.
    """
    by_method: dict[str, list[dict]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    summary: dict = {"n_frames": len({r["frame_id"] for r in rows}), "methods": {}}
    fine_mean_cells = np.mean([r["n_cells"] for r in by_method.get("uniform_fine", [{"n_cells": float("nan")}])])

    for method, method_rows in by_method.items():
        cells = np.array([r["n_cells"] for r in method_rows], dtype=np.float64)
        build_ms = np.array([r["build_time_ms"] for r in method_rows], dtype=np.float64)
        kib = np.array([r["table_kib"] for r in method_rows], dtype=np.float64)
        total_pipeline_s = float(np.sum(build_ms)) / 1000.0  # build only; see note below
        entry = {
            "mean_cells": float(np.mean(cells)),
            "median_cells": float(np.median(cells)),
            "mean_build_time_ms": float(np.mean(build_ms)),
            "mean_table_kib": float(np.mean(kib)),
            "cell_reduction_vs_uniform_fine_pct": (
                float((1 - np.mean(cells) / fine_mean_cells) * 100)
                if fine_mean_cells else float("nan")
            ),
            "build_only_fps_estimate": (
                float(len(method_rows) / total_pipeline_s) if total_pipeline_s > 0 else float("nan")
            ),
            "note_fps": (
                "derived from build_time_ms only, excludes frame load/preprocess "
                "and all rendering; see README for the full end-to-end figure"
            ),
            "bands": {},
        }
        for band in _band_columns(bands_m):
            rmse_vals = [r[f"rmse_{band}"] for r in method_rows if r[f"rmse_{band}"] != ""]
            agree_vals = [r[f"agree_{band}"] for r in method_rows if r[f"agree_{band}"] != ""]
            entry["bands"][band] = {
                "mean_elevation_rmse_m": float(np.mean(rmse_vals)) if rmse_vals else None,
                "mean_semantic_agreement": float(np.mean(agree_vals)) if agree_vals else None,
                "frames_with_data": len(rmse_vals),
            }
        summary["methods"][method] = entry

    summary["render_fps"] = "unavailable: no dashboard yet (Milestone 5)"
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--sequence", default=None, help="sequence id (default: first found)")
    parser.add_argument(
        "--frames", default="all",
        help="'all', an integer N (first N frames), or a comma-separated list of frame ids",
    )
    parser.add_argument("--out", default="results", help="output directory")
    parser.add_argument("--repeats", type=int, default=None, help="override benchmark.repeats")
    parser.add_argument(
        "--matched-iterations", type=int, default=12,
        help="bisection iterations for the uniform_matched cell-size search",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.repeats is not None:
        cfg = dataclasses.replace(
            cfg, benchmark=dataclasses.replace(cfg.benchmark, repeats=args.repeats)
        )

    try:
        seq = require_sequence(cfg.dataset.root, cfg.dataset.max_search_depth, args.sequence)
        frame_ids = _select_frames(seq, args.frames)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark.csv"
    summary_path = out_dir / "benchmark_summary.json"

    print(f"sequence          {seq.seq_id}")
    print(f"frames            {len(frame_ids)} of {len(seq)}")
    print(f"repeats           {cfg.benchmark.repeats}")
    print(f"matched search    {args.matched_iterations} bisection iterations")
    print(f"writing           {csv_path}")
    print()

    all_rows: list[dict] = []
    t_start = time.perf_counter()
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames(cfg.benchmark.bands_m))
        writer.writeheader()
        for i, frame_id in enumerate(frame_ids, 1):
            rows = run_frame(frame_id, seq, cfg, args.matched_iterations)
            for row in rows:
                writer.writerow(row)
            all_rows.extend(rows)
            fh.flush()
            elapsed = time.perf_counter() - t_start
            print(
                f"  [{i:>3}/{len(frame_ids)}] frame {frame_id}  "
                f"adaptive={rows[0]['n_cells']:,} cells  "
                f"({elapsed:.1f}s elapsed)"
            )

    summary = summarize(all_rows, cfg.benchmark.bands_m)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"done in {time.perf_counter() - t_start:.1f}s")
    print(f"wrote             {csv_path}  ({len(all_rows)} rows)")
    print(f"wrote             {summary_path}")
    print()
    print(f"{'method':<18}{'mean cells':>12}{'vs fine':>10}{'mean build':>13}")
    for method in METHODS:
        m = summary["methods"].get(method)
        if not m:
            continue
        print(
            f"{method:<18}{m['mean_cells']:>12,.0f}{m['cell_reduction_vs_uniform_fine_pct']:>9.1f}%"
            f"{m['mean_build_time_ms']:>11.1f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
