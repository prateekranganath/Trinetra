"""Streamlit dashboard: browse the sequence, compare uniform vs. adaptive maps.

Run with:
    streamlit run app/dashboard.py

This module only wires Streamlit widgets to the `avrmap` library and to
`panels.py`'s figure builders — it loads no pickles, filters no points, and
builds no grids itself. Every heavy call is cached (`st.cache_resource` for
dataset discovery, `st.cache_data` for the per-frame load/preprocess/grid
bundle) so moving a slider re-renders instantly instead of re-reading pkl
files.

Scope note: this is Milestone 5, the static dashboard. Every required view
renders for any frame. Play/pause autoplay, live-editable zone and filter
controls that trigger a rebuild, and measured UI frame rate are Milestone 6;
the zone ladder and preprocessing filters below are shown read-only.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import panels  # noqa: E402
from avrmap.config import ConfigError, default_config_path, load_config  # noqa: E402
from avrmap.dataset import (  # noqa: E402
    discover_sequences,
    load_classes,
    require_sequence,
    validate_sequence,
)
from avrmap.frames import load_frame  # noqa: E402
from avrmap.grids.adaptive import (  # noqa: E402
    build_adaptive_map_with_point_cells,
    uncovered_point_count,
)
from avrmap.grids.uniform import (  # noqa: E402
    build_uniform_map_with_point_cells,
    find_matching_uniform_cell_size,
)
from avrmap.metrics import dense_equivalent_cells, quality_by_band  # noqa: E402
from avrmap.preprocess import preprocess_frame  # noqa: E402
from avrmap.render import rasterize_scalar, rasterize_semantic, rasterize_zone  # noqa: E402
from avrmap.semantics import GROUP_NAMES, build_group_lookup, groups_mask  # noqa: E402

st.set_page_config(page_title="Trinetra — Adaptive LiDAR Mapping", layout="wide")


# --------------------------------------------------------------------------
# Cached data access. Only primitive (hashable, cheap-to-hash) arguments are
# passed into cached functions — config and sequence are always reloaded from
# a path/id inside, which is a few milliseconds, rather than risking
# Streamlit's hashing of custom dataclasses.
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def discover(config_path: str):
    cfg = load_config(config_path)
    sequences = discover_sequences(cfg.dataset.root, cfg.dataset.max_search_depth)
    return cfg, sequences


@st.cache_resource(show_spinner=False)
def validate(config_path: str, seq_id: str):
    cfg = load_config(config_path)
    seq = require_sequence(cfg.dataset.root, cfg.dataset.max_search_depth, seq_id=seq_id)
    return validate_sequence(seq)


@st.cache_data(show_spinner="Loading frame, preprocessing, building both grids...")
def get_frame_bundle(config_path: str, seq_id: str, frame_id: str):
    cfg = load_config(config_path)
    seq = require_sequence(cfg.dataset.root, cfg.dataset.max_search_depth, seq_id=seq_id)
    frame = load_frame(seq, frame_id)
    pre = preprocess_frame(frame, cfg.preprocess)
    uniform = build_uniform_map_with_point_cells(
        pre, cfg.grid.uniform_cell_m, cfg.grid.min_points_per_cell, cfg.grid.elevation_stat
    )
    adaptive = build_adaptive_map_with_point_cells(
        pre, cfg.grid.zones, cfg.grid.min_points_per_cell, cfg.grid.elevation_stat
    )
    return cfg, frame, pre, uniform, adaptive


@st.cache_data(show_spinner="Building uniform_coarse and searching for uniform_matched...")
def get_extra_baselines(config_path: str, seq_id: str, frame_id: str, target_cells: int, matched_iterations: int):
    cfg = load_config(config_path)
    seq = require_sequence(cfg.dataset.root, cfg.dataset.max_search_depth, seq_id=seq_id)
    frame = load_frame(seq, frame_id)
    pre = preprocess_frame(frame, cfg.preprocess)
    coarse = build_uniform_map_with_point_cells(
        pre, cfg.grid.coarsest_cell_m, cfg.grid.min_points_per_cell, cfg.grid.elevation_stat
    )
    matched_size, matched_cells = find_matching_uniform_cell_size(
        pre, target_cells, cfg.grid.min_points_per_cell, cfg.grid.elevation_stat,
        iterations=matched_iterations,
    )
    matched = build_uniform_map_with_point_cells(
        pre, matched_size, cfg.grid.min_points_per_cell, cfg.grid.elevation_stat
    )
    return coarse, matched, matched_size


@st.cache_data(show_spinner=False)
def load_summary_json(path: str):
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


def main() -> None:
    st.title("Trinetra — Adaptive Variable-Resolution 2.5D LiDAR Mapping")

    config_path = str(default_config_path())
    try:
        cfg, sequences = discover(config_path)
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()

    if not sequences:
        st.error(
            f"No PandaSet sequence found under {cfg.dataset.root}. A sequence "
            "directory needs both lidar/*.pkl and annotations/semseg/*.pkl. "
            "See README.md for the expected layout."
        )
        st.stop()

    # --- sidebar: dataset & frame selection --------------------------------
    st.sidebar.header("Dataset")
    seq_ids = [s.seq_id for s in sequences]
    seq_id = st.sidebar.selectbox("Sequence", seq_ids, index=0)
    seq = next(s for s in sequences if s.seq_id == seq_id)
    st.sidebar.caption(f"{len(seq)} paired frames found at `{seq.root}`")

    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0
    st.session_state.frame_idx = min(st.session_state.frame_idx, len(seq) - 1)

    st.sidebar.header("Frame")
    col_prev, col_next = st.sidebar.columns(2)
    if col_prev.button("< Prev", width="stretch", disabled=st.session_state.frame_idx == 0):
        st.session_state.frame_idx -= 1
    if col_next.button("Next >", width="stretch", disabled=st.session_state.frame_idx >= len(seq) - 1):
        st.session_state.frame_idx += 1
    st.session_state.frame_idx = st.sidebar.slider(
        "Frame index", 0, len(seq) - 1, st.session_state.frame_idx
    )
    frame_id = seq.frame_ids[st.session_state.frame_idx]
    st.sidebar.caption(f"frame id: `{frame_id}`  ({st.session_state.frame_idx + 1} of {len(seq)})")

    with st.sidebar.expander("Dataset status"):
        report = validate(config_path, seq_id)
        if report.ok:
            st.success(f"{report.n_frames_ok}/{report.n_frames_checked} frames valid")
        else:
            st.warning(f"{report.n_frames_ok}/{report.n_frames_checked} frames valid — see errors below")
            for e in report.errors:
                st.caption(f"error: {e}")
        for w in report.warnings:
            st.caption(f"warning: {w}")

    st.sidebar.header("Point cloud display")
    max_points_3d = st.sidebar.number_input(
        "Max points in 3D view", min_value=1000, max_value=170_000,
        value=cfg.render.max_points_3d, step=1000,
    )
    shown_groups = st.sidebar.multiselect(
        "Show semantic groups (display only — does not affect grids)",
        list(GROUP_NAMES), default=list(GROUP_NAMES),
    )

    with st.sidebar.expander("Zone configuration (read-only — editable live in Milestone 6)"):
        for i, z in enumerate(cfg.grid.zones):
            st.caption(f"zone {i}: [{z.r_min:g}, {z.r_max:g}) m -> {z.cell_m:g} m cells")
    with st.sidebar.expander("Preprocessing filters (read-only — editable live in Milestone 6)"):
        st.caption(f"range: [{cfg.preprocess.min_range_m:g}, {cfg.preprocess.max_range_m:g}) m")
        st.caption(f"height: [{cfg.preprocess.z_min_m:g}, {cfg.preprocess.z_max_m:g}] m")
        st.caption(f"dropped classes: {list(cfg.preprocess.drop_classes)}")
        st.caption(f"devices kept: {list(cfg.preprocess.devices)}")

    # --- load & grid this frame --------------------------------------------
    try:
        cfg, frame, pre, (uniform_table, uniform_rows), (adaptive_table, adaptive_rows) = get_frame_bundle(
            config_path, seq_id, frame_id
        )
    except Exception as exc:  # noqa: BLE001 - surfacing any loader failure to the UI
        st.error(f"Could not load or grid frame '{frame_id}': {exc}")
        st.stop()

    classes = load_classes(seq.classes_path)
    uncovered = uncovered_point_count(pre, cfg.grid.zones)

    tab_cloud, tab_maps, tab_compare, tab_metrics = st.tabs(
        ["3D Point Cloud", "2.5D Maps", "Comparison", "Metrics"]
    )

    # --- tab: 3D point clouds -----------------------------------------------
    with tab_cloud:
        group_lookup = build_group_lookup()
        mask = groups_mask(pre.sem, shown_groups, group_lookup) if shown_groups else np.zeros(pre.n_kept, dtype=bool)
        idx = np.flatnonzero(mask)
        idx = idx[panels.subsample_indices(len(idx), max_points_3d)]
        st.caption(
            f"{pre.n_kept:,} points after preprocessing; showing {len(idx):,} "
            f"after the group filter and the {max_points_3d:,}-point display cap."
        )
        c1, c2 = st.columns(2)
        with c1:
            fig = panels.point_cloud_figure(
                pre.x[idx], pre.y[idx], pre.z[idx], pre.z[idx],
                colorscale="Viridis", is_rgb=False, colorbar_title="z (m)",
                title="Raw point cloud (coloured by elevation)",
            )
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = panels.semantic_point_cloud_figure(
                pre.x[idx], pre.y[idx], pre.z[idx], pre.sem[idx], classes,
                title="Semantic point cloud",
            )
            st.plotly_chart(fig, width="stretch")
        st.subheader("Classes present in this frame")
        st.dataframe(panels.class_breakdown_table(pre.sem, classes), width="stretch", hide_index=True)

    # --- tab: 2.5D maps -------------------------------------------------------
    with tab_maps:
        extent = cfg.render.extent_m
        res = cfg.render.display_res_m
        z_all = np.concatenate([uniform_table.z_ref, adaptive_table.z_ref]) if len(uniform_table) and len(adaptive_table) else np.array([0.0])
        zmin, zmax = float(np.nanmin(z_all)), float(np.nanmax(z_all))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Uniform grid** — {cfg.grid.uniform_cell_m:g} m cells, {len(uniform_table):,} cells")
            r = rasterize_scalar(uniform_table, uniform_table.z_ref, extent, res)
            st.plotly_chart(
                panels.raster_scalar_figure(r, "Viridis", "Uniform — elevation", "z_ref (m)", zmin, zmax),
                width="stretch",
            )
        with c2:
            st.markdown(f"**Adaptive grid** — {len(cfg.grid.zones)} zones, {len(adaptive_table):,} cells")
            r = rasterize_scalar(adaptive_table, adaptive_table.z_ref, extent, res)
            st.plotly_chart(
                panels.raster_scalar_figure(r, "Viridis", "Adaptive — elevation", "z_ref (m)", zmin, zmax),
                width="stretch",
            )

        c3, c4 = st.columns(2)
        with c3:
            r = rasterize_semantic(uniform_table, extent, res)
            st.plotly_chart(panels.raster_rgb_figure(r, "Uniform — semantic"), width="stretch")
        with c4:
            r = rasterize_semantic(adaptive_table, extent, res)
            st.plotly_chart(panels.raster_rgb_figure(r, "Adaptive — semantic"), width="stretch")
            rz = rasterize_zone(adaptive_table, extent, res)
            st.plotly_chart(
                panels.raster_zone_figure(rz, len(cfg.grid.zones), "Adaptive — zone / cell-size map"),
                width="stretch",
            )
        if uncovered:
            st.warning(f"{uncovered:,} points fall outside every configured zone and are absent from the adaptive grid.")

    # --- tab: comparison --------------------------------------------------
    with tab_compare:
        st.subheader("Uniform vs. adaptive, same frame")
        cell_counts = {"uniform_fine": len(uniform_table), "adaptive": len(adaptive_table)}
        st.plotly_chart(
            panels.cost_comparison_figure(list(cell_counts), list(cell_counts.values())),
            width="stretch",
        )
        reduction = 1 - len(adaptive_table) / len(uniform_table) if len(uniform_table) else float("nan")
        col1, col2, col3 = st.columns(3)
        col1.metric("Uniform cells", f"{len(uniform_table):,}")
        col2.metric(
            "Adaptive cells", f"{len(adaptive_table):,}",
            delta=f"-{reduction*100:.1f}% vs uniform", delta_color="inverse",
        )
        col3.metric("Table memory (adaptive vs uniform)", f"{adaptive_table.nbytes/1024:.0f} / {uniform_table.nbytes/1024:.0f} KiB")

        near_r = cfg.grid.zones[0].r_max
        near_u = int((np.hypot(uniform_table.cx, uniform_table.cy) < near_r).sum())
        near_a = int((np.hypot(adaptive_table.cx, adaptive_table.cy) < near_r).sum())
        st.caption(
            f"Near-field (< {near_r:g} m) cell count: uniform {near_u:,} vs adaptive {near_a:,} "
            "— should match closely, since both use the same cell size there."
        )

    # --- tab: metrics -------------------------------------------------------
    with tab_metrics:
        st.subheader("This frame")
        run_matched = st.checkbox(
            "Also compute uniform_coarse and uniform_matched (a few hundred ms extra)",
            value=False,
        )
        methods = {"uniform_fine": (uniform_table, uniform_rows), "adaptive": (adaptive_table, adaptive_rows)}
        if run_matched:
            t0 = time.perf_counter()
            (coarse, coarse_rows), (matched, matched_rows), matched_size = get_extra_baselines(
                config_path, seq_id, frame_id, len(adaptive_table), 12
            )
            methods["uniform_coarse"] = (coarse, coarse_rows)
            methods["uniform_matched"] = (matched, matched_rows)
            st.caption(f"uniform_matched found cell size {matched_size:.4f} m in {time.perf_counter()-t0:.2f} s")

        cost_rows = []
        band_rmse: dict[str, list[float]] = {}
        band_agree: dict[str, list[float]] = {}
        band_labels = [f"{lo:g}-{hi:g}m" for lo, hi in zip(cfg.benchmark.bands_m, cfg.benchmark.bands_m[1:])]
        for name, (table, rows) in methods.items():
            cell_size = float(table.size[0]) if len(table) else cfg.grid.uniform_cell_m
            cost_rows.append({
                "method": name,
                "cells": len(table),
                "table_KiB": round(table.nbytes / 1024, 1),
                "dense_equivalent_cells": dense_equivalent_cells(cfg.preprocess.max_range_m, cell_size),
            })
            bq = quality_by_band(pre, table, rows, cfg.benchmark.bands_m)
            band_rmse[name] = [b.elevation_rmse_m for b in bq]
            band_agree[name] = [b.semantic_agreement for b in bq]

        st.dataframe(cost_rows, width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                panels.band_quality_figure(band_labels, band_rmse, "RMSE (m)", "Elevation RMSE by distance band"),
                width="stretch",
            )
        with c2:
            st.plotly_chart(
                panels.band_quality_figure(band_labels, band_agree, "agreement", "Semantic agreement by distance band"),
                width="stretch",
            )
        st.caption(
            "Quality is measured directly against this frame's own points — no synthetic "
            "ground truth. Bands match the zone ladder exactly."
        )

        st.subheader("Full 80-frame benchmark (reference)")
        summary = load_summary_json("results/benchmark_summary.json")
        if summary is None:
            st.info(
                "No results/benchmark_summary.json found. Run `python scripts/run_benchmark.py` "
                "to produce one."
            )
        else:
            st.caption(f"From {summary['n_frames']} frames. render_fps: {summary['render_fps']}")
            rows = [
                {
                    "method": m,
                    "mean_cells": round(v["mean_cells"]),
                    "reduction_vs_uniform_fine_pct": round(v["cell_reduction_vs_uniform_fine_pct"], 1),
                    "mean_build_time_ms": round(v["mean_build_time_ms"], 1),
                }
                for m, v in summary["methods"].items()
            ]
            st.dataframe(rows, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
