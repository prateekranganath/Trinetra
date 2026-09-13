"""Streamlit dashboard: browse the sequence, edit zones and filters live, and
compare uniform vs. adaptive maps.

Run with:
    streamlit run app/dashboard.py

This module only wires Streamlit widgets to the `avrmap` library and to
`panels.py`'s figure builders — it loads no pickles and builds no grids
itself. Every heavy call is cached (`st.cache_resource` for dataset discovery
and validation, `st.cache_data` for the per-frame load/preprocess/grid
bundle) so moving a slider re-renders instantly instead of re-reading pkl
files.

Milestone 6 additions over the static Milestone 5 dashboard: play/pause
autoplay with an honestly measured UI rate (not an assumed target), a live
zone-ladder editor validated by the same `avrmap.config.validate_zones` the
config loader uses, and preprocessing filter controls that actually rebuild
the grids rather than only filtering the 3D display.
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import panels  # noqa: E402
from avrmap.config import (  # noqa: E402
    ConfigError,
    PreprocessConfig,
    ZoneConfig,
    default_config_path,
    load_config,
    validate_zones,
)
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
from avrmap.semantics import (  # noqa: E402
    GROUP_NAMES,
    UNLABELED_CLASS,
    build_group_lookup,
    class_name,
    groups_mask,
)

st.set_page_config(page_title="Trinetra — Adaptive LiDAR Mapping", layout="wide")

DEFAULT_PLAYBACK_HZ = 3.0
FPS_WINDOW = 20  # reruns averaged for the measured UI rate


# --------------------------------------------------------------------------
# Cached data access. Only primitive (hashable, cheap-to-hash) arguments are
# passed into cached functions — config, sequence, and preprocess/grid
# settings are always reconstructed from primitives inside, rather than
# risking Streamlit's hashing of custom dataclasses, and so that every
# edit made in the sidebar (a filter, a zone) is naturally part of the cache
# key without any extra bookkeeping.
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


@st.cache_data(show_spinner="Loading, preprocessing, and gridding this frame...")
def get_frame_bundle(
    config_path: str,
    seq_id: str,
    frame_id: str,
    min_range_m: float,
    max_range_m: float,
    z_min_m: float,
    z_max_m: float,
    drop_classes: tuple[int, ...],
    devices: tuple[int, ...],
    zones: tuple[tuple[float, float, float], ...],
    min_points_per_cell: int,
    elevation_stat: str,
):
    cfg = load_config(config_path)
    seq = require_sequence(cfg.dataset.root, cfg.dataset.max_search_depth, seq_id=seq_id)
    frame = load_frame(seq, frame_id)
    pp = PreprocessConfig(
        min_range_m=min_range_m, max_range_m=max_range_m,
        z_min_m=z_min_m, z_max_m=z_max_m,
        drop_classes=drop_classes, devices=devices,
    )
    pre = preprocess_frame(frame, pp)
    zone_objs = tuple(ZoneConfig(r_min=r0, r_max=r1, cell_m=c) for r0, r1, c in zones)
    uniform_cell_m = min(c for _, _, c in zones)  # uniform_fine tracks the finest edited zone
    uniform = build_uniform_map_with_point_cells(pre, uniform_cell_m, min_points_per_cell, elevation_stat)
    adaptive = build_adaptive_map_with_point_cells(pre, zone_objs, min_points_per_cell, elevation_stat)
    return frame, pre, uniform, adaptive


@st.cache_data(show_spinner="Building uniform_coarse and searching for uniform_matched...")
def get_extra_baselines(
    config_path: str,
    seq_id: str,
    frame_id: str,
    min_range_m: float,
    max_range_m: float,
    z_min_m: float,
    z_max_m: float,
    drop_classes: tuple[int, ...],
    devices: tuple[int, ...],
    coarsest_cell_m: float,
    target_cells: int,
    min_points_per_cell: int,
    elevation_stat: str,
    matched_iterations: int,
):
    cfg = load_config(config_path)
    seq = require_sequence(cfg.dataset.root, cfg.dataset.max_search_depth, seq_id=seq_id)
    frame = load_frame(seq, frame_id)
    pp = PreprocessConfig(
        min_range_m=min_range_m, max_range_m=max_range_m,
        z_min_m=z_min_m, z_max_m=z_max_m,
        drop_classes=drop_classes, devices=devices,
    )
    pre = preprocess_frame(frame, pp)
    coarse = build_uniform_map_with_point_cells(pre, coarsest_cell_m, min_points_per_cell, elevation_stat)
    matched_size, _ = find_matching_uniform_cell_size(
        pre, target_cells, min_points_per_cell, elevation_stat, iterations=matched_iterations,
    )
    matched = build_uniform_map_with_point_cells(pre, matched_size, min_points_per_cell, elevation_stat)
    return coarse, matched, matched_size


@st.cache_data(show_spinner=False)
def load_summary_json(path: str):
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Small pure functions, pulled out of main() so they can be unit-tested with
# ordinary pytest. streamlit.testing.v1.AppTest cannot safely exercise the
# actual autoplay loop below: st.rerun() inside a playing=True branch causes
# AppTest to unroll every subsequent script pass synchronously within one
# .run() call, with no way for a test to interject a Pause click mid-loop the
# way a real browser session can. See tests/test_dashboard_logic.py and the
# Dashboard section of README.md for how playback is verified instead.
# --------------------------------------------------------------------------


def next_playback_index(current: int, n_frames: int) -> int:
    """The next frame index during autoplay: advance by one, then loop back
    to the start after the last frame."""
    return 0 if current >= n_frames - 1 else current + 1


# --------------------------------------------------------------------------
# Sidebar sections that hold real logic (zone editing needs validation, not
# just widget wiring) live in their own functions for readability.
# --------------------------------------------------------------------------


def zone_editor(default_zones: tuple[ZoneConfig, ...], max_range_m: float):
    """Per-zone r_max / cell_m editor. r_min is derived, so it stays contiguous
    by construction; only the ordering and sizing rules can be violated, and
    those are checked with the exact function `load_config` itself uses.

    Returns ``(zones, error)``: on an invalid edit, ``zones`` is the last
    known-good ladder and ``error`` explains what was rejected, so the grids
    below keep rendering with a valid configuration instead of crashing.
    """
    st.sidebar.caption(
        "Editing r_max and cell size per zone. Rules: contiguous, cell size "
        "non-decreasing with distance, and each step an integer multiple of "
        "the previous cell size (so coarse cells nest onto fine ones)."
    )
    edited = []
    r_min = 0.0
    for i, default_zone in enumerate(default_zones):
        col_a, col_b = st.sidebar.columns(2)
        r_max = col_a.number_input(
            f"zone {i} r_max", min_value=r_min + 0.1, max_value=float(max_range_m),
            value=float(default_zone.r_max), step=1.0, key=f"zone_{i}_r_max",
            format="%.1f",
        )
        cell_m = col_b.number_input(
            f"zone {i} cell (m)", min_value=0.01, max_value=10.0,
            value=float(default_zone.cell_m), step=0.05, format="%.2f",
            key=f"zone_{i}_cell_m",
        )
        edited.append(ZoneConfig(r_min=r_min, r_max=r_max, cell_m=cell_m))
        r_min = r_max

    try:
        validate_zones(tuple(edited))
    except ConfigError as exc:
        return st.session_state.last_valid_zones, str(exc)
    st.session_state.last_valid_zones = tuple(edited)
    return tuple(edited), None


def filter_editor(defaults: PreprocessConfig, classes: dict) -> PreprocessConfig:
    """Range, height, class and device filters that rebuild the grids —
    unlike the display-only semantic-group filter in the point-cloud tab."""
    min_r, max_r = st.sidebar.slider(
        "Range crop (m)", 0.0, 150.0,
        (float(defaults.min_range_m), float(defaults.max_range_m)),
        step=1.0, key="flt_range",
    )
    z_min, z_max = st.sidebar.slider(
        "Height crop (m)", -20.0, 40.0,
        (float(defaults.z_min_m), float(defaults.z_max_m)),
        step=0.5, key="flt_height",
    )
    class_ids = sorted(set(classes) | {UNLABELED_CLASS})
    drop = st.sidebar.multiselect(
        "Drop classes", options=class_ids, default=list(defaults.drop_classes),
        format_func=lambda c: f"{c}: {class_name(classes, c)}", key="flt_drop_classes",
    )
    st.sidebar.caption("Empty means drop nothing.")
    devices = st.sidebar.multiselect(
        "Devices kept", options=[0, 1], default=list(defaults.devices), key="flt_devices",
    )
    st.sidebar.caption("Empty means keep every sensor.")
    return PreprocessConfig(
        min_range_m=min_r, max_range_m=max_r, z_min_m=z_min, z_max_m=z_max,
        drop_classes=tuple(drop), devices=tuple(devices),
    )


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
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "playback_hz" not in st.session_state:
        st.session_state.playback_hz = DEFAULT_PLAYBACK_HZ
    if "rerun_ts" not in st.session_state:
        st.session_state.rerun_ts = deque(maxlen=FPS_WINDOW)
    if "last_valid_zones" not in st.session_state:
        st.session_state.last_valid_zones = cfg.grid.zones
    st.session_state.frame_idx = min(st.session_state.frame_idx, len(seq) - 1)

    st.sidebar.header("Frame")
    col_prev, col_next = st.sidebar.columns(2)
    # on_click callbacks run before the script reruns from the top, so the
    # disabled= conditions below see the post-click state, not the state from
    # before this click. Checking a button's return value inline instead
    # (`if col_prev.button(...):`) would leave the *other* button showing a
    # stale, one-click-behind disabled state — genuinely unclickable in a
    # real browser for that one render, not just a testing artifact.
    col_prev.button(
        "< Prev", width="stretch", disabled=st.session_state.frame_idx == 0,
        on_click=lambda: st.session_state.__setitem__(
            "frame_idx", max(0, st.session_state.frame_idx - 1)
        ),
    )
    col_next.button(
        "Next >", width="stretch", disabled=st.session_state.frame_idx >= len(seq) - 1,
        on_click=lambda: st.session_state.__setitem__(
            "frame_idx", min(len(seq) - 1, st.session_state.frame_idx + 1)
        ),
    )
    st.session_state.frame_idx = st.sidebar.slider(
        "Frame index", 0, len(seq) - 1, st.session_state.frame_idx
    )
    frame_id = seq.frame_ids[st.session_state.frame_idx]
    st.sidebar.caption(f"frame id: `{frame_id}`  ({st.session_state.frame_idx + 1} of {len(seq)})")

    st.sidebar.header("Playback")
    play_col, pause_col = st.sidebar.columns(2)
    play_col.button(
        "Play", width="stretch", disabled=st.session_state.playing,
        on_click=lambda: st.session_state.__setitem__("playing", True),
    )
    pause_col.button(
        "Pause", width="stretch", disabled=not st.session_state.playing,
        on_click=lambda: st.session_state.__setitem__("playing", False),
    )
    st.session_state.playback_hz = st.sidebar.slider(
        "Target rate (Hz)", 1.0, 10.0, st.session_state.playback_hz, step=0.5,
        help="What autoplay aims for. The measured rate below is what actually happens.",
    )
    if len(st.session_state.rerun_ts) >= 2:
        measured_fps = 1.0 / (sum(st.session_state.rerun_ts) / len(st.session_state.rerun_ts))
        st.sidebar.metric(
            "Measured UI rate", f"{measured_fps:.1f} fps",
            help=(
                f"Actual wall-clock time between the last "
                f"{len(st.session_state.rerun_ts)} reruns — includes load, "
                "preprocessing, gridding, and rendering every tab. Not an "
                "assumed target."
            ),
        )
    else:
        st.sidebar.caption("Measured UI rate: step or play a couple more frames.")

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
        "Show semantic groups (display only — does not rebuild grids)",
        list(GROUP_NAMES), default=list(GROUP_NAMES),
    )

    classes = load_classes(seq.classes_path)

    if st.sidebar.button("Reset zones and filters to config defaults"):
        for i in range(len(cfg.grid.zones)):
            st.session_state.pop(f"zone_{i}_r_max", None)
            st.session_state.pop(f"zone_{i}_cell_m", None)
        for key in ("flt_range", "flt_height", "flt_drop_classes", "flt_devices"):
            st.session_state.pop(key, None)
        st.session_state.last_valid_zones = cfg.grid.zones
        st.rerun()

    with st.sidebar.expander("Zone ladder (editable)", expanded=False):
        zones, zone_error = zone_editor(cfg.grid.zones, cfg.preprocess.max_range_m)
    if zone_error:
        st.sidebar.error(f"Invalid zone edit, using the last valid ladder: {zone_error}")

    with st.sidebar.expander("Preprocessing filters (editable)", expanded=False):
        preprocess_cfg = filter_editor(cfg.preprocess, classes)

    # --- load & grid this frame --------------------------------------------
    try:
        frame, pre, (uniform_table, uniform_rows), (adaptive_table, adaptive_rows) = get_frame_bundle(
            config_path, seq_id, frame_id,
            preprocess_cfg.min_range_m, preprocess_cfg.max_range_m,
            preprocess_cfg.z_min_m, preprocess_cfg.z_max_m,
            preprocess_cfg.drop_classes, preprocess_cfg.devices,
            tuple((z.r_min, z.r_max, z.cell_m) for z in zones),
            cfg.grid.min_points_per_cell, cfg.grid.elevation_stat,
        )
    except Exception as exc:  # noqa: BLE001 - surfacing any loader failure to the UI
        st.error(f"Could not load or grid frame '{frame_id}': {exc}")
        st.stop()

    if pre.n_kept == 0:
        st.warning(
            "The current filters leave zero points for this frame. Widen the "
            "range/height crop or drop fewer classes."
        )

    uncovered = uncovered_point_count(pre, zones)
    uniform_cell_m = min(z.cell_m for z in zones)

    tab_cloud, tab_maps, tab_compare, tab_metrics = st.tabs(
        ["3D Point Cloud", "2.5D Maps", "Comparison", "Metrics"]
    )

    # --- tab: 3D point clouds -----------------------------------------------
    with tab_cloud:
        group_lookup = build_group_lookup()
        mask = (
            groups_mask(pre.sem, shown_groups, group_lookup)
            if shown_groups else np.zeros(pre.n_kept, dtype=bool)
        )
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
        if len(uniform_table) and len(adaptive_table):
            z_all = np.concatenate([uniform_table.z_ref, adaptive_table.z_ref])
            zmin, zmax = float(np.nanmin(z_all)), float(np.nanmax(z_all))
        else:
            zmin, zmax = 0.0, 1.0

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Uniform grid** — {uniform_cell_m:g} m cells, {len(uniform_table):,} cells")
            r = rasterize_scalar(uniform_table, uniform_table.z_ref, extent, res)
            st.plotly_chart(
                panels.raster_scalar_figure(r, "Viridis", "Uniform — elevation", "z_ref (m)", zmin, zmax),
                width="stretch",
            )
        with c2:
            st.markdown(f"**Adaptive grid** — {len(zones)} zones, {len(adaptive_table):,} cells")
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
                panels.raster_zone_figure(rz, len(zones), "Adaptive — zone / cell-size map"),
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
        col3.metric(
            "Table memory (adaptive vs uniform)",
            f"{adaptive_table.nbytes/1024:.0f} / {uniform_table.nbytes/1024:.0f} KiB",
        )

        near_r = zones[0].r_max
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
        if run_matched and len(adaptive_table):
            t0 = time.perf_counter()
            (coarse, coarse_rows), (matched, matched_rows), matched_size = get_extra_baselines(
                config_path, seq_id, frame_id,
                preprocess_cfg.min_range_m, preprocess_cfg.max_range_m,
                preprocess_cfg.z_min_m, preprocess_cfg.z_max_m,
                preprocess_cfg.drop_classes, preprocess_cfg.devices,
                max(z.cell_m for z in zones), len(adaptive_table),
                cfg.grid.min_points_per_cell, cfg.grid.elevation_stat, 12,
            )
            methods["uniform_coarse"] = (coarse, coarse_rows)
            methods["uniform_matched"] = (matched, matched_rows)
            st.caption(f"uniform_matched found cell size {matched_size:.4f} m in {time.perf_counter()-t0:.2f} s")

        cost_rows = []
        band_rmse: dict[str, list[float]] = {}
        band_agree: dict[str, list[float]] = {}
        band_labels = [f"{lo:g}-{hi:g}m" for lo, hi in zip(cfg.benchmark.bands_m, cfg.benchmark.bands_m[1:])]
        for name, (table, rows) in methods.items():
            cell_size = float(table.size[0]) if len(table) else uniform_cell_m
            cost_rows.append({
                "method": name,
                "cells": len(table),
                "table_KiB": round(table.nbytes / 1024, 1),
                "dense_equivalent_cells": dense_equivalent_cells(preprocess_cfg.max_range_m, cell_size),
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
            "ground truth. Bands match the default zone ladder's edges, not any live edit."
        )

        st.subheader("Full 80-frame benchmark (reference, default config)")
        summary = load_summary_json("results/benchmark_summary.json")
        if summary is None:
            st.info(
                "No results/benchmark_summary.json found. Run `python scripts/run_benchmark.py` "
                "to produce one."
            )
        else:
            st.caption(
                f"From {summary['n_frames']} frames at the shipped default config — not this "
                "session's live edits. render_fps in that file is reported as unavailable, "
                "since the CLI benchmark renders nothing; the sidebar's measured UI rate above "
                "is this dashboard's own, real figure."
            )
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

    # --- playback: measure the real rerun interval, then advance if playing -
    now = time.perf_counter()
    if st.session_state.get("_last_rerun_ts") is not None:
        st.session_state.rerun_ts.append(now - st.session_state._last_rerun_ts)
    st.session_state._last_rerun_ts = now

    if st.session_state.playing:
        st.session_state.frame_idx = next_playback_index(st.session_state.frame_idx, len(seq))
        target_dt = 1.0 / st.session_state.playback_hz
        already_spent = time.perf_counter() - now
        time.sleep(max(0.0, target_dt - already_spent))
        st.rerun()


if __name__ == "__main__":
    main()
