# Trinetra

Adaptive variable-resolution 2.5D LiDAR mapping for dynamic environment perception.
Repository: https://github.com/prateekranganath/Trinetra

Turning sequential PandaSet LiDAR sweeps into a 2.5D elevation-and-semantic map on a
**foveated grid**: fine cells close to the ego vehicle, progressively coarser cells further
out. The point is to keep near-field detail while spending far fewer cells than a uniform
fine grid, and to prove that with measurements rather than assertions.

Every number in this file was produced by running the code in this repository. Nothing is
estimated.

---

## Status

| Milestone | State |
| --- | --- |
| 1. Data foundation and validation | **done** |
| 2. Preprocessing and uniform grid | **done** |
| 3. Adaptive grid | **done** |
| 4. Benchmark harness | **done** |
| 5. Dashboard | **done** |
| 6. Playback and live controls | **done** |
| 7. Multi-frame accumulation and demo | not started |

---

## Setup

Python 3.13 on Windows.

```bash
git clone https://github.com/prateekranganath/Trinetra.git
cd Trinetra
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The PandaSet sequence itself is not tracked in this repository (see `.gitignore`); drop it
into a top-level `Data/` folder, or anywhere findable by structure per the section below.

Then validate the dataset and run the tests:

```bash
python scripts/validate_dataset.py          # checks all 80 frames, about 1.5 s
python scripts/validate_dataset.py --quick  # first, middle and last frame only
python scripts/build_uniform_map.py         # preprocess + grid one frame, print a summary
python scripts/compare_grids.py             # uniform vs. adaptive on one frame
python scripts/run_benchmark.py             # full benchmark, all 80 frames, a few minutes
streamlit run app/dashboard.py              # dashboard, opens at localhost:8501
python -m pytest -q                         # full suite, about 30 s
python -m pytest -q -m "not slow"           # skip tests that read the real data, about 1 s
```

`streamlit` pins `pandas<3` in every release before 1.63.0, so `requirements.txt` pins
streamlit to 1.63.0 in order to keep pandas 3.0.1. Installing an older streamlit will fail
to resolve.

---

## The data

One PandaSet sequence lives in `Data/`. Discovery locates sequences by **structure, not by
name**: any directory containing both `lidar/*.pkl` and `annotations/semseg/*.pkl` counts as
one. So the sequence folder can be renamed, or moved under `data/` or `data/001/`, without
touching any code or config.

```
Data/
  lidar/            00.pkl .. 79.pkl, poses.json, timestamps.json
  annotations/
    semseg/         00.pkl .. 79.pkl, classes.json
    cuboids/        00.pkl .. 79.pkl        (present, unused in the MVP)
  camera/           6 cameras x 80 jpg      (present, unused in the MVP)
  meta/             gps.json, timestamps.json
```

What validation reports, read from the files at run time:

| Property | Value |
| --- | --- |
| Paired frames | 80 of 80, matching row counts and matching indices |
| Total points | 13,718,418 |
| Points per frame | min 163,176, mean 171,480, max 176,747 |
| Frame rate | 0.100 s interval, 10.0 Hz, 8.0 s of driving |
| Ego path length | 84.6 m |
| Semantic classes declared | 42 |
| Semantic classes present | 23 |

Each LiDAR pickle is a pandas DataFrame with columns `x, y, z, i, t, d`. Column `d` is the
sensor id: 0 is the 360-degree spinning LiDAR, 1 is the forward-facing one. The matching
semseg pickle has a single `class` column with one row per LiDAR point.

### Two data quirks worth knowing

**Unlabelled points.** Five points, all in frame 21, carry class `0`, which `classes.json`
does not declare. PandaSet uses it for unlabelled returns. Validation reports this as a
warning rather than an error, and preprocessing drops class 0 by default along with classes
1 to 4, which are Smoke, Exhaust, Spray or rain, and Reflection. Those are sensor artefacts
rather than surfaces.

**Range.** Returns reach past 300 m, but 98.8% of points in frame 00 fall within 100 m. The
default range limit is 100 m, which discards about 1.2% of returns. It is configurable.

---

## Coordinate frames

Three frames matter, and getting them wrong is the easiest way to produce a plausible-looking
but meaningless elevation map.

**World.** The frame the pickles are already stored in, with its origin at the frame-0 ego
position. It is gravity-aligned: a plane fitted to road-class points within 60 m is tilted
**0.30 degrees**.

**Map.** World axes translated so the origin follows the ego vehicle each frame. No rotation.
This is the frame the 2.5D grids are built in.

**Ego.** The vehicle body frame, x forward. The same road-plane fit tilts by **3.06 degrees**
here, because the body is pitched and rolled relative to the road.

The map frame uses world axes precisely because of that 3-degree difference. Elevation is only
meaningful about gravity. And since the distance zones are radial circles, they need only the
ego *position*, so the heading never enters the core pipeline at all.

### The heading convention

This was determined empirically, not assumed. Reading the stored quaternion as `(w, x, y, z)`
into a standard rotation matrix `R`, the transform that puts the vehicle's forward direction
on ego `+x` is `R @ (p - T)` and **not** its transpose. In other words the stored quaternion
is the world-to-ego rotation, the inverse of the more common convention.

Two independent physical facts confirm it, and both are printed by the validation CLI:

```
forward sensor azimuth  -4.2 deg from +x  [PASS: expected near 0]
ego motion to next      [+0.783, -0.011, -0.021] m  [PASS: +x should dominate]
```

The forward-facing sensor points where the car is going, and the car drives forwards. Under
the transpose, both land on `+y` instead. `tests/test_geometry.py` asserts both against the
real data, so the convention cannot silently regress. The synthetic pose fixture in that file
deliberately negates its yaw for the same reason, and says so.

---

## Design decisions

**Sparse cell tables, not dense arrays.** A dense 0.1 m grid over 200 x 200 m is 4 million
cells, of which roughly 1.6% are occupied, and it cannot represent mixed resolutions at all.
Both grid types share one representation, `CellTable` (`avrmap/grids/common.py`): parallel
NumPy arrays with one row per occupied cell. Same fields, same dtypes, only the row count
differs, which is what makes the memory comparison in the eventual benchmark honest.

**No database.** Eighty immutable frames, read in about 10 ms each when warm. There is nothing
to index and no query workload. Benchmark output goes to CSV and JSON under `results/`, which
is readable and diffable. SQLite would add schema work and buy no capability.

**No Open3D.** Its wheels are unreliable on Python 3.13 and it cannot embed in a browser
dashboard. Visualization will be Plotly for the 3D cloud plus NumPy rasterization for the maps.

**Zone ladder.** For a spinning LiDAR at fixed angular resolution, the spacing between returns
grows roughly linearly with range, so cell area should grow with range to keep points-per-cell
roughly constant. The default ladder follows `cell_size ~ range / 100`, quantized so each
coarser size is an exact multiple of the finer one, which means zone boundaries fall on cell
edges and no cell straddles two zones. The defaults double at each step; the config validator
enforces the weaker requirement that the ratio be a whole number.

Measured on frame 00 during planning:

| Zone | Range | Cell size | Points | Cells | Points per cell |
| --- | --- | --- | --- | --- | --- |
| 0 | 0-10 m | 0.10 m | 18,094 | 3,405 | 5.3 |
| 1 | 10-30 m | 0.20 m | 88,451 | 16,300 | 5.4 |
| 2 | 30-60 m | 0.40 m | 52,899 | 6,664 | 7.9 |
| 3 | 60-100 m | 0.80 m | 7,669 | 1,031 | 7.4 |

Points per cell stays between 5.3 and 7.9 across a tenfold range span, which supports the
rationale. **These are defensible defaults, not a validated optimum.** They live in
`configs/default.yaml` and are meant to be changed.

The same prototype measured 27,400 adaptive cells against 63,749 for a uniform 0.1 m grid on
the same points, a 57% reduction at identical near-field resolution. Milestone 3 reproduces
that through the full pipeline with default filtering applied, and across every frame — see
below.

---

## Preprocessing and the uniform grid

`avrmap/preprocess.py` crops and filters one frame's points in the map frame: range,
elevation, dropped classes, dropped devices. The four per-criterion drop counts are
independent, not a partition — a point failing two filters is counted under both — so only
`n_kept` and the arrays themselves reflect the actual combined result.

`avrmap/grids/common.py` holds `aggregate_cells`, the single kernel both grid types call. It
takes already-filtered points at one cell size and returns a `CellTable`. There is no Python
loop over points anywhere in it:

- **Cell keying.** Points are binned by `floor(coord / cell_size)`, then packed into one int64
  key via row-major indexing over the *actual* bounding box of occupied cells (`(ix - min_ix)
  * span_y + (iy - min_iy)`). This needs no assumption about how far the data ranges and cannot
  collide, unlike a fixed-stride key sized for a guessed extent.
- **Elevation.** One sort orders points by `(cell, z)`. Min, max and mean fall out of segment
  boundaries and `np.add.reduceat`. `z_ref` defaults to a nearest-rank 95th percentile computed
  from the same sort, which barely moves for a single spurious high return where a max would
  jump to it entirely — asserted directly in `test_grids.py`.
- **Semantic majority vote.** Points are grouped by the combined key `cell * 256 + class`,
  giving per-cell per-class counts from one more `np.unique`. A monotone score,
  `count * 256 - class_id`, is reduced per cell with `np.maximum.reduceat`: a strictly higher
  count always wins, and equal counts resolve to the lower class id, deterministically.

`avrmap/grids/uniform.py` calls this kernel once, with `zone_id=0` and one cell size for the
whole range. `avrmap/grids/adaptive.py` (below) calls it once per zone and concatenates the
results with `CellTable.concat`.

Measured on frame 00 through the full pipeline, default config (range crop, class and device
filters, elevation crop all applied):

| Cell size | Cells | Notes |
| --- | --- | --- |
| 0.1 m | 63,611 | matches the 63,749 measured during planning; the gap is from now also dropping the noise classes present in this frame |
| 0.2 m | 33,946 | |
| 0.4 m | 15,921 | |
| 0.8 m | 6,573 | |

`python scripts/build_uniform_map.py` reproduces this, plus preprocessing counts, timing,
z-range, mean confidence and a per-class cell breakdown, for any frame.

---

## The adaptive grid

`avrmap/grids/adaptive.py` builds the foveated grid. Because the map frame only translates to
the ego position and never rotates (see Coordinate frames, above), a point's distance zone
depends on its planar radius alone — no heading, no per-zone geometry beyond a radius test.
`build_adaptive_map` loops over the configured zones, masks each zone's points by
`ZoneConfig.contains` (the half-open `[r_min, r_max)` test the config validator already
enforces to be gap-free and non-overlapping), calls `aggregate_cells` once per zone at that
zone's own cell size, and concatenates the per-zone `CellTable`s. `zone_id` on each output cell
is the zone's index, so results can always be traced back to the config.

A point can in principle fall outside every zone, if the zone ladder's outer edge is
configured narrower than `preprocess.max_range_m`. `uncovered_point_count` reports that count
explicitly rather than letting it disappear silently; for the shipped default config the two
match exactly, so it is always zero.

Measured on frame 00 through the full pipeline, default four-zone config, against a uniform
grid at the same 0.1 m finest resolution:

| Grid | Cells | Table memory |
| --- | --- | --- |
| `uniform_fine` (0.1 m everywhere) | 63,611 | 2112.1 KiB |
| adaptive (4 zones, 0.1-0.8 m) | 27,295 | 906.3 KiB |

That is a **57.1% cell reduction**, matching the 57% measured during planning, now produced by
the real pipeline with default filtering applied. Near-field resolution is preserved: within
the innermost 10 m zone, the adaptive grid has 3,348 cells against the uniform grid's 3,324 —
matching within rounding, since a circular radius cutoff clips each grid's square cells
slightly differently at the boundary. Point count is exactly conserved across zones (166,327
in, 166,327 aggregated), confirming no point is double-counted or silently lost at a zone
boundary.

The reduction holds on every one of the 80 frames, not just frame 00 — `test_grids.py` asserts
`len(adaptive) < len(uniform_fine)` across the full sequence, with the worst frame still at
less than half the uniform cell count. This is the central, falsifiable claim of the project,
checked directly rather than assumed from one frame.

`python scripts/compare_grids.py` reproduces the full comparison for any frame, including the
per-zone cell and point breakdown:

```
grid                             cells      time   mean pts/cell
uniform_fine (0.1 m)            63,611     59.3ms             2.6
uniform_coarse (0.8 m)           6,573     44.2ms            25.3
adaptive (4 zones)              27,295     51.2ms             6.1

adaptive vs uniform_fine     57.1% fewer cells   [measured]
adaptive vs uniform_fine    906.3 KiB vs 2112.1 KiB table memory   [measured]
```

This script is a one-frame sanity check, not the benchmark — see the next section.

---

## The benchmark harness

`scripts/run_benchmark.py` runs four methods per frame across the whole sequence and writes
`results/benchmark.csv` (one row per frame per method) and `results/benchmark_summary.json`
(the same data aggregated across frames). Every column is labelled measured or derived in
`avrmap/metrics.py`'s docstrings; nothing is filled in from an earlier run.

**Three uniform baselines**, not one, because a single baseline is easy to rig:

- `uniform_fine` — uniform at the adaptive grid's finest cell size (0.1 m). The primary *cost*
  baseline: adaptive gives identical near-field resolution, so any cell saving here is real.
- `uniform_matched` — a uniform cell size found by bisection (`find_matching_uniform_cell_size`
  in `avrmap/grids/uniform.py`) so its cell count lands close to the adaptive grid's for that
  frame. The primary *quality* baseline: at an equal cell budget, does adaptive spend its cells
  where they matter? Occupied-cell count is a non-increasing function of cell size for a fixed
  point cloud, which is what makes bisection valid here; the search runs up to 14 full grid
  builds per frame and reports the cell size and count it actually found, not an assumed exact
  match.
- `uniform_coarse` — uniform at the coarsest configured zone size (0.8 m). A floor reference.

**Quality is scored directly against the frame's own points**, so no synthetic ground truth is
needed. `avrmap/grids/common.py`'s `aggregate_cells` gained a `return_point_cells` option that
maps every input point to the row of the output table it landed in (or -1 if that cell was
dropped), entirely vectorized — a stable sort's inverse permutation, no Python loop over
points. `avrmap/metrics.py`'s `quality_by_band` then computes, per distance band:

- **Elevation RMSE** — root-mean-square `|point.z - cell.z_ref|` over matched points.
- **Semantic agreement** — the fraction of matched points whose own class equals their cell's
  majority class.

Bands are the same four edges as the zone ladder (`0, 10, 30, 60, 100` m), so the
"near-field" comparison is exact, not approximate.

### Results, full sequence, 80 frames

| Method | Mean cells | vs `uniform_fine` | Mean build time | Mean table memory |
| --- | --- | --- | --- | --- |
| `uniform_fine` (0.1 m) | 49,788 | — | 91.5 ms | 1,653 KiB |
| `uniform_matched` (~0.1-0.9 m, searched per frame) | 22,425 | 55.0% fewer | 83.3 ms | 745 KiB |
| `uniform_coarse` (0.8 m) | 5,408 | 89.1% fewer | 72.3 ms | 180 KiB |
| **adaptive** (4 zones) | **22,436** | **54.9% fewer** | 73.1 ms | 745 KiB |

`uniform_matched` and adaptive land within 11 cells of each other on average — the search does
its job — which makes the 0-10 m band comparison below an honest equal-budget test:

| Method | RMSE, 0-10 m | Semantic agreement, 0-10 m |
| --- | --- | --- |
| `uniform_fine` | 0.512 m | 0.981 |
| **adaptive** | **0.512 m** | **0.981** |
| `uniform_matched` | 0.582 m | 0.969 |
| `uniform_coarse` | 0.706 m | 0.938 |

This is the plan's central, falsifiable claim, now checked on the full sequence rather than one
frame: **adaptive matches `uniform_fine` almost exactly in the near band while using 55% fewer
cells overall, and clearly beats `uniform_matched` in that same band at essentially the same
cell budget.**

The honest trade-off shows up further out. Because adaptive tapers to 0.8 m cells beyond 60 m
while `uniform_matched` spreads its budget evenly at roughly 0.24 m everywhere, adaptive is
*less* accurate than `uniform_matched` in the 60-100 m band (RMSE 3.06 m vs 2.37 m, agreement
0.983 vs 0.995). Concentrating resolution near the ego is a deliberate choice for a driving
context where nearby geometry matters more, not a free win in every band — the zone ladder
trades distant accuracy for near-field detail and a smaller overall footprint, and the
benchmark reports that trade honestly rather than only in the band that flatters it.

**What's measured, derived, or unavailable**, per the plan's requirement to distinguish them:

| Metric | Status |
| --- | --- |
| Cell counts, build time, table memory | measured |
| Dense-equivalent cell count | derived (no dense array is ever built) |
| Cell reduction percentages | derived from two measured counts |
| Elevation RMSE, semantic agreement | measured, against the frame's own points |
| `build_only_fps_estimate` (e.g. adaptive: 13.7, `uniform_fine`: 10.9) | derived from build time alone; excludes frame load, preprocessing, and all rendering |
| Render / UI FPS | **unavailable from this CLI** — it renders nothing to time. The dashboard (below) reports its own measured UI rate from actual Streamlit rerun intervals during playback, which is a different, honestly-measured number, not filled in here |

Reproduce with `python scripts/run_benchmark.py` (a few minutes for all 80 frames), or
`--frames 10` / `--frames 00,10,20` for a quicker pass while iterating.

---

## The dashboard

`streamlit run app/dashboard.py` opens a browser-based dashboard at `localhost:8501`. It wires
Streamlit widgets directly to the `avrmap` library — it loads no pickles, filters no points, and
builds no grids itself; `app/panels.py` builds every Plotly figure from data the dashboard
already has, so neither file duplicates logic that belongs in `avrmap`.

**Sidebar.** Sequence selector (one sequence today; the dropdown works for more without any
code change, since discovery is already multi-sequence-capable). Frame selection by slider or
Prev/Next buttons. Play/Pause autoplay with a target rate and an honestly *measured* UI rate
(below). A dataset-status expander running the same validation from Milestone 1. Display
controls for the 3D point cap and which semantic groups to show. Live, validated editors for
the zone ladder and the preprocessing filters, both of which actually rebuild the grids rather
than only filtering the display — see below.

**Tabs.**

- **3D Point Cloud** — the raw cloud coloured by elevation, and the semantic cloud coloured by
  the PandaSet palette, side by side, both subsampled to the configured point cap with a fixed
  random seed so the same points show on every rerun. A class-breakdown table lists every
  semantic class present in the frame with its point count.
- **2.5D Maps** — uniform and adaptive elevation rasters on a shared colour scale, their
  semantic-class rasters, and a zone/cell-size raster that makes the fovea visible at a glance:
  each pixel is coloured by which distance zone (and therefore which cell size) produced it.
- **Comparison** — cell counts side by side, a live reduction percentage, table memory for both,
  and the near-field cell-count check (should match closely, since both grids use the same cell
  size inside the innermost zone).
- **Metrics** — cost and per-band quality for `uniform_fine` and adaptive, computed live for the
  selected frame via the same `avrmap.metrics` functions the benchmark script uses.
  `uniform_coarse` and `uniform_matched` are gated behind a checkbox rather than computed on
  every rerun, since the matched-size bisection search costs a few hundred milliseconds per
  frame. Below that, the full 80-frame `results/benchmark_summary.json` is shown for reference
  when present, with an `st.info` pointing at `run_benchmark.py` when it isn't.

**Playback, and a measured UI rate rather than an assumed one.** Play sets a
`playing` flag; after every tab renders, the dashboard sleeps to a target interval and calls
`st.rerun()`, which is the standard Streamlit pattern for this — clicking Pause is processed
before the script reruns from the top (an `on_click` callback, not a same-run inline check; see
below), so it takes effect on the very next tick rather than lagging by one frame. Every actual
gap between reruns is recorded, and the sidebar reports the mean of the last 20 as "Measured UI
rate" — the true cost of loading, preprocessing, gridding, and rendering every tab at whatever
the live zone and filter settings currently are, not a number computed once and assumed to
hold. This is what fills the "Render / UI FPS: unavailable" gap the benchmark harness
(Milestone 4) left open, since the CLI benchmark has no UI to time.

**A stale-button bug the fix is worth naming.** Streamlit evaluates a button's `disabled=`
condition at the moment it's drawn, in top-to-bottom script order — so checking a click inline
(`if col_prev.button(...):`) means that when Next is clicked, Prev (drawn first, from the
frame index *before* this click's effect) renders disabled for that one page, and a real
browser user genuinely cannot click it, not just an artifact of testing. `streamlit.testing.v1.AppTest`
caught this directly (`Cannot update a disabled button widget`) when a test clicked Next then
immediately tried Prev. The fix uses `on_click` callbacks instead: the callback mutates
`st.session_state` *before* the script reruns from the top, so `disabled=` sees the post-click
state. All four Prev/Next/Play/Pause buttons use this pattern now.

**The zone editor.** Each zone's `r_max` and cell size are editable (`r_min` is derived from the
previous zone's `r_max`, so contiguity can't be broken by the UI); `r_min`/`r_max`/`cell_m`
combine into a fresh `ZoneConfig` tuple and are checked with `avrmap.config.validate_zones` —
the exact function `load_config` itself uses, now public for this reason — before anything is
rebuilt. An edit that fails validation (say, a cell size that shrinks going outward) shows the
precise `ConfigError` message and the dashboard keeps rendering with the last valid ladder,
rather than crashing or silently accepting a broken configuration. Zone *count* stays fixed at
whatever the config specifies; adding or removing a zone isn't supported by the UI.

**The filter editor.** Range crop, height crop, dropped classes, and kept devices are all live
sliders and multiselects that feed a fresh `PreprocessConfig` into the same
`get_frame_bundle` cache function used everywhere else — an edit here really does reprocess the
frame and rebuild both grids, unlike the point-cloud tab's semantic-group filter, which stays
display-only on purpose (it doesn't need a rebuild, so it doesn't pay for one). Narrowing the
filters to zero surviving points shows a plain warning instead of an empty, confusing set of
tabs. A "Reset zones and filters to config defaults" button clears every edited widget back to
the shipped `configs/default.yaml` values.

**Rasterizing a sparse grid into an image.** `avrmap/render.py` paints a `CellTable` into a
fixed-size array — a rendering device, never the storage format, exactly as the technical plan
specifies. Cells sharing an edge length (every cell in a uniform grid; every cell in one
adaptive zone) are painted together with one broadcasted fancy-index assignment per group,
rather than a Python loop over cells. The image resolution is capped independently of the
configured display resolution (900 px per side by default), so a very fine zone size can never
make a dashboard rerun slow — the cap only coarsens the *picture*, never the underlying
`CellTable` or any measurement taken from it.

**Caching.** `st.cache_resource` covers dataset discovery and validation (per session, rarely
invalidated); `st.cache_data` covers the per-frame load-preprocess-and-grid bundle, now keyed on
every editable value (config path, sequence id, frame id, the range/height/class/device filter
settings, and the zone ladder as a tuple of plain `(r_min, r_max, cell_m)` tuples) — deliberately
primitives throughout, rather than the dataclasses those resolve to, since Streamlit's hashing
of custom objects is a needless risk when reconstructing them costs a few milliseconds anyway.
This also means every live edit is automatically part of the cache key with no extra
bookkeeping: stepping back to a frame/filter/zone combination already seen in this session is
instant.

**Testing the dashboard.** Streamlit's UI cannot be driven by ordinary pytest assertions on
pixels, so `app/dashboard.py` is smoke-tested with `streamlit.testing.v1.AppTest`, which runs
the actual script headlessly and surfaces any Python exception it raises. Verified this way,
without a single exception: the initial render (four tabs, ~30 sidebar elements), Prev/Next at
both sequence boundaries, the frame slider at an arbitrary and the last frame, a valid zone edit
and an invalid one (confirmed to show the precise rejection message and keep the last valid
ladder), the range/height filter sliders, emptying the drop-classes and devices multiselects,
narrowing the filters to zero surviving points (confirmed to show the warning rather than
crash), the Reset button, and the heaviest combined path — the last frame, the
`uniform_matched`/`uniform_coarse` checkbox, and a live zone edit together. `app/panels.py`'s
non-Plotly logic is covered in `tests/test_panels.py`.

The one thing AppTest cannot safely exercise is the autoplay loop itself: `st.rerun()` inside a
`playing=True` branch causes AppTest to unroll every subsequent script pass synchronously within
one `.run()` call, with no way for a test to interject a Pause click mid-loop the way a real
browser session's incoming message can — confirmed once, deliberately, as a timeout rather than
a hang. The pure decision logic autoplay depends on (`next_playback_index`: advance by one, loop
back to the start after the last frame) is instead unit-tested directly in
`tests/test_dashboard_logic.py`, decoupled from Streamlit entirely.

---

## Layout

```
avrmap/          processing library, no UI imports anywhere
  config.py        typed config, validated on load; validate_zones is public
                   so the dashboard's live zone editor reuses it directly
  dataset.py       discovery, indexing, validation
  frames.py        Frame dataclass and cached loading
  geometry.py      quaternions, poses, the three coordinate frames
  semantics.py     class names, groups, colour palette, vectorized group filtering
  preprocess.py    range/height crop, class and device filtering
  grids/
    common.py        CellTable, aggregate_cells, and its point-to-cell map
    uniform.py       the uniform grid, and the uniform_matched cell-size search
    adaptive.py      the foveated grid: per-zone cell sizes, concatenated
  metrics.py       build-time timing, dense-equivalent memory, per-band quality
  render.py        rasterizing a CellTable into a display image
app/             Streamlit dashboard
  dashboard.py     entry point: streamlit run app/dashboard.py
  panels.py        Plotly figure builders, no Streamlit calls
scripts/         command-line entry points, including run_benchmark.py
configs/         default.yaml
tests/           pytest suite, mirrors the module names
results/         benchmark.csv and benchmark_summary.json
cache/           generated frame cache
```

Nothing ever writes inside the sequence directory.

---

## Testing

```bash
python -m pytest -q              # everything
python -m pytest -q -m "not slow"  # synthetic fixtures only, no dataset needed
```

Fast tests build a throwaway PandaSet-shaped tree under pytest's `tmp_path` and provoke real
failure modes there: unpaired frames, row-count mismatches, missing columns, undeclared
labels, wrong pose counts. Tests marked `slow` read the actual dataset and skip cleanly when
it is absent.

210 tests total. `test_config.py` covers `load_config`'s validation for every rejected shape
(empty/gapped/non-contiguous/finer-outward/non-integer-ratio zones, inverted range or height
crops, an out-of-range zone ladder, unknown elevation statistics) and `validate_zones` directly,
including the empty-tuple guard the live editor depends on not to crash on. `test_preprocess.py`
covers each filter in isolation plus their overlap.
`test_grids.py` covers the aggregation kernel with hand-computable answers (a flat plane, a
dense square of known side and resolution, elevation statistics on `0..99`, semantic
tie-breaking, `min_points_per_cell` filtering, `CellTable.concat`), the point-to-cell mapping
(`return_point_cells`), the adaptive grid's zone assignment (boundary handling, per-zone cell
size, point conservation, skipped-empty-zone, `uncovered_point_count`), and
`find_matching_uniform_cell_size`'s bisection search — plus real-data regressions for the
frame-00 cell count, the frame-00 adaptive-vs-uniform comparison, and the full 80-frame
`adaptive < uniform_fine` claim (the slowest thing in the suite, about 15 s). `test_metrics.py`
hand-verifies `quality_by_band`'s RMSE and agreement formulas, including unmatched points and
empty bands. `test_run_benchmark.py` runs the benchmark CLI end to end on two frames and checks
the CSV and summary it produces are structurally sound — the full 80-frame run itself is left
to `python scripts/run_benchmark.py`, not the test suite, since it takes a few minutes.
`test_semantics.py` covers class colours, the palette, group assignment, and the vectorized
group filter against the same `group_of` logic computed one class at a time. `test_render.py`
hand-verifies pixel placement, block sizing for larger cells, the background colour, and the
image pixel cap. `test_panels.py` covers the dashboard's non-Plotly logic (subsampling, the
class-breakdown table). `test_dashboard_logic.py` covers the autoplay index-advance function
directly; the dashboard script itself is smoke-tested separately with `streamlit.testing.v1.AppTest`
rather than pytest — see the Dashboard section above for what that verified and why the
autoplay loop itself is the one thing it can't safely drive.

The full suite takes about 20-30 s; `-m "not slow"` skips every dataset-backed test and
finishes in about 1 s, useful while iterating on code that doesn't touch the real files.
