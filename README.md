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
| 4. Benchmark harness | not started |
| 5. Dashboard | not started |
| 6. Playback | not started |
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

This script is a one-frame sanity check, not the benchmark. Milestone 4 turns the same
building blocks into a proper multi-baseline (`uniform_fine`, `uniform_matched`,
`uniform_coarse`), multi-frame benchmark with elevation-error and semantic-agreement quality
metrics, written to `results/benchmark.csv`.

---

## Layout

```
avrmap/          processing library, no UI imports anywhere
  config.py        typed config, validated on load
  dataset.py       discovery, indexing, validation
  frames.py        Frame dataclass and cached loading
  geometry.py      quaternions, poses, the three coordinate frames
  semantics.py     class names, groups, colour palette
  preprocess.py    range/height crop, class and device filtering
  grids/
    common.py        CellTable and the aggregate_cells kernel
    uniform.py       the uniform-resolution grid
    adaptive.py      the foveated grid: per-zone cell sizes, concatenated
app/             Streamlit dashboard (milestone 5)
scripts/         command-line entry points
configs/         default.yaml
tests/           pytest suite, mirrors the module names
results/         generated benchmark output
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

95 tests total. `test_preprocess.py` covers each filter in isolation plus their overlap.
`test_grids.py` covers the aggregation kernel with hand-computable answers (a flat plane, a
dense square of known side and resolution, elevation statistics on `0..99`, semantic
tie-breaking, `min_points_per_cell` filtering, `CellTable.concat`), the adaptive grid's zone
assignment (boundary handling, per-zone cell size, point conservation, skipped-empty-zone,
`uncovered_point_count`), and pins two real-data regressions: the frame-00 cell count at 0.1 m
(63,611) and the frame-00 adaptive-vs-uniform comparison (27,295 cells, a 57.1% reduction).
The full 80-frame `adaptive < uniform_fine` check — the project's central claim — is itself a
test, not just a script output, and is the slowest thing in the suite at about 15 s.

The full suite takes about 30 s; `-m "not slow"` skips every dataset-backed test and finishes
in about 1 s, useful while iterating on code that doesn't touch the real files.
