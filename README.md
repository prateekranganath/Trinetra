# Adaptive Variable-Resolution 2.5D LiDAR Mapping

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
| 2. Preprocessing and uniform grid | not started |
| 3. Adaptive grid | not started |
| 4. Benchmark harness | not started |
| 5. Dashboard | not started |
| 6. Playback | not started |
| 7. Multi-frame accumulation and demo | not started |

---

## Setup

Python 3.13 on Windows. From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then validate the dataset and run the tests:

```bash
python scripts/validate_dataset.py          # checks all 80 frames, about 1.5 s
python scripts/validate_dataset.py --quick  # first, middle and last frame only
python -m pytest -q                         # full suite
python -m pytest -q -m "not slow"           # skip tests that read the real data
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
Both grid types will therefore share one representation: parallel NumPy arrays with one entry
per occupied cell. Same fields, same dtypes, only the row count differs, which is what makes
the memory comparison honest.

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
the same points, a 57% reduction at identical near-field resolution. Milestone 4 will
reproduce that properly across all 80 frames.

---

## Layout

```
avrmap/          processing library, no UI imports anywhere
  config.py        typed config, validated on load
  dataset.py       discovery, indexing, validation
  frames.py        Frame dataclass and cached loading
  geometry.py      quaternions, poses, the three coordinate frames
  semantics.py     class names, groups, colour palette
  grids/           (milestones 2 and 3)
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
