"""The small pieces of app/panels.py that hold real logic rather than pure
Plotly wiring: subsampling and the class-breakdown table.

The Streamlit dashboard itself (app/dashboard.py) is smoke-tested manually
via streamlit.testing.v1.AppTest rather than in this suite — see README.md's
Dashboard section for how and why.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from panels import class_breakdown_table, subsample_indices  # noqa: E402

from avrmap.config import ZoneConfig


class TestAddResolutionZones:
    def _fig(self):
        import plotly.graph_objects as go

        return go.Figure()

    def test_adds_one_ring_and_label_per_zone_plus_ego_marker(self):
        from panels import add_resolution_zones

        zones = (
            ZoneConfig(0.0, 10.0, 0.1),
            ZoneConfig(10.0, 30.0, 0.2),
            ZoneConfig(30.0, 60.0, 0.4),
            ZoneConfig(60.0, 100.0, 0.8),
        )
        fig = add_resolution_zones(self._fig(), zones)

        traces = [t for t in fig.data if t.mode == "lines"]
        assert len(traces) == len(zones)  # one circle per zone boundary
        markers = [t for t in fig.data if t.mode == "markers"]
        assert len(markers) == 1  # the ego vehicle

        radii = sorted(t.x.max() for t in traces)
        assert np.allclose(radii, [z.r_max for z in zones])

        labels = [a.text for a in fig.layout.annotations]
        assert labels == [f"{z.cell_m:g} m" for z in zones]

    def test_ring_radii_track_a_narrowed_ladder(self):
        from panels import add_resolution_zones

        zones = (ZoneConfig(0.0, 5.0, 0.05), ZoneConfig(5.0, 20.0, 0.1))
        fig = add_resolution_zones(self._fig(), zones)
        traces = [t for t in fig.data if t.mode == "lines"]
        assert np.allclose(sorted(t.x.max() for t in traces), [5.0, 20.0])


class TestSubsampleIndices:
    def test_returns_every_index_when_under_the_cap(self):
        idx = subsample_indices(10, max_points=100)
        assert list(idx) == list(range(10))

    def test_returns_exactly_max_points_when_over_the_cap(self):
        idx = subsample_indices(10_000, max_points=500)
        assert len(idx) == 500
        assert len(set(idx.tolist())) == 500  # no duplicates

    def test_indices_stay_within_range(self):
        idx = subsample_indices(1000, max_points=200)
        assert idx.min() >= 0 and idx.max() < 1000

    def test_is_deterministic_across_calls(self):
        a = subsample_indices(5000, max_points=300)
        b = subsample_indices(5000, max_points=300)
        assert np.array_equal(a, b)

    def test_exactly_at_the_cap_returns_every_index(self):
        idx = subsample_indices(50, max_points=50)
        assert len(idx) == 50


class TestClassBreakdownTable:
    def test_counts_and_sorts_by_frequency_descending(self):
        sem = np.array([7, 7, 7, 13, 41, 41], dtype=np.uint8)
        rows = class_breakdown_table(sem, {7: "Road", 13: "Car", 41: "Building"})
        assert rows[0] == {"class": 7, "name": "Road", "count": 3}
        assert [r["count"] for r in rows] == sorted([r["count"] for r in rows], reverse=True)

    def test_every_present_class_appears_exactly_once(self):
        sem = np.array([7, 13, 7, 13, 13], dtype=np.uint8)
        rows = class_breakdown_table(sem, {7: "Road", 13: "Car"})
        assert {r["class"] for r in rows} == {7, 13}
        assert len(rows) == 2

    def test_undeclared_class_falls_back_to_a_bare_name(self):
        sem = np.array([99], dtype=np.uint8)
        rows = class_breakdown_table(sem, {})
        assert rows[0]["name"] == "class 99"
