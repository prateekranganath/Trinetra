"""End-to-end checks for the benchmark CLI, on a couple of real frames.

Only two frames are run here — the full 80-frame benchmark takes a few
minutes and is meant to be run deliberately (`python scripts/run_benchmark.py`),
not on every test invocation. This test instead checks the CSV and summary
this script produces are structurally sound and internally consistent.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.run_benchmark as run_benchmark  # noqa: E402


@pytest.mark.slow
class TestRunBenchmarkCLI:
    def test_two_frames_produce_a_well_formed_csv_and_summary(self, tmp_path, config):
        out_dir = tmp_path / "results"
        exit_code = run_benchmark.main(
            [
                "--config", str(config.source_path),
                "--frames", "00,01",
                "--out", str(out_dir),
                "--repeats", "1",
                "--matched-iterations", "6",
            ]
        )
        assert exit_code == 0

        csv_path = out_dir / "benchmark.csv"
        assert csv_path.is_file()
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2 * len(run_benchmark.METHODS)
        assert {r["frame_id"] for r in rows} == {"00", "01"}
        assert {r["method"] for r in rows} == set(run_benchmark.METHODS)

        by_frame_method = {(r["frame_id"], r["method"]): r for r in rows}
        for frame_id in ("00", "01"):
            fine = int(by_frame_method[(frame_id, "uniform_fine")]["n_cells"])
            adaptive = int(by_frame_method[(frame_id, "adaptive")]["n_cells"])
            coarse = int(by_frame_method[(frame_id, "uniform_coarse")]["n_cells"])
            assert adaptive < fine, "adaptive must use fewer cells than uniform_fine"
            assert coarse < adaptive, "uniform_coarse must use fewer cells than adaptive"
            # near-band quality should match closely: same 0.1 m cells there
            band0 = config.benchmark.bands_m[0], config.benchmark.bands_m[1]
            col = f"rmse_{band0[0]:g}_{band0[1]:g}m"
            assert by_frame_method[(frame_id, "uniform_fine")][col] == \
                by_frame_method[(frame_id, "adaptive")][col]

        summary_path = out_dir / "benchmark_summary.json"
        assert summary_path.is_file()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["n_frames"] == 2
        assert set(summary["methods"]) == set(run_benchmark.METHODS)
        assert summary["methods"]["adaptive"]["cell_reduction_vs_uniform_fine_pct"] > 0
        assert "render_fps" in summary  # explicitly reported as unavailable, not omitted

    def test_frames_all_selects_the_whole_sequence(self, real_sequence):
        assert run_benchmark._select_frames(real_sequence, "all") == list(real_sequence.frame_ids)

    def test_frames_n_selects_a_prefix(self, real_sequence):
        assert run_benchmark._select_frames(real_sequence, "5") == list(real_sequence.frame_ids[:5])

    def test_unknown_frame_id_in_a_list_is_rejected(self, real_sequence):
        from avrmap.dataset import DatasetError

        with pytest.raises(DatasetError, match="not in sequence"):
            run_benchmark._select_frames(real_sequence, "00,does-not-exist")

    def test_out_of_range_frame_count_is_rejected(self, real_sequence):
        from avrmap.dataset import DatasetError

        with pytest.raises(DatasetError, match="out of range"):
            run_benchmark._select_frames(real_sequence, str(len(real_sequence) + 1))
