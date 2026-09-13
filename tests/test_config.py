"""Config loading and validation.

The config is meant to fail loudly and early on a bad file, so most of this
suite provokes a specific bad section and checks the error names it plainly.
A minimal valid config is built fresh per test via ``write_config`` rather
than editing the shipped default, so tests can't drift from what's actually
deployed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from avrmap.config import (
    ConfigError,
    GridConfig,
    ZoneConfig,
    default_config_path,
    load_config,
    project_root,
    validate_zones,
)

MINIMAL = {
    "dataset": {"root": ".", "max_search_depth": 2},
    "preprocess": {
        "min_range_m": 0.0, "max_range_m": 100.0,
        "z_min_m": -10.0, "z_max_m": 30.0,
        "drop_classes": [0, 1, 2, 3, 4], "devices": [0, 1],
    },
    "grid": {
        "uniform_cell_m": 0.1, "min_points_per_cell": 1, "elevation_stat": "p95",
        "zones": [
            {"r_min": 0.0, "r_max": 10.0, "cell_m": 0.1},
            {"r_min": 10.0, "r_max": 30.0, "cell_m": 0.2},
            {"r_min": 30.0, "r_max": 60.0, "cell_m": 0.4},
            {"r_min": 60.0, "r_max": 100.0, "cell_m": 0.8},
        ],
    },
    "render": {"display_res_m": 0.2, "extent_m": 100.0, "max_points_3d": 40000},
    "benchmark": {"repeats": 3, "bands_m": [0.0, 10.0, 30.0, 60.0, 100.0]},
}


def write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    """Deep-ish merge of ``overrides`` onto MINIMAL, one section at a time."""
    import copy

    doc = copy.deepcopy(MINIMAL)
    for section, values in (overrides or {}).items():
        if isinstance(values, dict) and isinstance(doc.get(section), dict):
            doc[section].update(values)
        else:
            doc[section] = values
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


class TestLoadConfigHappyPath:
    def test_the_shipped_default_config_loads_and_validates(self):
        cfg = load_config(default_config_path())
        assert cfg.grid.zones
        assert cfg.source_path == default_config_path().resolve()

    def test_a_minimal_valid_config_round_trips_into_the_expected_values(self, tmp_path):
        cfg = load_config(write_config(tmp_path))
        assert cfg.preprocess.max_range_m == 100.0
        assert cfg.grid.uniform_cell_m == 0.1
        assert len(cfg.grid.zones) == 4
        assert cfg.benchmark.bands_m == (0.0, 10.0, 30.0, 60.0, 100.0)

    def test_relative_dataset_root_resolves_two_levels_above_the_config_file(self, tmp_path):
        # load_config assumes a config lives at <project_root>/configs/<name>.yaml,
        # matching how default_config_path() and every script in this repo use it.
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        path = write_config(configs_dir, {"dataset": {"root": "Data"}})
        cfg = load_config(path)
        assert cfg.dataset.root == (tmp_path / "Data").resolve()

    def test_the_shipped_default_config_root_is_the_actual_project_root(self):
        # A concrete regression for the relative-root convention above, using
        # the real file this project ships rather than a synthetic layout.
        cfg = load_config(default_config_path())
        assert cfg.dataset.root == project_root().resolve()

    def test_missing_optional_sections_fall_back_to_documented_defaults(self, tmp_path):
        doc = {"grid": MINIMAL["grid"]}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(doc), encoding="utf-8")
        cfg = load_config(path)
        assert cfg.preprocess.max_range_m == 100.0  # PreprocessConfig's own default
        assert cfg.render.max_points_3d == 40_000


class TestLoadConfigErrors:
    def test_missing_file_is_a_config_error(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_non_mapping_root_is_rejected(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="must be a mapping"):
            load_config(path)

    def test_empty_zones_list_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="non-empty"):
            load_config(write_config(tmp_path, {"grid": {**MINIMAL["grid"], "zones": []}}))

    def test_zone_missing_a_required_key_is_rejected(self, tmp_path):
        bad_zones = [{"r_min": 0.0, "r_max": 10.0}]  # no cell_m
        with pytest.raises(ConfigError, match="missing"):
            load_config(write_config(tmp_path, {"grid": {**MINIMAL["grid"], "zones": bad_zones}}))

    def test_zone_with_r_max_not_exceeding_r_min_is_rejected(self, tmp_path):
        bad_zones = [{"r_min": 10.0, "r_max": 10.0, "cell_m": 0.1}]
        with pytest.raises(ConfigError, match="r_max <= r_min"):
            load_config(write_config(tmp_path, {"grid": {**MINIMAL["grid"], "zones": bad_zones}}))

    def test_zone_with_non_positive_cell_size_is_rejected(self, tmp_path):
        bad_zones = [{"r_min": 0.0, "r_max": 10.0, "cell_m": 0.0}]
        with pytest.raises(ConfigError, match="non-positive cell_m"):
            load_config(write_config(tmp_path, {"grid": {**MINIMAL["grid"], "zones": bad_zones}}))

    def test_zones_not_starting_at_zero_is_rejected(self, tmp_path):
        bad_zones = [{"r_min": 1.0, "r_max": 10.0, "cell_m": 0.1}]
        with pytest.raises(ConfigError, match="must start at r_min: 0.0"):
            load_config(write_config(tmp_path, {"grid": {**MINIMAL["grid"], "zones": bad_zones}}))

    def test_a_gap_between_zones_is_rejected(self, tmp_path):
        bad_zones = [
            {"r_min": 0.0, "r_max": 10.0, "cell_m": 0.1},
            {"r_min": 11.0, "r_max": 20.0, "cell_m": 0.2},  # gap: 10 to 11
        ]
        with pytest.raises(ConfigError, match="must be contiguous"):
            load_config(write_config(tmp_path, {"grid": {**MINIMAL["grid"], "zones": bad_zones}}))

    def test_a_zone_getting_finer_with_distance_is_rejected(self, tmp_path):
        bad_zones = [
            {"r_min": 0.0, "r_max": 10.0, "cell_m": 0.4},
            {"r_min": 10.0, "r_max": 20.0, "cell_m": 0.1},  # finer than the previous
        ]
        with pytest.raises(ConfigError, match="must not get finer"):
            load_config(write_config(tmp_path, {"grid": {**MINIMAL["grid"], "zones": bad_zones}}))

    def test_a_non_integer_cell_size_ratio_between_zones_is_rejected(self, tmp_path):
        bad_zones = [
            {"r_min": 0.0, "r_max": 10.0, "cell_m": 0.1},
            {"r_min": 10.0, "r_max": 20.0, "cell_m": 0.15},  # 1.5x, not a whole number
        ]
        with pytest.raises(ConfigError, match="integer multiple"):
            load_config(write_config(tmp_path, {"grid": {**MINIMAL["grid"], "zones": bad_zones}}))

    def test_zones_reaching_beyond_the_range_crop_is_rejected(self, tmp_path):
        cfg_dict = {
            "preprocess": {**MINIMAL["preprocess"], "max_range_m": 50.0},
        }
        with pytest.raises(ConfigError, match="could never receive points"):
            load_config(write_config(tmp_path, cfg_dict))

    def test_unknown_elevation_stat_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="elevation_stat"):
            load_config(write_config(tmp_path, {"grid": {**MINIMAL["grid"], "elevation_stat": "bogus"}}))

    def test_non_positive_uniform_cell_size_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="uniform_cell_m must be positive"):
            load_config(write_config(tmp_path, {"grid": {**MINIMAL["grid"], "uniform_cell_m": 0.0}}))

    def test_min_points_per_cell_below_one_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="min_points_per_cell must be at least 1"):
            load_config(write_config(tmp_path, {"grid": {**MINIMAL["grid"], "min_points_per_cell": 0}}))

    def test_inverted_range_crop_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="max_range_m must exceed"):
            load_config(write_config(tmp_path, {"preprocess": {**MINIMAL["preprocess"], "max_range_m": 0.0}}))

    def test_inverted_height_crop_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="z_max_m must exceed"):
            load_config(write_config(tmp_path, {"preprocess": {**MINIMAL["preprocess"], "z_max_m": -20.0}}))

    def test_too_few_benchmark_bands_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="bands_m"):
            load_config(write_config(tmp_path, {"benchmark": {**MINIMAL["benchmark"], "bands_m": [0.0]}}))

    def test_non_increasing_benchmark_bands_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="bands_m"):
            load_config(write_config(tmp_path, {"benchmark": {**MINIMAL["benchmark"], "bands_m": [0.0, 10.0, 5.0]}}))


class TestValidateZonesDirectly:
    """The public entry point the dashboard's live zone editor (Milestone 6)
    calls before rebuilding the adaptive grid with user-edited zones."""

    GOOD = (
        ZoneConfig(0.0, 10.0, 0.1),
        ZoneConfig(10.0, 30.0, 0.2),
    )

    def test_a_good_ladder_raises_nothing(self):
        validate_zones(self.GOOD)  # no exception

    def test_empty_tuple_is_rejected_without_an_index_error(self):
        with pytest.raises(ConfigError, match="must not be empty"):
            validate_zones(())

    def test_equal_cell_sizes_between_zones_are_allowed(self):
        # non-decreasing, not strictly increasing, is the actual rule
        validate_zones((ZoneConfig(0.0, 10.0, 0.2), ZoneConfig(10.0, 20.0, 0.2)))


class TestGridConfigProperties:
    def test_finest_and_coarsest_cell_m(self):
        grid = GridConfig(zones=(ZoneConfig(0, 10, 0.1), ZoneConfig(10, 30, 0.2), ZoneConfig(30, 60, 0.4)))
        assert grid.finest_cell_m == 0.1
        assert grid.coarsest_cell_m == 0.4

    def test_max_zone_range_m_is_the_outermost_edge(self):
        grid = GridConfig(zones=(ZoneConfig(0, 10, 0.1), ZoneConfig(10, 30, 0.2)))
        assert grid.max_zone_range_m == 30.0


class TestZoneConfigContains:
    def test_half_open_interval_excludes_the_upper_bound(self):
        zone = ZoneConfig(0.0, 10.0, 0.1)
        import numpy as np

        radius = np.array([0.0, 5.0, 9.999, 10.0])
        assert list(zone.contains(radius)) == [True, True, True, False]
