"""Typed configuration loaded from YAML.

The config is validated eagerly and loudly. A bad zone ladder should fail when
the file is read, not halfway through a benchmark run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is structurally invalid."""


@dataclass(frozen=True)
class ZoneConfig:
    """One distance zone of the adaptive grid.

    The interval is half-open, [r_min, r_max), so a point sitting exactly on a
    boundary belongs to the outer zone and never to both.
    """

    r_min: float
    r_max: float
    cell_m: float

    def contains(self, radius):
        return (radius >= self.r_min) & (radius < self.r_max)


@dataclass(frozen=True)
class GridConfig:
    uniform_cell_m: float = 0.1
    min_points_per_cell: int = 1
    elevation_stat: str = "p95"
    zones: tuple[ZoneConfig, ...] = ()

    @property
    def finest_cell_m(self) -> float:
        return min(z.cell_m for z in self.zones)

    @property
    def coarsest_cell_m(self) -> float:
        return max(z.cell_m for z in self.zones)

    @property
    def max_zone_range_m(self) -> float:
        return max(z.r_max for z in self.zones)


@dataclass(frozen=True)
class PreprocessConfig:
    min_range_m: float = 0.0
    max_range_m: float = 100.0
    z_min_m: float = -10.0
    z_max_m: float = 30.0
    drop_classes: tuple[int, ...] = (1, 2, 3, 4)
    devices: tuple[int, ...] = (0, 1)


@dataclass(frozen=True)
class DatasetConfig:
    root: Path = Path(".")
    max_search_depth: int = 2


@dataclass(frozen=True)
class RenderConfig:
    display_res_m: float = 0.2
    extent_m: float = 100.0
    max_points_3d: int = 40_000


@dataclass(frozen=True)
class BenchmarkConfig:
    repeats: int = 3
    bands_m: tuple[float, ...] = (0.0, 10.0, 30.0, 60.0, 100.0)


@dataclass(frozen=True)
class PipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    source_path: Path | None = None


VALID_ELEVATION_STATS = ("p95", "max", "mean", "min")


def _require_mapping(value: Any, where: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{where}' must be a mapping, got {type(value).__name__}")
    return value


def _parse_zones(raw: Any) -> tuple[ZoneConfig, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError("grid.zones must be a non-empty list")
    zones = []
    for i, item in enumerate(raw):
        item = _require_mapping(item, f"grid.zones[{i}]")
        missing = {"r_min", "r_max", "cell_m"} - set(item)
        if missing:
            raise ConfigError(f"grid.zones[{i}] is missing {sorted(missing)}")
        zone = ZoneConfig(
            r_min=float(item["r_min"]),
            r_max=float(item["r_max"]),
            cell_m=float(item["cell_m"]),
        )
        if zone.r_max <= zone.r_min:
            raise ConfigError(f"grid.zones[{i}] has r_max <= r_min")
        if zone.cell_m <= 0:
            raise ConfigError(f"grid.zones[{i}] has a non-positive cell_m")
        zones.append(zone)
    return tuple(zones)


def validate_zones(zones: tuple[ZoneConfig, ...]) -> None:
    """Zones must tile the range without gaps or overlap, coarsening outwards.

    Public (not prefixed with an underscore) because the dashboard's live
    zone editor (Milestone 6) reuses this exact check before rebuilding the
    adaptive grid with user-edited zones, rather than re-implementing a
    second, possibly-diverging notion of "valid zones".
    """
    if not zones:
        raise ConfigError("grid.zones must not be empty")
    if zones[0].r_min != 0.0:
        raise ConfigError("the first grid zone must start at r_min: 0.0")
    for prev, cur in zip(zones, zones[1:]):
        if abs(cur.r_min - prev.r_max) > 1e-9:
            raise ConfigError(
                f"grid.zones must be contiguous: a zone ending at {prev.r_max} m is "
                f"followed by one starting at {cur.r_min} m"
            )
        if cur.cell_m < prev.cell_m:
            raise ConfigError(
                "grid.zones must not get finer with distance: "
                f"{prev.cell_m} m at {prev.r_min}-{prev.r_max} m is followed by "
                f"{cur.cell_m} m at {cur.r_min}-{cur.r_max} m"
            )
        ratio = cur.cell_m / prev.cell_m
        if abs(ratio - round(ratio)) > 1e-6:
            raise ConfigError(
                "each zone's cell size must be an integer multiple of the previous "
                f"one so coarse cells nest onto fine ones, but {cur.cell_m} / "
                f"{prev.cell_m} = {ratio:.4f}"
            )


def load_config(path: str | Path) -> PipelineConfig:
    """Read and validate a YAML configuration file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = _require_mapping(raw, "<root>")

    ds = _require_mapping(raw.get("dataset"), "dataset")
    # A relative dataset root resolves two directories above the config file
    # itself, i.e. it assumes the config lives at <project_root>/configs/*.yaml,
    # exactly where default_config_path() and every script's --config flag in
    # this repo expect it. A config file placed anywhere else must use an
    # absolute dataset.root.
    root = Path(ds.get("root", "."))
    if not root.is_absolute():
        root = (path.resolve().parent.parent / root).resolve()
    dataset = DatasetConfig(
        root=root, max_search_depth=int(ds.get("max_search_depth", 2))
    )

    pp = _require_mapping(raw.get("preprocess"), "preprocess")
    preprocess = PreprocessConfig(
        min_range_m=float(pp.get("min_range_m", 0.0)),
        max_range_m=float(pp.get("max_range_m", 100.0)),
        z_min_m=float(pp.get("z_min_m", -10.0)),
        z_max_m=float(pp.get("z_max_m", 30.0)),
        drop_classes=tuple(int(c) for c in pp.get("drop_classes", ())),
        devices=tuple(int(d) for d in pp.get("devices", ())),
    )
    if preprocess.max_range_m <= preprocess.min_range_m:
        raise ConfigError("preprocess.max_range_m must exceed preprocess.min_range_m")
    if preprocess.z_max_m <= preprocess.z_min_m:
        raise ConfigError("preprocess.z_max_m must exceed preprocess.z_min_m")

    gd = _require_mapping(raw.get("grid"), "grid")
    zones = _parse_zones(gd.get("zones"))
    validate_zones(zones)
    stat = str(gd.get("elevation_stat", "p95"))
    if stat not in VALID_ELEVATION_STATS:
        raise ConfigError(
            f"grid.elevation_stat must be one of {VALID_ELEVATION_STATS}, got '{stat}'"
        )
    grid = GridConfig(
        uniform_cell_m=float(gd.get("uniform_cell_m", 0.1)),
        min_points_per_cell=int(gd.get("min_points_per_cell", 1)),
        elevation_stat=stat,
        zones=zones,
    )
    if grid.uniform_cell_m <= 0:
        raise ConfigError("grid.uniform_cell_m must be positive")
    if grid.min_points_per_cell < 1:
        raise ConfigError("grid.min_points_per_cell must be at least 1")
    if grid.max_zone_range_m > preprocess.max_range_m + 1e-9:
        raise ConfigError(
            f"the outermost grid zone reaches {grid.max_zone_range_m} m but "
            f"preprocess.max_range_m is {preprocess.max_range_m} m, so that zone "
            "could never receive points"
        )

    rd = _require_mapping(raw.get("render"), "render")
    render = RenderConfig(
        display_res_m=float(rd.get("display_res_m", 0.2)),
        extent_m=float(rd.get("extent_m", 100.0)),
        max_points_3d=int(rd.get("max_points_3d", 40_000)),
    )

    bd = _require_mapping(raw.get("benchmark"), "benchmark")
    bands = tuple(float(b) for b in bd.get("bands_m", (0.0, 10.0, 30.0, 60.0, 100.0)))
    if len(bands) < 2 or any(b <= a for a, b in zip(bands, bands[1:])):
        raise ConfigError(
            "benchmark.bands_m must be at least two strictly increasing values"
        )
    benchmark = BenchmarkConfig(repeats=int(bd.get("repeats", 3)), bands_m=bands)

    return PipelineConfig(
        dataset=dataset,
        preprocess=preprocess,
        grid=grid,
        render=render,
        benchmark=benchmark,
        source_path=path.resolve(),
    )


def project_root() -> Path:
    """Directory containing the avrmap package."""
    return Path(__file__).resolve().parent.parent


def default_config_path() -> Path:
    """Path to the configuration shipped with the project."""
    return project_root() / "configs" / "default.yaml"
