"""Shared fixtures.

Two kinds of fixture live here. ``synthetic_sequence`` builds a throwaway
PandaSet-shaped tree under ``tmp_path`` so that failure modes can be provoked
without ever touching the real data. ``real_sequence`` points at the actual
dataset and skips the test when it is absent, so the suite still runs on a
checkout without the 760 MB of pickles.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from avrmap.config import default_config_path, load_config
from avrmap.dataset import discover_sequences


def write_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh)


def make_lidar_frame(n: int, seed: int = 0) -> pd.DataFrame:
    """A LiDAR frame with the same columns and dtypes as the real files."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.uniform(-50, 50, n),
            "y": rng.uniform(-50, 50, n),
            "z": rng.uniform(-2, 5, n),
            "i": rng.integers(0, 256, n).astype(float),
            "t": np.full(n, 1557539924.0 + seed * 0.1),
            "d": rng.integers(0, 2, n),
        }
    )


def make_semseg_frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1000)
    return pd.DataFrame({"class": rng.choice([5, 7, 13, 41], size=n)})


@pytest.fixture
def synthetic_sequence(tmp_path: Path) -> Path:
    """A minimal but complete three-frame sequence. Returns its root directory."""
    root = tmp_path / "seq_a"
    counts = [120, 130, 140]
    for i, n in enumerate(counts):
        fid = f"{i:02d}"
        write_pickle(root / "lidar" / f"{fid}.pkl", make_lidar_frame(n, seed=i))
        write_pickle(
            root / "annotations" / "semseg" / f"{fid}.pkl", make_semseg_frame(n, seed=i)
        )

    classes = {"5": "Vegetation", "7": "Road", "13": "Car", "41": "Building"}
    (root / "annotations" / "semseg" / "classes.json").write_text(
        json.dumps(classes), encoding="utf-8"
    )

    poses = [
        {
            "position": {"x": float(i), "y": 0.0, "z": 0.0},
            "heading": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        }
        for i in range(len(counts))
    ]
    (root / "lidar" / "poses.json").write_text(json.dumps(poses), encoding="utf-8")
    (root / "lidar" / "timestamps.json").write_text(
        json.dumps([1557539924.0 + 0.1 * i for i in range(len(counts))]),
        encoding="utf-8",
    )
    return root


@pytest.fixture(scope="session")
def config():
    return load_config(default_config_path())


@pytest.fixture(scope="session")
def real_sequence(config):
    """The actual PandaSet sequence, or a skip when it is not present."""
    sequences = discover_sequences(config.dataset.root, config.dataset.max_search_depth)
    if not sequences:
        pytest.skip(f"no sequence found under {config.dataset.root}")
    return sequences[0]
