"""PandaSet semantic classes: names, groups and display colours.

Class ids and names come from each sequence's ``annotations/semseg/classes.json``
rather than from a hardcoded table, so a sequence with a different label set
still works. Only the colours and the coarse groupings live here.

The groups below are our own interpretation, not part of PandaSet. They exist so
the pipeline can answer questions like "is this cell ground?" without scattering
magic class numbers through the code.
"""

from __future__ import annotations

import numpy as np

# PandaSet leaves a handful of points unlabelled and marks them 0. The id is not
# listed in classes.json, so it is named here instead. Measured on this sequence:
# 5 points in 1 of 80 frames, or 0.00004% of all returns.
UNLABELED_CLASS = 0
UNLABELED_NAME = "Unlabeled"

# Sensor artefacts rather than surfaces. Removed by default in preprocessing.
NOISE_CLASSES = (1, 2, 3, 4)
# Drivable and walkable surfaces, including painted markings.
GROUND_CLASSES = (6, 7, 8, 9, 10, 11, 12)
# Things that can move, whether or not they are moving in a given frame.
DYNAMIC_CLASSES = tuple(range(13, 34))
# Fixed structures and street furniture.
STATIC_CLASSES = (5,) + tuple(range(34, 43))

# Base colours, chosen so the groups read apart at a glance: greys and purples
# for road surface, warm hues for vehicles, bright accents for vulnerable road
# users, greens for vegetation, blues for buildings. Any class absent here falls
# back to a deterministic colour derived from its id.
_EXPLICIT_COLORS: dict[int, tuple[int, int, int]] = {
    0: (30, 30, 34),      # Unlabeled
    1: (140, 140, 140),   # Smoke
    2: (120, 120, 120),   # Exhaust
    3: (150, 160, 170),   # Spray or rain
    4: (170, 170, 150),   # Reflection
    5: (60, 150, 70),     # Vegetation
    6: (110, 95, 75),     # Ground
    7: (70, 70, 80),      # Road
    8: (230, 230, 235),   # Lane Line Marking
    9: (240, 200, 90),    # Stop Line Marking
    10: (200, 180, 140),  # Other Road Marking
    11: (150, 130, 160),  # Sidewalk
    12: (135, 120, 145),  # Driveway
    13: (220, 70, 60),    # Car
    14: (235, 110, 50),   # Pickup Truck
    15: (245, 150, 40),   # Medium-sized Truck
    16: (200, 90, 30),    # Semi-truck
    17: (175, 120, 80),   # Towed Object
    18: (230, 60, 140),   # Motorcycle
    19: (250, 190, 40),   # Other Vehicle - Construction Vehicle
    20: (190, 100, 120),  # Other Vehicle - Uncommon
    21: (180, 80, 150),   # Other Vehicle - Pedicab
    22: (255, 70, 70),    # Emergency Vehicle
    23: (240, 130, 70),   # Bus
    24: (120, 200, 220),  # Personal Mobility Device
    25: (90, 190, 200),   # Motorized Scooter
    26: (60, 180, 210),   # Bicycle
    27: (130, 110, 200),  # Train
    28: (120, 100, 190),  # Trolley
    29: (110, 90, 185),   # Tram / Subway
    30: (250, 240, 60),   # Pedestrian
    31: (235, 220, 90),   # Pedestrian with Object
    32: (200, 230, 120),  # Animals - Bird
    33: (185, 215, 110),  # Animals - Other
    34: (160, 160, 200),  # Pylons
    35: (145, 150, 175),  # Road Barriers
    36: (95, 175, 230),   # Signs
    37: (255, 140, 0),    # Cones
    38: (255, 170, 60),   # Construction Signs
    39: (215, 165, 100),  # Temporary Construction Barriers
    40: (170, 145, 120),  # Rolling Containers
    41: (70, 120, 190),   # Building
    42: (125, 135, 150),  # Other Static Object
}

_UNKNOWN_COLOR = (255, 0, 255)  # deliberately garish, so gaps are obvious


def _fallback_color(class_id: int) -> tuple[int, int, int]:
    """Deterministic colour for a class the palette does not name."""
    golden = 0.618033988749895
    hue = (class_id * golden) % 1.0
    # Simple HSV to RGB at full saturation and value, kept dependency-free.
    h6 = hue * 6.0
    sector = int(h6) % 6
    frac = h6 - int(h6)
    q, t = 1.0 - frac, frac
    table = [(1, t, 0), (q, 1, 0), (0, 1, t), (0, q, 1), (t, 0, 1), (1, 0, q)]
    return tuple(int(round(c * 235)) for c in table[sector])  # type: ignore[return-value]


def class_color(class_id: int) -> tuple[int, int, int]:
    """RGB colour for one class id."""
    if class_id in _EXPLICIT_COLORS:
        return _EXPLICIT_COLORS[class_id]
    if class_id < 0:
        return _UNKNOWN_COLOR
    return _fallback_color(class_id)


def build_palette(max_class_id: int = 255) -> np.ndarray:
    """(max_class_id + 1, 3) uint8 lookup table indexed directly by class id.

    A lookup table means colouring N points is one fancy-index operation rather
    than a per-point dictionary lookup.
    """
    palette = np.zeros((max_class_id + 1, 3), dtype=np.uint8)
    for class_id in range(max_class_id + 1):
        palette[class_id] = class_color(class_id)
    return palette


def colorize(labels: np.ndarray, palette: np.ndarray | None = None) -> np.ndarray:
    """Map an array of class ids to an (..., 3) uint8 RGB array."""
    if palette is None:
        palette = build_palette()
    return palette[np.asarray(labels)]


def class_name(classes: dict[int, str], class_id: int) -> str:
    """Human-readable name, falling back to the bare id for unknown classes."""
    cid = int(class_id)
    if cid == UNLABELED_CLASS:
        return classes.get(cid, UNLABELED_NAME)
    return classes.get(cid, f"class {cid}")


def group_of(class_id: int) -> str:
    """Coarse group name: unlabeled, noise, ground, dynamic, static, or other."""
    cid = int(class_id)
    if cid == UNLABELED_CLASS:
        return "unlabeled"
    if cid in NOISE_CLASSES:
        return "noise"
    if cid in GROUND_CLASSES:
        return "ground"
    if cid in DYNAMIC_CLASSES:
        return "dynamic"
    if cid in STATIC_CLASSES:
        return "static"
    return "other"


def group_mask(labels: np.ndarray, group: str) -> np.ndarray:
    """Boolean mask selecting points whose class belongs to ``group``."""
    lookup = {
        "noise": NOISE_CLASSES,
        "ground": GROUND_CLASSES,
        "dynamic": DYNAMIC_CLASSES,
        "static": STATIC_CLASSES,
    }
    if group not in lookup:
        raise ValueError(f"unknown class group '{group}', expected one of {sorted(lookup)}")
    return np.isin(labels, np.asarray(lookup[group], dtype=labels.dtype))
