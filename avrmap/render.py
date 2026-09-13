"""Rasterizing a CellTable into a display image for the dashboard.

A raster is **a rendering device, not the storage format**. The CellTable
stays sparse everywhere else in this project; painting one into a fixed-size
image only happens here, so a variable-resolution grid can be shown as one
picture. The image resolution is capped independently of the configured
display resolution, so a very fine zone size can never blow up rendering
time or memory — that cap is purely about drawing speed and has nothing to
do with how the grid itself is stored or measured.

Cells share their edge length within a zone (a uniform grid has exactly one
size; an adaptive grid has one size per zone), so painting is vectorized by
grouping cells on that size rather than looping over individual cells: every
cell in a group is painted with one broadcasted fancy-index assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grids.common import CellTable
from .semantics import build_palette

#: Image side length is never allowed to exceed this, regardless of the
#: requested display resolution, so a dashboard rerun stays fast.
DEFAULT_MAX_IMAGE_PX = 900

#: Colour shown where no cell exists.
BACKGROUND_RGB = (24, 24, 28)


@dataclass(frozen=True)
class Raster:
    """A square image plus the real-world extent it covers.

    Attributes:
        image: ``(H, W)`` for a scalar raster (NaN where empty) or
            ``(H, W, 3)`` uint8 for an RGB raster.
        extent_m: The image covers ``[-extent_m, extent_m]`` on both axes.
        resolution_m: Actual metres per pixel used, which may be coarser
            than requested if the request would have exceeded the pixel cap.
    """

    image: np.ndarray
    extent_m: float
    resolution_m: float

    @property
    def size_px(self) -> int:
        return self.image.shape[0]

    def coord_axis(self) -> np.ndarray:
        """Pixel-centre coordinates along one axis, for plotting real ticks."""
        n = self.size_px
        return np.linspace(
            -self.extent_m + self.resolution_m / 2,
            self.extent_m - self.resolution_m / 2,
            n,
        )


def _effective_resolution(extent_m: float, requested_res_m: float, max_px: int) -> float:
    """Never let the image exceed ``max_px`` per side."""
    min_res = (2 * extent_m) / max_px
    return max(requested_res_m, min_res)


def _pixel_blocks(table: CellTable, extent_m: float, res_m: float, img_size: int):
    """Group cells by (rounded) pixel block size and yield paint targets.

    Row 0 corresponds to y = +extent_m (top of the image); row increases as y
    decreases. Column 0 corresponds to x = -extent_m. A cell's block can be
    clipped at the image edge (a cell straddling the configured extent), which
    is harmless: the clipped indices still only ever refer to that same
    cell's own pixels.
    """
    if len(table) == 0:
        return
    for size in np.unique(table.size):
        mask = table.size == size
        block = max(1, int(round(float(size) / res_m)))
        cx, cy = table.cx[mask].astype(np.float64), table.cy[mask].astype(np.float64)
        col0 = np.floor((cx - size / 2 + extent_m) / res_m).astype(np.int64)
        row0 = np.floor((extent_m - cy - size / 2) / res_m).astype(np.int64)
        yield mask, row0, col0, block


def _paint(img: np.ndarray, row0: np.ndarray, col0: np.ndarray, block: int, values: np.ndarray, img_size: int) -> None:
    rows = np.clip(row0[:, None] + np.arange(block)[None, :], 0, img_size - 1)
    cols = np.clip(col0[:, None] + np.arange(block)[None, :], 0, img_size - 1)
    if img.ndim == 2:
        img[rows[:, :, None], cols[:, None, :]] = values[:, None, None]
    else:
        img[rows[:, :, None], cols[:, None, :], :] = values[:, None, None, :]


def rasterize_scalar(
    table: CellTable,
    values: np.ndarray,
    extent_m: float,
    display_res_m: float,
    max_image_px: int = DEFAULT_MAX_IMAGE_PX,
) -> Raster:
    """A float raster of an arbitrary per-cell scalar (e.g. ``z_ref``).

    Empty pixels are NaN, which every caller must handle explicitly (Plotly's
    continuous colorscales already render NaN as transparent).
    """
    res = _effective_resolution(extent_m, display_res_m, max_image_px)
    img_size = int(round(2 * extent_m / res))
    img = np.full((img_size, img_size), np.nan, dtype=np.float32)
    for mask, row0, col0, block in _pixel_blocks(table, extent_m, res, img_size):
        _paint(img, row0, col0, block, values[mask].astype(np.float32), img_size)
    return Raster(image=img, extent_m=extent_m, resolution_m=res)


def rasterize_semantic(
    table: CellTable,
    extent_m: float,
    display_res_m: float,
    max_image_px: int = DEFAULT_MAX_IMAGE_PX,
    palette: np.ndarray | None = None,
) -> Raster:
    """An RGB raster colouring each cell by its majority semantic class."""
    if palette is None:
        palette = build_palette()
    res = _effective_resolution(extent_m, display_res_m, max_image_px)
    img_size = int(round(2 * extent_m / res))
    img = np.full((img_size, img_size, 3), BACKGROUND_RGB, dtype=np.uint8)
    for mask, row0, col0, block in _pixel_blocks(table, extent_m, res, img_size):
        colors = palette[table.sem_class[mask]]
        _paint(img, row0, col0, block, colors, img_size)
    return Raster(image=img, extent_m=extent_m, resolution_m=res)


def rasterize_zone(
    table: CellTable,
    extent_m: float,
    display_res_m: float,
    max_image_px: int = DEFAULT_MAX_IMAGE_PX,
) -> Raster:
    """An integer raster of ``zone + 1`` (0 means empty).

    This is what makes the fovea legible at a glance: colour each pixel by
    which distance zone (and therefore which cell size) produced it.
    """
    res = _effective_resolution(extent_m, display_res_m, max_image_px)
    img_size = int(round(2 * extent_m / res))
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    for mask, row0, col0, block in _pixel_blocks(table, extent_m, res, img_size):
        _paint(img, row0, col0, block, (table.zone[mask].astype(np.int32) + 1).astype(np.uint8), img_size)
    return Raster(image=img, extent_m=extent_m, resolution_m=res)
