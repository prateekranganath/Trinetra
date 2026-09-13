"""Rasterizing a CellTable into a display image.

A raster is a rendering device, never the storage format, so every test here
checks the image against a hand-computed pixel layout rather than comparing
against a second implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from avrmap.grids.common import CellTable
from avrmap.render import (
    BACKGROUND_RGB,
    Raster,
    rasterize_scalar,
    rasterize_semantic,
    rasterize_zone,
)
from avrmap.semantics import class_color


def one_cell(cx=0.5, cy=0.5, size=1.0, zone=0, z_ref=3.0, sem_class=7, n_points=5) -> CellTable:
    return CellTable(
        cx=np.array([cx], dtype=np.float32), cy=np.array([cy], dtype=np.float32),
        size=np.array([size], dtype=np.float32), zone=np.array([zone], dtype=np.uint8),
        z_ref=np.array([z_ref], dtype=np.float32), z_min=np.array([z_ref], dtype=np.float32),
        z_max=np.array([z_ref], dtype=np.float32), n_points=np.array([n_points], dtype=np.uint32),
        sem_class=np.array([sem_class], dtype=np.uint8), sem_conf=np.array([1.0], dtype=np.float32),
    )


class TestRasterizeScalar:
    def test_a_single_cell_paints_exactly_one_pixel_at_unit_resolution(self):
        t = one_cell(cx=0.5, cy=0.5, size=1.0, z_ref=3.0)
        r = rasterize_scalar(t, t.z_ref, extent_m=2.0, display_res_m=1.0)
        assert r.image.shape == (4, 4)
        assert r.image[1, 2] == 3.0
        assert np.sum(~np.isnan(r.image)) == 1

    def test_empty_pixels_are_nan_not_zero(self):
        t = one_cell()
        r = rasterize_scalar(t, t.z_ref, extent_m=2.0, display_res_m=1.0)
        assert np.isnan(r.image[0, 0])

    def test_empty_table_produces_an_all_nan_image(self):
        r = rasterize_scalar(CellTable.empty(), np.zeros(0, dtype=np.float32), extent_m=5.0, display_res_m=1.0)
        assert np.isnan(r.image).all()
        assert r.image.shape[0] == r.image.shape[1] == 10

    def test_a_larger_cell_paints_a_correctly_sized_block(self):
        # a 2 m cell at 0.5 m resolution should paint a 4x4 pixel block
        t = one_cell(cx=1.0, cy=1.0, size=2.0, z_ref=7.0)
        r = rasterize_scalar(t, t.z_ref, extent_m=4.0, display_res_m=0.5)
        assert np.sum(r.image == 7.0) == 16

    def test_arbitrary_scalar_values_pass_through_unchanged(self):
        t = one_cell(z_ref=1.0)
        custom = np.array([42.5], dtype=np.float32)
        r = rasterize_scalar(t, custom, extent_m=2.0, display_res_m=1.0)
        assert np.nanmax(r.image) == pytest.approx(42.5)


class TestRasterizeSemantic:
    def test_a_cell_is_painted_its_classs_palette_colour(self):
        t = one_cell(sem_class=7)
        r = rasterize_semantic(t, extent_m=2.0, display_res_m=1.0)
        assert tuple(int(c) for c in r.image[1, 2]) == class_color(7)

    def test_empty_pixels_use_the_background_colour(self):
        t = one_cell()
        r = rasterize_semantic(t, extent_m=2.0, display_res_m=1.0)
        assert tuple(int(c) for c in r.image[0, 0]) == BACKGROUND_RGB

    def test_image_is_uint8_rgb(self):
        t = one_cell()
        r = rasterize_semantic(t, extent_m=2.0, display_res_m=1.0)
        assert r.image.dtype == np.uint8
        assert r.image.shape[-1] == 3


class TestRasterizeZone:
    def test_zone_zero_is_encoded_as_one_not_zero(self):
        # 0 is reserved for "empty", so zone id 0 must not collide with it.
        t = one_cell(zone=0)
        r = rasterize_zone(t, extent_m=2.0, display_res_m=1.0)
        assert r.image[1, 2] == 1
        assert r.image[0, 0] == 0

    def test_different_cell_sizes_are_grouped_and_painted_independently(self):
        t = CellTable(
            cx=np.array([0.05, 5.5], dtype=np.float32), cy=np.array([0.05, 0.5], dtype=np.float32),
            size=np.array([0.1, 1.0], dtype=np.float32), zone=np.array([0, 1], dtype=np.uint8),
            z_ref=np.array([1.0, 2.0], dtype=np.float32), z_min=np.array([1.0, 2.0], dtype=np.float32),
            z_max=np.array([1.0, 2.0], dtype=np.float32), n_points=np.array([3, 4], dtype=np.uint32),
            sem_class=np.array([7, 13], dtype=np.uint8), sem_conf=np.array([1.0, 1.0], dtype=np.float32),
        )
        r = rasterize_zone(t, extent_m=10.0, display_res_m=0.1)
        assert np.sum(r.image == 1) == 1     # the 0.1 m cell: exactly one pixel
        assert np.sum(r.image == 2) == 100   # the 1.0 m cell: a 10x10 block at 0.1 m res


class TestImagePixelCap:
    def test_resolution_is_coarsened_to_respect_the_pixel_cap(self):
        t = one_cell()
        r = rasterize_scalar(t, t.z_ref, extent_m=100.0, display_res_m=0.01, max_image_px=200)
        assert r.size_px <= 200
        assert r.resolution_m > 0.01  # coarsened from the requested resolution

    def test_a_request_within_the_cap_is_honoured_exactly(self):
        t = one_cell()
        r = rasterize_scalar(t, t.z_ref, extent_m=10.0, display_res_m=1.0, max_image_px=900)
        assert r.resolution_m == pytest.approx(1.0)


class TestRasterCoordAxis:
    def test_coord_axis_spans_the_extent_at_pixel_centres(self):
        r = Raster(image=np.zeros((4, 4)), extent_m=2.0, resolution_m=1.0)
        axis = r.coord_axis()
        assert len(axis) == 4
        assert axis[0] == pytest.approx(-1.5)
        assert axis[-1] == pytest.approx(1.5)
