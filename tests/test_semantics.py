"""PandaSet semantic classes: names, groups, colours, and vectorized filtering."""

from __future__ import annotations

import numpy as np
import pytest

from avrmap.semantics import (
    DYNAMIC_CLASSES,
    GROUND_CLASSES,
    GROUP_NAMES,
    NOISE_CLASSES,
    STATIC_CLASSES,
    UNLABELED_CLASS,
    build_group_lookup,
    build_palette,
    class_color,
    class_name,
    colorize,
    group_mask,
    group_of,
    groups_mask,
)


class TestClassColor:
    def test_named_classes_return_their_explicit_colour(self):
        assert class_color(7) == (70, 70, 80)  # Road

    def test_unlabeled_and_unknown_negative_ids_differ(self):
        # 0 (unlabeled) has its own dark colour; a negative id is the garish
        # "something is wrong" fallback, and the two must not collide.
        assert class_color(0) != class_color(-1)

    def test_an_undeclared_positive_class_gets_a_deterministic_fallback(self):
        a = class_color(200)
        b = class_color(200)
        assert a == b  # same id -> same colour every time

    def test_fallback_colours_differ_between_classes(self):
        assert class_color(150) != class_color(151)


class TestPalette:
    def test_palette_covers_every_class_id_up_to_the_max(self):
        p = build_palette(max_class_id=50)
        assert p.shape == (51, 3)
        assert p.dtype == np.uint8

    def test_palette_rows_match_class_color(self):
        p = build_palette(max_class_id=50)
        assert tuple(p[7]) == class_color(7)
        assert tuple(p[0]) == class_color(0)

    def test_colorize_maps_an_array_through_the_palette(self):
        labels = np.array([7, 13, 0], dtype=np.uint8)
        rgb = colorize(labels)
        assert rgb.shape == (3, 3)
        assert tuple(rgb[0]) == class_color(7)
        assert tuple(rgb[2]) == class_color(0)

    def test_colorize_preserves_input_shape(self):
        labels = np.array([[7, 13], [41, 0]], dtype=np.uint8)
        rgb = colorize(labels)
        assert rgb.shape == (2, 2, 3)


class TestClassName:
    def test_declared_class_uses_the_provided_name(self):
        assert class_name({7: "Road"}, 7) == "Road"

    def test_unlabeled_falls_back_to_a_fixed_name_when_undeclared(self):
        assert class_name({}, UNLABELED_CLASS) == "Unlabeled"

    def test_other_undeclared_classes_fall_back_to_a_bare_id(self):
        assert class_name({}, 99) == "class 99"

    def test_declared_unlabeled_overrides_the_default_name(self):
        # if a sequence's classes.json ever does declare 0, that wins
        assert class_name({0: "Custom"}, 0) == "Custom"


class TestGroupOf:
    def test_every_declared_group_constant_maps_back_to_its_own_group(self):
        for cid in NOISE_CLASSES:
            assert group_of(cid) == "noise"
        for cid in GROUND_CLASSES:
            assert group_of(cid) == "ground"
        for cid in DYNAMIC_CLASSES:
            assert group_of(cid) == "dynamic"
        for cid in STATIC_CLASSES:
            assert group_of(cid) == "static"

    def test_unlabeled_is_its_own_group(self):
        assert group_of(UNLABELED_CLASS) == "unlabeled"

    def test_an_id_outside_every_range_is_other(self):
        assert group_of(255) == "other"

    def test_groups_partition_the_declared_ranges_without_overlap(self):
        all_ids = set(NOISE_CLASSES) | set(GROUND_CLASSES) | set(DYNAMIC_CLASSES) | set(STATIC_CLASSES)
        assert len(all_ids) == (
            len(NOISE_CLASSES) + len(GROUND_CLASSES) + len(DYNAMIC_CLASSES) + len(STATIC_CLASSES)
        )


class TestGroupMask:
    def test_selects_only_the_requested_group(self):
        labels = np.array([7, 13, 41, 5], dtype=np.uint8)  # ground, dynamic, static, static
        mask = group_mask(labels, "static")
        assert list(mask) == [False, False, True, True]

    def test_unknown_group_name_is_rejected(self):
        with pytest.raises(ValueError, match="unknown class group"):
            group_mask(np.array([7], dtype=np.uint8), "bogus")


class TestBuildGroupLookup:
    def test_matches_group_of_for_every_class_id(self):
        lookup = build_group_lookup(max_class_id=255)
        for cid in range(256):
            assert GROUP_NAMES[lookup[cid]] == group_of(cid)

    def test_output_length_matches_max_class_id_plus_one(self):
        lookup = build_group_lookup(max_class_id=50)
        assert lookup.shape == (51,)


class TestGroupsMask:
    def test_selects_the_union_of_the_requested_groups(self):
        labels = np.array([0, 1, 7, 13, 41], dtype=np.uint8)  # unlabeled,noise,ground,dynamic,static
        mask = groups_mask(labels, ["ground", "dynamic"])
        assert list(mask) == [False, False, True, True, False]

    def test_matches_group_of_computed_one_at_a_time(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 43, size=500).astype(np.uint8)
        wanted = ["ground", "static"]
        mask = groups_mask(labels, wanted)
        expected = np.array([group_of(int(c)) in wanted for c in labels])
        assert np.array_equal(mask, expected)

    def test_unknown_group_name_is_rejected(self):
        with pytest.raises(ValueError, match="unknown group name"):
            groups_mask(np.array([7], dtype=np.uint8), ["bogus"])

    def test_an_empty_group_list_selects_nothing(self):
        labels = np.array([7, 13], dtype=np.uint8)
        mask = groups_mask(labels, [])
        assert not mask.any()

    def test_accepts_a_precomputed_lookup_table(self):
        lookup = build_group_lookup()
        labels = np.array([7, 13], dtype=np.uint8)
        assert np.array_equal(
            groups_mask(labels, ["ground"], lookup), groups_mask(labels, ["ground"])
        )
