"""The small pieces of app/dashboard.py that are plain Python logic rather
than Streamlit wiring, tested directly with pytest.

The autoplay loop itself (Play sets playing=True, then a
time.sleep + st.rerun cycle advances the frame until Pause is clicked) is not
exercised here. streamlit.testing.v1.AppTest runs a script's st.rerun() calls
synchronously within one .run() invocation, with no way for a test to
interject a Pause click mid-loop the way a real browser session's incoming
websocket message can — so calling .run() after clicking Play does not
return; it correctly keeps looping (proving the mechanism works) until the
test's own timeout fires. That is confirmed once, deliberately, with a short
timeout in tests/test_dashboard_apptest.py rather than treated as a bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from dashboard import frame_prefetch_key, next_playback_index  # noqa: E402

from avrmap.config import PreprocessConfig


class TestNextPlaybackIndex:
    def test_advances_by_one_in_the_middle_of_the_sequence(self):
        assert next_playback_index(5, 80) == 6

    def test_loops_back_to_the_start_after_the_last_frame(self):
        assert next_playback_index(79, 80) == 0

    def test_advances_from_the_first_frame(self):
        assert next_playback_index(0, 80) == 1

    def test_a_single_frame_sequence_loops_to_itself(self):
        assert next_playback_index(0, 1) == 0


class TestFramePrefetchKey:
    def test_contains_sequence_and_every_filter(self):
        key = frame_prefetch_key(
            "001",
            PreprocessConfig(
                min_range_m=0.0, max_range_m=100.0,
                z_min_m=-10.0, z_max_m=30.0,
                drop_classes=(0, 1, 2, 3, 4), devices=(0, 1),
            ),
        )
        assert key[0] == "001"
        assert 0.0 in key and 100.0 in key and -10.0 in key and 30.0 in key
        assert (0, 1, 2, 3, 4) in key and (0, 1) in key

    def test_different_filters_change_the_key(self):
        a = frame_prefetch_key("001", PreprocessConfig(max_range_m=100.0))
        b = frame_prefetch_key("001", PreprocessConfig(max_range_m=50.0))
        assert a != b

    def test_different_sequence_changes_the_key(self):
        a = frame_prefetch_key("001", PreprocessConfig(max_range_m=100.0))
        b = frame_prefetch_key("002", PreprocessConfig(max_range_m=100.0))
        assert a != b
