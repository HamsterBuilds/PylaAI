import unittest
from unittest.mock import patch
import time
import numpy as np
import cv2
import state_finder as state
from play import Play
from unittest.mock import Mock
from detect import _numpy_nms


class SuppressionTests(unittest.TestCase):
    def test_capped_suppression_matches_full_prefix(self):
        rng = np.random.default_rng(7)
        for count in (0, 1, 65, 250):
            boxes = rng.uniform(0, 640, (count, 4)).astype(np.float32)
            boxes[:, 2:] = boxes[:, :2] + rng.uniform(1, 100, (count, 2))
            scores = rng.random(count).astype(np.float32)
            for threshold in (0.0, 0.6, 1.0):
                expected = _numpy_nms(boxes, scores, threshold)[:64]
                actual = _numpy_nms(boxes, scores, threshold, max_output=64)
                np.testing.assert_array_equal(actual, expected)


class MatchingTests(unittest.TestCase):
    def test_templates_share_one_region_snapshot(self):
        state._match_cache.history = state.OrderedDict()
        state._match_cache.history_bytes = 0
        first_key = ('first', 320, 180, (0, 0, 320, 180))
        second_key = ('second', 320, 180, (0, 0, 320, 180))
        second_template = self.crop[60:90, 20:60].copy()
        state._template_score(self.crop, self.crop, self.template, first_key)
        state._template_score(self.crop, self.crop, second_template, second_key)
        self.assertEqual(state._match_cache.history_bytes, self.crop.nbytes)
        with patch.object(cv2, 'matchTemplate', wraps=cv2.matchTemplate) as match:
            frame = self.crop.copy()
            state._template_score(frame, frame, self.template, first_key)
            state._template_score(frame, frame, second_template, second_key)
            match.assert_not_called()

    def test_identical_pixels_in_new_frame_reuse_score(self):
        with patch.object(cv2, 'matchTemplate', wraps=cv2.matchTemplate) as match:
            first = state._template_score(self.crop, self.crop, self.template, 'copy')
            frame = self.crop.copy()
            second = state._template_score(frame, frame, self.template, 'copy')
            self.assertEqual(first, second)
            self.assertEqual(match.call_count, 1)

    def setUp(self):
        for name in ("frame_ref", "crops", "scores", "result_buffers"):
            if hasattr(state._match_cache, name):
                delattr(state._match_cache, name)
        rng = np.random.default_rng(12)
        self.crop = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
        self.template = self.crop[30:70, 50:100].copy()

    def test_unchanged_region_skips_matching(self):
        with patch.object(cv2, 'matchTemplate', wraps=cv2.matchTemplate) as match:
            first = state._template_score(
                self.crop, self.crop, self.template, 'a'
            )
            second = state._template_score(
                self.crop, self.crop, self.template, 'a'
            )
            self.assertEqual(first, second)
            self.assertEqual(match.call_count, 1)

    def test_changed_pixels_recompute_exact_score(self):
        state._template_score(self.crop, self.crop, self.template, 'a')
        self.crop[30:70, 50:100] = 0
        changed_frame = self.crop.copy()
        expected = cv2.minMaxLoc(cv2.matchTemplate(changed_frame, self.template, cv2.TM_CCOEFF_NORMED))[1]
        self.assertEqual(
            state._template_score(
                changed_frame, changed_frame, self.template, 'a'
            ),
            expected,
        )

    def test_result_buffer_is_reused_for_same_shape(self):
        first = state._match_result_buffer(141, 271)
        second = state._match_result_buffer(141, 271)
        self.assertIs(first, second)


class PlayWorkTests(unittest.TestCase):
    def make_play(self, data):
        play = Mock()
        play.get_main_data.return_value = data
        play.time_since_walls_checked = 0
        play.walls_treshold = 0.2
        play.last_walls_data = []
        play.last_bushes_data = []
        play.validate_game_data.return_value = False
        play.time_since_player_last_found = time.time()
        play.time_since_last_proceeding = time.time()
        play.no_detection_proceed_delay = 8
        play.wall_scene_gate.should_refresh.return_value = (True, None)
        play.wall_inference_count = 0
        play.wall_cache_count = 0
        play._wall_navigation_ready = False
        play._wall_navigation_initialized = False
        play._wall_navigation_grace_started_at = 0.0
        play.wall_navigation_maximum_age = 1.2
        return play

    def test_absent_player_skips_wall_network(self):
        play = self.make_play({})
        Play.main(play, None, 'test', Mock())
        play.get_tile_data.assert_not_called()
        self.assertEqual(play.time_since_walls_checked, 0)

    def test_returning_player_immediately_refreshes_walls(self):
        play = self.make_play({'player': [[1, 2, 3, 4]]})
        play.process_tile_data.return_value = (['wall'], ['bush'])
        Play.main(play, None, 'test', Mock())
        play.get_tile_data.assert_called_once()
        self.assertEqual(play.last_walls_data, ['wall'])
        self.assertGreater(play.time_since_walls_checked, 0)

    def test_disabled_debug_does_not_build_payload(self):
        play = Mock()
        play.window_controller.debug_view.enabled = False
        Play.publish_debug_view(play, None, None, 'match')
        play.window_controller.debug_view.publish.assert_not_called()

    def test_empty_player_list_is_not_valid_game_data(self):
        data = {"player": [], "enemy": [], "teammate": []}
        self.assertFalse(Play.validate_game_data(data))

    def test_malformed_player_box_is_not_valid_game_data(self):
        data = {"player": [[], [1, 2, 3]], "enemy": []}
        self.assertFalse(Play.validate_game_data(data))

    def test_fresh_raw_player_can_gate_redundant_state_scan(self):
        play = Mock()
        play._raw_player_observed_at = time.monotonic()
        self.assertTrue(Play.has_fresh_raw_player(play))
        play._raw_player_observed_at = time.monotonic() - 0.6
        self.assertFalse(Play.has_fresh_raw_player(play))




if __name__ == '__main__':
    unittest.main()
