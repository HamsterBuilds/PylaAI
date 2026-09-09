import unittest
from unittest.mock import patch
import time
import numpy as np
import cv2
import state_finder as state
from play import Play
from unittest.mock import Mock


class MatchingTests(unittest.TestCase):
    def setUp(self):
        state._match_cache.entries = state.OrderedDict()
        state._match_cache.bytes = 0
        rng = np.random.default_rng(12)
        self.crop = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
        self.template = self.crop[30:70, 50:100].copy()

    def test_unchanged_region_skips_matching(self):
        with patch.object(cv2, 'matchTemplate', wraps=cv2.matchTemplate) as match:
            first = state._template_score(self.crop, self.template, 'a')
            second = state._template_score(self.crop.copy(), self.template, 'a')
            self.assertEqual(first, second)
            self.assertEqual(match.call_count, 1)

    def test_changed_pixels_recompute_exact_score(self):
        state._template_score(self.crop, self.template, 'a')
        self.crop[30:70, 50:100] = 0
        expected = cv2.minMaxLoc(cv2.matchTemplate(self.crop, self.template, cv2.TM_CCOEFF_NORMED))[1]
        self.assertEqual(state._template_score(self.crop, self.template, 'a'), expected)

    def test_memory_and_entry_bounds(self):
        for i in range(100):
            state._template_score(self.crop, self.template, i)
        self.assertLessEqual(state._match_cache.bytes, state._MATCH_CACHE_BYTES)
        self.assertLessEqual(len(state._match_cache.entries), state._MATCH_CACHE_ENTRIES)
        self.assertEqual(state._match_cache.bytes, sum(item[0].nbytes for item in state._match_cache.entries.values()))


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


if __name__ == '__main__':
    unittest.main()
