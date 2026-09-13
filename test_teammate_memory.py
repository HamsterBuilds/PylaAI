import math
import time
import unittest
from unittest.mock import Mock
from play import Play


class TeammateMemoryTests(unittest.TestCase):
    def test_passed_enemy_memory_is_retired(self):
        play = object.__new__(Play)
        play.objective_position = (100, 100)
        play.objective_expires_at = time.time() + 10
        play._objective_approach_vector = (60, 0)
        self.assertIsNone(play.get_recent_objective_movement((160, 100)))
        self.assertIsNone(play.objective_position)

    def make_play(self):
        play = Mock()
        play._decision_cache = {}
        play.last_teammate_position = (100.0, 100.0)
        play.last_teammate_direction = (1.0, 0.0)
        play.last_teammate_seen_at = time.time()
        play.teammate_memory_duration = 8
        play.TILE_SIZE = 54
        play.window_controller.scale_factor = 1
        play._formation_side = 1
        play.get_distance.side_effect = math.dist
        return play

    def test_passed_unseen_teammate_does_not_pull_back(self):
        play = self.make_play()
        target, _ = Play.find_closest_teammate(play, [], (160, 100), [])
        self.assertIsNone(target)
        self.assertIsNone(play.last_teammate_position)

    def test_unseen_teammate_ahead_remains_target(self):
        play = self.make_play()
        target, distance = Play.find_closest_teammate(play, [], (40, 100), [])
        self.assertEqual(target, (100, 100))
        self.assertEqual(distance, 60)

    def test_visible_teammate_behind_is_not_discarded(self):
        play = self.make_play()
        play.target_match_distance = 100
        play.target_switch_ratio = 1.18
        play.get_player_position.return_value = (90, 100)
        target, _ = Play.find_closest_teammate(play, [[80, 80, 100, 110]], (160, 100), [])
        self.assertEqual(target, (90, 100))
