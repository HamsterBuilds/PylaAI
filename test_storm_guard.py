import math
import unittest
from unittest.mock import Mock
from play import Play
from pathfinding import LocalPathPlanner


class StormGuardTests(unittest.TestCase):
    def make_play(self):
        play = Mock()
        play.navigation_player_radius = 2
        play.window_controller.scale_factor = 1
        play._poison_danger_regions = ((-100, -100, 100, -10),)
        play.get_player_position.return_value = (0, 0)
        play.is_path_blocked.return_value = False
        play.rotate_movement.side_effect = lambda vector, angle: (
            vector[0] * math.cos(angle) - vector[1] * math.sin(angle),
            vector[0] * math.sin(angle) + vector[1] * math.cos(angle))
        play._movement_poison_risk.side_effect = lambda v: max(0, -v[1])
        return play

    def test_upward_move_is_redirected_outside_storm(self):
        play = self.make_play()
        result = Play.guard_storm_movement(play, [0, 0, 1, 1], (0, -10), [], 30)
        end = tuple(value / math.hypot(*result) * 30 for value in result)
        self.assertTrue(LocalPathPlanner._segment_clear((0, 0), end,
                                                      play._poison_danger_regions))

    def test_clear_move_is_preserved(self):
        play = self.make_play()
        self.assertEqual(Play.guard_storm_movement(play, [], (0, 10), [], 30), (0, 10))

    def test_character_edge_cannot_enter_storm(self):
        play = self.make_play()
        play.navigation_player_radius = 5
        play._poison_danger_regions = ((-100, -100, 100, -33),)
        result = Play.guard_storm_movement(play, [], (0, -10), [], 30)
        self.assertNotEqual(result, (0, -10))

    def test_escape_from_inside_storm_is_preserved(self):
        play = self.make_play()
        play.get_player_position.return_value = (0, -20)
        self.assertEqual(Play.guard_storm_movement(play, [], (0, 10), [], 30), (0, 10))

    def test_no_clear_alternative_stops(self):
        play = self.make_play()
        play.is_path_blocked.return_value = True
        self.assertEqual(Play.guard_storm_movement(play, [], (0, -10), [], 30), (0, 0))
