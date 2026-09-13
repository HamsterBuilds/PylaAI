import unittest
from types import SimpleNamespace
from pathlib import Path
import cv2
import numpy as np
from play import Play


def detector(frame):
    play = object.__new__(Play)
    play.frame = frame
    play.window_controller = SimpleNamespace(scale_factor=1)
    play.TILE_SIZE = 54
    play.navigation_player_radius = 26
    play.power_cube_detection_interval = 0
    play._power_cube_cache_at = 0
    play._power_cube_cache_player = None
    play._power_cube_buffer_shape = None
    play._power_cube_tracks = []
    play.power_cube_scans = 0
    play.power_cube_cache_hits = 0
    play.power_cube_candidates_rejected = 0
    return play


class PowerCubeTests(unittest.TestCase):
    def test_moderate_saturation_cube_beyond_old_scan_radius(self):
        hsv = np.zeros((1080, 1920, 3), np.uint8)
        hsv[950:1015, 645:695] = (58, 128, 217)
        hsv[975:1000, 645:658] = (25, 180, 240)
        # A similarly sized green patch with no lightning emblem is noise.
        hsv[850:915, 1200:1250] = (55, 108, 229)
        play = detector(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        player = [900, 330, 1010, 600]
        play.detect_power_cubes(player, [player])
        cubes = play.detect_power_cubes(player, [player])
        self.assertEqual(len(cubes), 1)
        self.assertLess(abs(Play.get_entity_pos(cubes[0])[0] - 670), 20)

    def test_real_capture_excludes_character_icons(self):
        path = Path('benchmark_captures/20260912-120221/0000.png')
        if not path.exists():
            self.skipTest('Local private replay is not distributed')
        play = detector(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
        player = [870, 320, 1050, 615]
        entities = [player, [630, 630, 790, 885], [790, 600, 910, 815]]
        play.detect_power_cubes(player, entities)
        cubes = play.detect_power_cubes(player, entities)
        self.assertEqual(len(cubes), 1, cubes)
        x, y = Play.get_entity_pos(cubes[0])
        self.assertTrue(640 < x < 705 and 945 < y < 1025, cubes)
