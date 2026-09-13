import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from stage_manager import StageManager
from state_finder import find_lower_right_green_action


class PrestigeButtonDetectionTests(unittest.TestCase):
    def _frame_with_button(self, width, height, left, top, right, bottom):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # RGB representation of a Brawl Stars-style bright green button.
        cv2.rectangle(frame, (left, top), (right, bottom), (30, 210, 70), -1)
        return frame

    def test_detects_lower_left_lets_go_button(self):
        frame = self._frame_with_button(1920, 1080, 250, 760, 740, 860)
        center = find_lower_right_green_action(frame)
        self.assertIsNotNone(center)
        self.assertAlmostEqual(center[0], 495, delta=8)
        self.assertAlmostEqual(center[1], 810, delta=8)

    def test_detects_scaled_lower_right_button(self):
        frame = self._frame_with_button(1280, 720, 790, 500, 1150, 570)
        center = find_lower_right_green_action(frame)
        self.assertIsNotNone(center)
        self.assertAlmostEqual(center[0], 970, delta=8)
        self.assertAlmostEqual(center[1], 535, delta=8)

    def test_rejects_small_green_game_element(self):
        frame = self._frame_with_button(1920, 1080, 1300, 700, 1380, 760)
        self.assertIsNone(find_lower_right_green_action(frame))


class PrestigeClickTests(unittest.TestCase):
    def test_clicks_detected_live_button(self):
        manager = StageManager.__new__(StageManager)
        manager.runtime_control = None
        manager._last_prestige_press = 0.0
        manager.window_controller = Mock()
        manager.window_controller.screenshot.return_value = np.zeros(
            (1080, 1920, 3), dtype=np.uint8
        )
        with patch("stage_manager.find_lower_right_green_action", return_value=(500, 810)):
            manager.click_prestige_continue()
        manager.window_controller.release_movement.assert_called_once()
        manager.window_controller.click.assert_called_once_with(500, 810, delay=0.08)
        manager.window_controller.press.assert_not_called()


if __name__ == "__main__":
    unittest.main()
