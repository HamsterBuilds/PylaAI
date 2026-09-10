import os
import time

import cv2
import numpy as np
from utils import (
    load_toml_as_dict, config_bool, load_brawlers_info,
    normalize_brawler_filename,
)


class LobbyAutomation:

    def __init__(self, window_controller):
        self.gray_pixels_treshold = load_toml_as_dict("./cfg/bot_config.toml").get('idle_pixels_minimum', 500)
        self.idle_reconnect_coords = load_toml_as_dict("cfg/buttons_config.toml")["idle_reconnect"]
        if self.idle_reconnect_coords and isinstance(self.idle_reconnect_coords[0], (int, float)):
            self.idle_reconnect_coords = [self.idle_reconnect_coords]
        self.window_controller = window_controller
        self.verbose_debug = config_bool(load_toml_as_dict("cfg/debug_settings.toml").get('verbose_debug'), False)
        self.idle_disconnect_hsv_high_bounds = load_toml_as_dict("cfg/lobby_config.toml").get("hsv_bounds", {}).get("idle_reconnect_high_bounds", [[10, 22, 42], [10, 22, 90], [118, 66, 46]])
        self._idle_crop_scale = None
        self._idle_crop_coordinates = None
        self._idle_buffer_shape = None
        self._idle_hsv_buffer = None
        self._idle_mask_buffer = None

    def check_for_idle(self, frame):
        wr = self.window_controller.width_ratio
        hr = self.window_controller.height_ratio
        scale = (wr, hr)
        if scale != self._idle_crop_scale:
            self._idle_crop_scale = scale
            self._idle_crop_coordinates = (
                int(460 * wr), int(1460 * wr),
                int(400 * hr), int(675 * hr),
            )
        x_start, x_end, y_start, y_end = self._idle_crop_coordinates
        idle_crop = frame[y_start:y_end, x_start:x_end]
        if idle_crop.size == 0:
            self.window_controller.reset_to_default_resolution()
            return
        crop_shape = idle_crop.shape
        if crop_shape != self._idle_buffer_shape:
            self._idle_buffer_shape = crop_shape
            self._idle_hsv_buffer = np.empty(crop_shape, dtype=np.uint8)
            self._idle_mask_buffer = np.empty(crop_shape[:2], dtype=np.uint8)
        try:
            hsv_crop = cv2.cvtColor(
                idle_crop, cv2.COLOR_RGB2HSV, dst=self._idle_hsv_buffer
            )
        except cv2.error as error:
            print(f"Idle detection color conversion failed: {error}")
            self.window_controller.reset_to_default_resolution()
            return
        if self.verbose_debug:
            print(f"gray pixels (if > {self.gray_pixels_treshold} then bot will try to unidle)")
        for idle_disconnect_hsv_high_bound in self.idle_disconnect_hsv_high_bounds:
            mask = cv2.inRange(
                hsv_crop, (0, 0, 0), tuple(idle_disconnect_hsv_high_bound),
                dst=self._idle_mask_buffer,
            )
            gray_pixels = cv2.countNonZero(mask)
            if self.verbose_debug:
                try:
                    cv2.imwrite(f"./debug_frames/idle_detection_{gray_pixels}_{len(os.listdir('./debug_frames'))}.png", cv2.cvtColor(idle_crop, cv2.COLOR_BGR2RGB))
                except Exception:
                    pass
            if gray_pixels > self.gray_pixels_treshold:
                print("Idle detected, clicking to unidle")
                for idle_reconnect_coord in self.idle_reconnect_coords:
                    self.window_controller.click(idle_reconnect_coord[0], idle_reconnect_coord[1], already_include_ratio=False)


    @staticmethod
    def _should_interrupt(runtime_control=None, stop_event=None):
        if runtime_control and (runtime_control.should_stop() or runtime_control.should_pause()):
            return True
        return stop_event is not None and stop_event.is_set()

    @staticmethod
    def _sleep_interruptible(duration, runtime_control=None, stop_event=None, poll_interval=0.1):
        end_time = time.time() + duration
        while time.time() < end_time:
            if LobbyAutomation._should_interrupt(runtime_control, stop_event):
                return True
            time.sleep(min(poll_interval, max(end_time - time.time(), 0)))
        return False

    def select_brawler(self, brawler, get_latest_state, stop_event=None, runtime_control=None):
        self.window_controller.screenshot()
        wr = self.window_controller.width_ratio
        hr = self.window_controller.height_ratio
        brawler = str(brawler).lower().strip()
        normalized_brawler = normalize_brawler_filename(brawler)
        brawler_info = load_brawlers_info().get(normalized_brawler, {})
        brawler_search_name = brawler_info.get("actual_name") or normalized_brawler

        x, y = load_toml_as_dict("cfg/buttons_config.toml")["brawlers_menu"]
        self.window_controller.click(x, y, already_include_ratio=False)
        time.sleep(1.25)
        print("Automatic brawler selection started for", brawler_search_name)
        for i in range(100):
            if self._should_interrupt(runtime_control, stop_event):
                print("Brawler selection aborted by user.")
                return "aborted"
            self.window_controller.screenshot()
            current_state = get_latest_state()
            if current_state == "shop":
                print("Brawler menu is still opening")
                time.sleep(1)
                continue

            if current_state != "brawler_selection":
                print("Latest screenshot is no longer of the lobby, aborting brawler selection...")
                return "stuck"

            self.window_controller.press("brawler_search")
            if self._sleep_interruptible(1, runtime_control, stop_event):
                print("Brawler selection aborted by user.")
                return "aborted"

            if not self.window_controller.type_text(brawler_search_name):
                print(f"Could not enter brawler name '{brawler_search_name}' in the search field.")
                return "error"
            if self._sleep_interruptible(0.5, runtime_control, stop_event):
                print("Brawler selection aborted by user.")
                return "aborted"

            first_brawler_x, first_brawler_y = load_toml_as_dict("cfg/buttons_config.toml")["first_brawler_icon"]
            self.window_controller.click(first_brawler_x, first_brawler_y, already_include_ratio=False)
            if self._sleep_interruptible(1, runtime_control, stop_event):
                print("Brawler selection aborted by user.")
                return "aborted"

            select_x, select_y = load_toml_as_dict("cfg/buttons_config.toml")["select_brawler"]
            self.window_controller.click(select_x, select_y, already_include_ratio=False)
            if self._sleep_interruptible(1.5, runtime_control, stop_event):
                print("Brawler selection aborted by user.")
                return "aborted"
            self.window_controller.screenshot()
            print("Selected brawler ", brawler_search_name)
            return "success"

        print(f"WARNING: Brawler '{brawler}' was not found after 100 scroll attempts.")
        return "failed"
