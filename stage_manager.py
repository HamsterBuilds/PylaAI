import sys
import time
import cv2
from concurrent.futures import ThreadPoolExecutor

from state_finder import (
    find_lower_right_green_action, get_state, is_in_lobby, is_underdog,
)
from trophy_observer import TrophyObserver, MatchResult
from utils import find_template_center, load_toml_as_dict, notify_user, save_brawler_data

def load_image(image_path, scale_factor):
    image = cv2.imread(image_path)
    if image is None:
        return None
    orig_height, orig_width = image.shape[:2]

    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)

    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


class StageManager:
    def __init__(self, brawlers_data, lobby_automator, window_controller, playstyle_info, state_getting, runtime_control=None):
        self.Lobby_automation = lobby_automator
        self.lobby_config = load_toml_as_dict("./cfg/lobby_config.toml")
        self.close_popup_icon = None
        self.brawlers_pick_data = brawlers_data
        self.Trophy_observer = TrophyObserver()
        self.time_since_last_stat_change = time.time()
        self.play_again_on_win = load_toml_as_dict("./cfg/bot_config.toml")["play_again_on_win"] == "yes"
        self.window_controller = window_controller
        self.states = {
            'shop': lambda: self.quit_shop('shop'),
            'brawler_selection': lambda: self.quit_shop('brawler_selection'),
            'popup': self.close_pop_up,
            'match': lambda: 0,
            'match_making': lambda: 0,
            'lobby': self.start_game,
            'star_drop_regular': lambda: self.click_star_drop("regular"),
            'star_drop_angelic': lambda: self.click_star_drop("angelic"),
            'star_drop_demonic': lambda: self.click_star_drop("demonic"),
            'star_drop_starr_nova': lambda: self.click_star_drop("starr_nova"),
            'trophy_reward': lambda: self.window_controller.press("proceed"),
            'prestige_milestone': self.click_prestige_continue,
            'end_draw': lambda: self.end_game('end_draw'),
            'end_victory': lambda: self.end_game('end_victory'),
            'end_defeat': lambda: self.end_game('end_defeat'),
            'end_trio_showdown_0': lambda: self.end_game('end_trio_showdown_0'),
            'end_trio_showdown_1': lambda: self.end_game('end_trio_showdown_1'),
            'end_trio_showdown_2': lambda: self.end_game('end_trio_showdown_2'),
            'end_trio_showdown_3': lambda: self.end_game('end_trio_showdown_3'),
        }
        self.matches_since_last_webhook_ping = 0
        self.ping_every_x_match = load_toml_as_dict("cfg/webhook_config.toml")['ping_every_x_match']
        self.runtime_control = runtime_control
        self.ping_when_stuck = load_toml_as_dict("cfg/webhook_config.toml")["ping_when_stuck"]
        self.playstyle_info = playstyle_info
        self.get_latest_state = state_getting
        self._trophy_api_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pyla-trophy-api"
        )
        self._pending_lobby_trophy_reads = None
        self._pending_lobby_trophy_deadline = 0.0
        self._last_completed_result = None
        self._last_lobby_play_press = 0.0
        self._last_end_handled_at = 0.0
        self._last_end_handled_state = None
        self._last_prestige_press = 0.0

    def _should_stop(self):
        return bool(self.runtime_control and self.runtime_control.should_stop())

    def _should_pause(self):
        return bool(self.runtime_control and self.runtime_control.should_pause())

    def _sleep_interruptible(self, duration, allow_pause=True, poll_interval=0.1):
        end_time = time.time() + duration
        while time.time() < end_time:
            if self._should_stop():
                return True
            if allow_pause and self._should_pause():
                return True
            time.sleep(min(poll_interval, max(end_time - time.time(), 0)))
        return False

    def _start_lobby_trophy_scan(self, first_frame=None, first_frame_time=0.0,
                                 result_filter=None):
        """Trophies are estimated locally from the detected match result."""
        return None

    def _finish_lobby_trophy_scan(self, wait=True):
        return self.Trophy_observer.current_trophies

    def _scan_selected_brawler_trophies(self):
        """Return the locally tracked estimate without network or OCR work."""
        return self.Trophy_observer.current_trophies

    def sync_selected_brawler_trophies(self):
        """Public selection hook used by the initial and later queue picks."""
        return self._scan_selected_brawler_trophies()

    @staticmethod
    def validate_trophies(trophies_string):
        trophies_string = trophies_string.lower()
        while "s" in trophies_string:
            trophies_string = trophies_string.replace("s", "5")
        numbers = ''.join(filter(str.isdigit, trophies_string))

        if not numbers:
            return False

        trophy_value = int(numbers)
        return trophy_value

    def start_game(self):
        if self._should_stop() or self._should_pause():
            return

        # State actions are intentionally checked often for fast navigation.
        # Do not fire the same coordinate repeatedly while the Play animation
        # is transitioning away from the lobby.
        now = time.monotonic()
        if now - self._last_lobby_play_press < 1.0:
            return

        lobby_frame, lobby_frame_time = self.window_controller.screenshot(
            with_timestamp=True
        )
        if not is_in_lobby(lobby_frame):
            print("Skipping stale lobby action: current frame is not lobby.")
            return

        print("state is lobby, starting game")
        current_entry = self.brawlers_pick_data[0]
        type_of_push = current_entry['type']
        push_current_brawler_till = current_entry['push_until']
        values = {
            "trophies": self.Trophy_observer.current_trophies,
            "wins": self.Trophy_observer.current_wins
        }

        value = values[type_of_push]

        if value >= push_current_brawler_till:
            if len(self.brawlers_pick_data) <= 1:
                print("Brawler reached required trophies/wins. No more brawlers selected for pushing in the menu. "
                      "Bot will now pause itself until closed.", value, push_current_brawler_till)
                screenshot = self.window_controller.screenshot()
                notify_user("completed", screenshot, self)
                print("Bot stopping: all targets completed with no more brawlers.")
                self.window_controller.release_movement()
                self.window_controller.close()
                sys.exit(0)
            ping_when_target_is_reached = load_toml_as_dict("cfg/webhook_config.toml")["ping_when_target_is_reached"]
            if ping_when_target_is_reached:
                screenshot = self.window_controller.screenshot()
                notify_user("brawler_goal", screenshot, self)
            print(f'Bot has reached the target trophies/wins for {self.brawlers_pick_data[0]["brawler"]}, moving on to the next one in the list.', value, push_current_brawler_till)
            self.brawlers_pick_data.pop(0)
            next_brawler_name = self.brawlers_pick_data[0]['brawler']
            if self.brawlers_pick_data[0]["automatically_pick"]:
                select_brawler = self.Lobby_automation.select_brawler(next_brawler_name, self.get_latest_state, runtime_control=self.runtime_control)
                while select_brawler in ["failed", "error"]:
                    if self.ping_when_stuck:
                        screenshot = self.window_controller.screenshot()
                        notify_user("bot_failed_brawler_selection", screenshot, self)
                        print(f"Skipping {select_brawler}")
                    if self._should_stop() or self._should_pause():
                        return
                    current_brawler = self.brawlers_pick_data.pop(0)
                    self.brawlers_pick_data.append(current_brawler)
                    next_brawler_name = self.brawlers_pick_data[0]['brawler']
                    self.quit_shop()
                    select_brawler = self.Lobby_automation.select_brawler(next_brawler_name, self.get_latest_state, runtime_control=self.runtime_control)
                if select_brawler == "aborted" or select_brawler == "stuck":
                    return
                if select_brawler == "success":
                    confirmed = self.Trophy_observer.select_brawler(
                        self.brawlers_pick_data[0]['brawler'],
                        self.brawlers_pick_data[0]['trophies'],
                    )
                    self.brawlers_pick_data[0]['trophies'] = confirmed
                    self.Trophy_observer.current_wins = self.brawlers_pick_data[0]['wins'] if self.brawlers_pick_data[0]['wins'] != "" else 0
                    self.Trophy_observer.win_streak = self.brawlers_pick_data[0]['win_streak']
                    self._scan_selected_brawler_trophies()
            else:
                confirmed = self.Trophy_observer.select_brawler(
                    self.brawlers_pick_data[0]['brawler'],
                    self.brawlers_pick_data[0]['trophies'],
                )
                self.brawlers_pick_data[0]['trophies'] = confirmed
                self.Trophy_observer.current_wins = self.brawlers_pick_data[0]['wins'] if self.brawlers_pick_data[0]['wins'] != "" else 0
                self.Trophy_observer.win_streak = self.brawlers_pick_data[0]['win_streak']
                self._scan_selected_brawler_trophies()
                print("Next brawler is in manual mode, waiting 10 seconds to let user switch.")
                if self._sleep_interruptible(10):
                    return
        save_brawler_data(self.brawlers_pick_data)
        self.matches_since_last_webhook_ping += 1
        if self.ping_every_x_match and self.matches_since_last_webhook_ping >= self.ping_every_x_match:
            screenshot = self.window_controller.screenshot()
            notify_user("regular_matches_ping", screenshot, self)
            self.matches_since_last_webhook_ping = 0

        if self._should_stop() or self._should_pause():
            return
        # API and queue updates may span a screen transition.
        # Refresh the lobby frame before choosing where to click.
        if not is_in_lobby(self.window_controller.screenshot()):
            print("Lobby changed during trophy synchronization; skipping stale Play press.")
            return
        self.window_controller.release_movement()
        self.window_controller.press("proceed")
        self._last_lobby_play_press = time.monotonic()
        print("Pressed lobby Play button")
        self._sleep_interruptible(0.25)

    def click_star_drop(self, drop_type="regular"):
        if hasattr(self, '_star_drop_thread') and self._star_drop_thread.is_alive():
            return

        def _handle_drop():
            if drop_type in ["angelic", "demonic", "starr_nova"]:
                self.window_controller.press("proceed", 8)
            else:
                for _ in range(8):
                    self.window_controller.press("proceed", 0.05)
                    time.sleep(0.1)

        import threading
        self._star_drop_thread = threading.Thread(target=_handle_drop, daemon=True)
        self._star_drop_thread.start()

    def click_prestige_continue(self):
        if self._should_stop() or self._should_pause():
            return
        now = time.monotonic()
        if now - self._last_prestige_press < 1.0:
            return

        # Reward animations can finish between state recognition and this
        # action. Use a fresh frame and briefly retry until the live button is
        # clickable instead of pressing an old hard-coded location.
        for attempt in range(4):
            screenshot = self.window_controller.screenshot()
            button_center = find_lower_right_green_action(screenshot)
            if button_center is not None:
                self.window_controller.release_movement()
                print(
                    "Prestige reward detected, pressing LET'S GO at "
                    f"{button_center}."
                )
                self.window_controller.click(*button_center, delay=0.08)
                self._last_prestige_press = time.monotonic()
                return
            if attempt < 3 and self._sleep_interruptible(0.15):
                return

        # The bundled template represents the older CONTINUE layout. Keep its
        # scaled coordinate fallback for that screen only; LET'S GO is always
        # clicked from its detected live bounds above.
        print("Prestige button template detected; using legacy continue point.")
        self.window_controller.press("continue_or_equip", delay=0.08)
        self._last_prestige_press = time.monotonic()

    def end_game(self, detected_state=None):
        screenshot = self.window_controller.screenshot()
        current_state = detected_state or get_state(screenshot)
        end_screen_time = time.time()
        if not current_state.startswith("end"):
            return
        now = time.monotonic()
        if (
            current_state == self._last_end_handled_state
            and now - self._last_end_handled_at < 20.0
        ):
            print("Ignoring duplicate end-screen action for the same round.")
            return
        self._last_end_handled_state = current_state
        self._last_end_handled_at = now

        # The state transition itself guarantees this is a new result screen;
        # the old 25-second guard delayed short rounds and could skip play-again.
        raw_found_result = '_'.join(current_state.split("_")[1:])
        parsed_result = self.Trophy_observer.parse_game_result(raw_found_result)
        self._last_completed_result = parsed_result.result.value
        current_brawler = self.brawlers_pick_data[0]['brawler']
        underdog = is_underdog(screenshot)
        if underdog:
            print("Underdog detected for this match.")

        # This result screen shows rank/player statistics, not the selected
        # brawler's total. Reading a total here produced false candidates or
        # guaranteed failure. The first stable lobby frame is authoritative.
        use_play_again = (
            self.play_again_on_win
            and parsed_result.result == MatchResult.VICTORY
            and not self._should_pause()
            and not self._should_stop()
        )
        button_name = "play_again" if use_play_again else "proceed"
        print("Game has ended, pressing", button_name)
        self.window_controller.press(button_name)
        last_button_press = time.monotonic()

        while current_state.startswith("end") and time.time() - end_screen_time < 20:
            if self._should_stop() or self._should_pause():
                return
            if self._sleep_interruptible(0.25):
                return
            screenshot, screenshot_time = self.window_controller.screenshot(
                with_timestamp=True
            )
            current_state = get_state(screenshot, previous_state=current_state)
            if time.monotonic() - last_button_press >= 1.5:
                # Never repeat this coordinate based on sticky state alone:
                # on the lobby the same position is the Play button and would
                # skip the authoritative post-match trophy scan.
                raw_state = get_state(screenshot, previous_state=None)
                if raw_state.startswith("end"):
                    self.window_controller.press(button_name)
                    last_button_press = time.monotonic()
                else:
                    current_state = raw_state

        self.Trophy_observer.add_trophies(
            parsed_result,
            current_brawler,
            self.playstyle_info,
            underdog,
            power_level=None,
            observed_trophies=None,
            calculate_when_missing=True,
        )
        self.Trophy_observer.add_win(parsed_result)
        self.time_since_last_stat_change = time.time()
        values = {
            "trophies": self.Trophy_observer.current_trophies,
            "wins": self.Trophy_observer.current_wins,
        }
        type_to_push = self.brawlers_pick_data[0]['type']
        self.brawlers_pick_data[0][type_to_push] = values[type_to_push]
        self.brawlers_pick_data[0]['win_streak'] = self.Trophy_observer.win_streak
        save_brawler_data(self.brawlers_pick_data)

        if current_state == "lobby" and not use_play_again:
            self._last_lobby_play_press = 0.0

        if use_play_again:
            print("Waiting for match to start...")
            start_wait_time = time.time()
            interrupted = False
            while time.time() - start_wait_time < 25:
                if self._should_stop() or self._should_pause():
                    interrupted = True
                    break
                screenshot = self.window_controller.screenshot()
                current_state = get_state(screenshot)
                if current_state == "match":
                    print("Match started successfully!")
                    return
                if self._sleep_interruptible(0.5):
                    interrupted = True
                    break

            if interrupted:
                print("Play-again wait interrupted by stop or pause; skipping game restart.")
                return
            print("Match did not start within 25s, restarting the game.")
            self.window_controller.restart_brawl_stars()
            time.sleep(2)
        elif time.time() - end_screen_time > 20:
            print("End screen timeout reached, restarting the game.")
            self.window_controller.restart_brawl_stars()
        print("Game has ended", current_state)

    def quit_shop(self, expected_state=None):
        if expected_state is not None:
            screenshot = self.window_controller.screenshot()
            # This coordinate is the back arrow in menus but the profile area
            # in the lobby, so lobby detection always has veto priority.
            if is_in_lobby(screenshot):
                print(f"Skipping stale {expected_state} action on lobby screen.")
                return
            fresh_state = get_state(screenshot, previous_state=expected_state)
            if fresh_state != expected_state:
                print(
                    f"Skipping stale {expected_state} back-button action; "
                    f"screen is now {fresh_state}."
                )
                return
        self.window_controller.click(100 * self.window_controller.width_ratio, 60 * self.window_controller.height_ratio)
        time.sleep(1)

    def close_pop_up(self):
        screenshot = self.window_controller.screenshot()
        if self.close_popup_icon is None:
            self.close_popup_icon = load_image("images/states/close_popup.png", self.window_controller.scale_factor)
        if self.close_popup_icon is None:
            return
        popup_location = find_template_center(screenshot, self.close_popup_icon)
        if popup_location:
            self.window_controller.click(*popup_location)

    def do_state(self, state, data=None):
        action = self.states.get(state)
        if action is None:
            return
        if data is not None:
            action(data)
            return
        action()
