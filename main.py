import argparse
import inspect
import os
import sys
import tomllib
from pathlib import Path

# Monkey-patch inspect.getfile to prevent Nuitka + PyTorch crash
_original_getfile = inspect.getfile
def _patched_getfile(obj):
    res = _original_getfile(obj)
    return res if res is not None else "<unknown_nuitka_file>"

inspect.getfile = _patched_getfile


if __name__ == "__main__" and len(sys.argv) >= 9 and sys.argv[1] == "--debug-viewer-worker":
    from debug_view import DEFAULT_DEBUG_VIEW_FPS, run_viewer_worker

    run_viewer_worker(
        shared_memory_name=sys.argv[2],
        debug_memory_name=sys.argv[3],
        height=int(sys.argv[4]),
        width=int(sys.argv[5]),
        channels=int(sys.argv[6]),
        dtype_text=sys.argv[7],
        title=sys.argv[8],
        clip_fps=float(sys.argv[9]) if len(sys.argv) >= 10 else DEFAULT_DEBUG_VIEW_FPS,
        record_clips=(len(sys.argv) >= 11 and sys.argv[10] == "1"),
    )
    sys.exit(0)

def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="PylaAI",
        description="PylaAI, the best free and open source brawl stars bot.",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Hide PylaAI's own console and write output to a log file.",
    )
    interface_group = parser.add_mutually_exclusive_group()
    interface_group.add_argument(
        "--desktop",
        dest="interface_mode",
        action="store_const",
        const="desktop",
        help="Force the UI to open in the integrated pywebview window.",
    )
    interface_group.add_argument(
        "--web",
        "--browser",
        "--no-webapp",
        dest="interface_mode",
        action="store_const",
        const="browser",
        help="Force the UI to open in the system browser instead of pywebview.",
    )
    interface_group.add_argument(
        "--headless",
        dest="interface_mode",
        action="store_const",
        const="headless",
        help="Force headless mode: serve the local web UI without opening it.",
    )
    args, _unknown_args = parser.parse_known_args(argv)
    return args


INTERFACE_MODES = frozenset({"desktop", "browser", "headless"})


def _startup_project_root():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent.parent
    return Path.cwd().resolve()


def load_saved_interface_mode(config_path=None):
    path = Path(config_path) if config_path is not None else _startup_project_root() / "cfg" / "general_config.toml"
    try:
        with path.open("rb") as config_file:
            configured_mode = str(tomllib.load(config_file).get("interface_mode", "desktop")).strip().lower()
    except (OSError, tomllib.TOMLDecodeError) as error:
        print(f"Could not read interface_mode from {path}: {error}. Using desktop mode.")
        return "desktop"

    if configured_mode not in INTERFACE_MODES:
        print(f"Unknown interface_mode {configured_mode!r} in {path}. Using desktop mode.")
        return "desktop"
    return configured_mode


def resolve_interface_mode(cli_args, config_path=None):
    return cli_args.interface_mode or load_saved_interface_mode(config_path)


# Parse these before the heavy application imports so console hiding happens as
# early as possible. Imported modules receive harmless default values.
CLI_ARGS = parse_cli_args(sys.argv[1:] if __name__ == "__main__" else [])
INTERFACE_MODE = resolve_interface_mode(CLI_ARGS)
CONSOLE_HIDDEN = False
CONSOLE_LOG_FILE = None

if CLI_ARGS.no_console:
    from desktop import console_log_path, hide_console

    CONSOLE_LOG_FILE = console_log_path()
    CONSOLE_HIDDEN = hide_console(CONSOLE_LOG_FILE)

if CLI_ARGS.no_console and not CONSOLE_HIDDEN:
    print(
        "--no-console ignored: this console belongs to the terminal PylaAI was "
        "started from, so output stays here."
    )


from adbutils import AdbError
import socket
import threading
import time
from lobby_automation import LobbyAutomation
from play import Play
from stage_manager import StageManager
from state_finder import get_state
from time_management import TimeManagement
from utils import load_toml_as_dict, current_wall_model_is_latest, api_base_url, load_pyla_script, save_brawler_data, \
    clean_queue, get_discord_link
from utils import get_brawler_list, update_missing_brawlers_info, check_version, notify_user, update_wall_model_classes, get_latest_wall_model_file, cprint
from window_controller import WindowController
from perception import StateConsensus


def apply_play_order(queue_data):
    play_order = str(load_toml_as_dict("cfg/general_config.toml").get("play_order", "in_order")).strip().lower()
    if play_order == "lowest_to_highest":
        ordered_data = sorted(queue_data, key=lambda item: int(item.get("trophies", 0) or 0))
    elif play_order == "highest_to_lowest":
        ordered_data = sorted(queue_data, key=lambda item: int(item.get("trophies", 0) or 0), reverse=True)
    else:
        return queue_data

    for item in ordered_data:
        item["automatically_pick"] = True
    return ordered_data


def pyla_main(discord_bot, queue_data, stop_event=None, runtime_control=None):
    class Main:
        def __init__(self):
            current_playstyle = load_toml_as_dict("cfg/bot_config.toml").get("current_playstyle", "default_up.pyla")
            try:
                self.max_fps = int(load_toml_as_dict("cfg/general_config.toml")['max_fps'])
            except ValueError:
                self.max_fps = None

            if self.max_fps:
                self.window_controller = WindowController(self.max_fps)
            else:
                self.window_controller = WindowController()
            data = clean_queue(queue_data)
            data = apply_play_order(data)
            if not data:
                raise ValueError("No valid brawler data found. Please add a brawler configuration in the UI before starting the bot.")
            save_brawler_data(data)
            print("Starting with queue data:", data)
            self.playstyle_info, pyla_code = load_pyla_script(current_playstyle)
            self.Play = Play(*self.load_models(), self.window_controller, pyla_code)
            self.Time_management = TimeManagement()
            self.lobby_automator = LobbyAutomation(self.window_controller)
            self.runtime_control = runtime_control
            self.Stage_manager = StageManager(data, self.lobby_automator, self.window_controller, self.playstyle_info, self.get_latest_state, runtime_control=runtime_control)
            self.states_requiring_data = ["lobby"]
            self.no_detections_action_threshold = 60 * 8
            self.state = None
            self.stop_event = stop_event
            self.state_lock = threading.Lock()
            self.latest_state_frame_time = 0.0
            bot_config = load_toml_as_dict("cfg/bot_config.toml")
            self.max_cached_state_age = max(0.25, float(bot_config.get("maximum_state_age", 1.0)))
            self.state_consensus = StateConsensus(
                confirmations=bot_config.get("state_change_confirmations", 2)
            )
            self.state_checker_stop_event = threading.Event()
            self.state_checker_thread = None
            self.update_trophy_observer()

            # The checker thread reads every field below immediately. Create
            # them before starting it so a fast first frame cannot race init.
            time_config = load_toml_as_dict("cfg/time_tresholds.toml")
            self.match_state_check_interval = max(0.1, float(time_config.get("match_state_check_interval", 0.25)))
            self.menu_state_check_interval = max(0.05, float(time_config.get("menu_state_check_interval", 0.10)))
            self.match_probe_interval = max(0.1, float(time_config.get("match_probe_interval", 0.25)))
            self.last_match_probe = 0.0
            self.match_probe_confirmations = 0
            self.full_state_scan_interval = max(
                0.5, float(time_config.get("full_state_scan_interval", 2.0))
            )
            self.last_full_state_scan = 0.0
            configured_entity_fps = str(
                load_toml_as_dict("cfg/general_config.toml").get(
                    "entity_inference_max_fps", "auto"
                )
            ).strip().lower()
            if configured_entity_fps == "auto":
                logical_cpus = os.cpu_count() or 4
                # Preserve fast combat reactions, but stop spending the same
                # inference budget while simply travelling/regrouping.
                entity_fps = 10 if logical_cpus <= 4 else 12 if logical_cpus <= 8 else 16
                idle_entity_fps = 7 if logical_cpus <= 4 else 8 if logical_cpus <= 8 else 12
            else:
                entity_fps = max(4, min(60, int(configured_entity_fps)))
                idle_entity_fps = entity_fps
            if self.max_fps:
                entity_fps = min(entity_fps, self.max_fps)
                idle_entity_fps = min(idle_entity_fps, self.max_fps)
            self.entity_inference_interval = 1.0 / entity_fps
            self.idle_entity_inference_interval = 1.0 / idle_entity_fps
            self.entity_combat_memory = 1.5
            self.last_entity_inference = 0.0
            self.entity_inference_count = 0
            self.entity_skip_count = 0
            self.last_performance_report = time.monotonic()
            if idle_entity_fps == entity_fps:
                print(f"Entity AI capped at {entity_fps} FPS for stable frame pacing.")
            else:
                print(
                    f"Entity AI adaptive: {idle_entity_fps} FPS travelling, "
                    f"{entity_fps} FPS in combat."
                )

            self.run_for_minutes = int(load_toml_as_dict("cfg/general_config.toml")['run_for_minutes'])
            self.webhook_ping_every_minutes = load_toml_as_dict("cfg/webhook_config.toml")['ping_every_x_minutes']
            self.time_since_last_webhook_ping = time.time()
            self.start_time = time.time()
            self.time_to_stop = False
            self.in_cooldown = False
            self.cooldown_start_time = 0
            self.cooldown_duration = 3 * 60
            self.window_controller.screenshot()
            discord_bot.set_window_controller(self.window_controller)
            self.start_state_checker()
            print("Initialization complete, starting main loop.")
            self.picked_first_brawler = False
            self.time_since_checked_if_brawl_stars_crashed = time.time()
            self.check_if_brawl_stars_crashed_timer = load_toml_as_dict("cfg/time_tresholds.toml")["check_if_brawl_stars_crashed"]
            self.ping_when_stuck = load_toml_as_dict("cfg/webhook_config.toml")["ping_when_stuck"]

        def update_trophy_observer(self):
            current_brawler_data = self.Stage_manager.brawlers_pick_data[0]
            self.Stage_manager.Trophy_observer.win_streak = current_brawler_data['win_streak']
            confirmed_trophies = self.Stage_manager.Trophy_observer.select_brawler(
                current_brawler_data['brawler'], current_brawler_data['trophies']
            )
            current_brawler_data['trophies'] = confirmed_trophies
            self.Stage_manager.Trophy_observer.current_wins = current_brawler_data['wins'] if current_brawler_data['wins'] != "" else 0

        @staticmethod
        def load_models():
            folder_path = "./models/"
            return [
                folder_path + 'mainInGameModel.onnx',
                folder_path + 'tileDetector.onnx',
                folder_path + 'closeTileDetector.onnx',
            ]

        def restart_brawl_stars(self):
            self.window_controller.restart_brawl_stars()
            self.time_since_checked_if_brawl_stars_crashed = time.time()
            self.Play.time_since_detections["player"] = time.time()
            self.Play.time_since_detections["enemy"] = time.time()
            if not self.window_controller.is_brawl_stars_running():
                ping_when_stuck = load_toml_as_dict("cfg/webhook_config.toml")["ping_when_stuck"]
                if ping_when_stuck:
                    screenshot = self.window_controller.screenshot()
                    notify_user("bot_is_stuck", screenshot, self.Stage_manager)
                    print("Bot got stuck. User notified.")
                print("Shutting down.")
                self.window_controller.release_movement()
                self.window_controller.close()
                discord_bot.set_window_controller(None)
                sys.exit(1)

        def should_stop(self):
            return bool(self.stop_event and self.stop_event.is_set()) or bool(self.runtime_control and self.runtime_control.should_stop())

        def should_pause(self):
            return bool(self.runtime_control and self.runtime_control.should_pause())

        def sleep_interruptible(self, duration, allow_pause=True, poll_interval=0.1):
            end_time = time.time() + duration
            while time.time() < end_time:
                if self.should_stop():
                    return "stop"
                if allow_pause and self.should_pause():
                    return "pause"
                time.sleep(min(poll_interval, max(end_time - time.time(), 0)))
            return None

        def stop_gracefully(self):
            cprint("Stop requested from UI - shutting down gracefully", "#AAE5A4")
            self.stop_state_checker()
            self.window_controller.release_movement()
            self.window_controller.close()
            discord_bot.set_window_controller(None)

        def start_state_checker(self):
            if self.state_checker_thread and self.state_checker_thread.is_alive():
                return
            self.state_checker_stop_event.clear()
            self.state_checker_thread = threading.Thread(
                target=self.state_checker_loop,
                daemon=True,
                name="pyla-state-checker"
            )
            self.state_checker_thread.start()

        def stop_state_checker(self):
            self.state_checker_stop_event.set()
            if self.state_checker_thread and self.state_checker_thread.is_alive():
                self.state_checker_thread.join(timeout=1.0)

        def set_latest_state(self, state, frame_time=None):
            with self.state_lock:
                self.state = state
                self.latest_state_frame_time = frame_time or time.time()

        def get_latest_state(self):
            with self.state_lock:
                if (
                    self.state is not None
                    and time.time() - self.latest_state_frame_time > self.max_cached_state_age
                ):
                    return None
                return self.state

        def observe_state(self, state, frame_time=None):
            stable_state = self.state_consensus.observe(state)
            if stable_state is not None:
                self.set_latest_state(stable_state, frame_time)
            return stable_state

        def confirm_match_from_entities(self, frame_time):
            self.state_consensus.force("match")
            self.set_latest_state("match", frame_time)

        def handle_detected_state(self, state, observed_at=None):
            if state is None:
                return
            if observed_at is not None:
                self.set_latest_state(state, observed_at)

            print(f"State: {state}")
            frame_data = None
            self.Stage_manager.do_state(state, frame_data)
            if state != "match":
                self.Play.reset_perception()
                self.Play.time_since_last_proceeding = time.time()

        def state_checker_loop(self):
            last_checked_frame_time = 0.0
            while not self.state_checker_stop_event.is_set():
                frame, frame_time = self.window_controller.wait_for_frame(
                    last_checked_frame_time, timeout=0.25
                )
                if frame is None or frame_time <= last_checked_frame_time:
                    continue

                last_checked_frame_time = frame_time
                try:
                    scan_now = time.monotonic()
                    previous_state = self.get_latest_state()
                    if scan_now - self.last_full_state_scan >= self.full_state_scan_interval:
                        previous_state = None
                        self.last_full_state_scan = scan_now
                    self.observe_state(
                        get_state(frame, previous_state=previous_state), frame_time
                    )
                except Exception as e:
                    print(f"State checker failed: {e}")
                # Menus need quick transitions; during a match, entity
                # inference has priority and end-state checks can run at 4 Hz.
                interval = self.match_state_check_interval if self.get_latest_state() == "match" else self.menu_state_check_interval
                self.state_checker_stop_event.wait(interval)

        def wait_while_paused(self):
            if not self.runtime_control:
                return

            self.window_controller.release_movement()
            self.runtime_control.mark_paused()
            cprint("Pyla is paused in the lobby. Waiting for Start to resume.", "#AAE5A4")

            while self.should_pause() and not self.should_stop():
                state = self.get_latest_state()
                if state is None:
                    if self.sleep_interruptible(0.25, allow_pause=False) == "stop":
                        return
                    continue
                if self.sleep_interruptible(1, allow_pause=False) == "stop":
                    return

            if not self.should_stop():
                self.runtime_control.mark_running()
                self.time_since_last_webhook_ping = time.time()
                print("Pause released, resuming run.")

        def handle_pause_request(self):
            if self.should_pause() and not self.should_stop():
                cprint("Pause requested from UI - waiting", "#AAE5A4")
                self.wait_while_paused()

        def manage_time_tasks(self, frame):
            if self.Time_management.state_check():
                state = self.get_latest_state()
                if state is not None:
                    self.handle_detected_state(state)
            if self.Time_management.no_detections_check():
                frame_data = self.Play.time_since_detections
                t_now = time.time()
                for key, value in frame_data.items():
                    if t_now - value > self.no_detections_action_threshold:
                        self.restart_brawl_stars()
            if self.Time_management.idle_check():
                self.lobby_automator.check_for_idle(frame)

            current_time = time.time()
            if self.webhook_ping_every_minutes and current_time - self.time_since_last_webhook_ping >= self.webhook_ping_every_minutes * 60:
                screenshot = self.window_controller.screenshot()
                notify_user("regular_minutes_ping", screenshot, self.Stage_manager)
                self.time_since_last_webhook_ping = current_time
                print(f"Sent regular webhook ping after {self.webhook_ping_every_minutes} minutes.")

        def check_and_handle_brawl_stars_crash(self):
            c_time = time.time()
            if c_time - self.time_since_checked_if_brawl_stars_crashed > self.check_if_brawl_stars_crashed_timer:
                try:
                    opened_app = self.window_controller.device.app_current().package.strip()
                    if not self.window_controller.is_brawl_stars_running(opened_app):
                        print(f"Brawl stars has crashed, {opened_app} is the app opened ! Restarting...")
                        self.window_controller.device.app_start(self.window_controller.BRAWL_STARS_PACKAGE)
                        time.sleep(3)
                        self.time_since_checked_if_brawl_stars_crashed = time.time()
                    else:
                        self.time_since_checked_if_brawl_stars_crashed = c_time
                except AdbError:
                    print("There was an error checking if Brawl Stars is running. Attempting to reconnect scrcpy...")
                    if not self.window_controller.reconnect_scrcpy():
                        print("Reconnect failed -- restarting Brawl Stars")
                        self.restart_brawl_stars()

        def main(self):
            s_time = time.time()
            c = 0
            last_processed_frame_time = 0.0
            self.time_since_last_webhook_ping = time.time()
            if self.runtime_control:
                self.runtime_control.mark_running()

            while True:
                # One synchronized state read per hot-loop pass.  The state checker
                # owns updates, so repeated lock acquisitions here only add jitter.
                loop_state = self.get_latest_state()
                if loop_state == "lobby":
                    if self.should_stop():
                        self.stop_gracefully()
                        break

                    if self.should_pause():
                        self.handle_pause_request()
                        if self.should_stop():
                            self.stop_gracefully()
                            break
                        if self.should_pause():
                            continue
                        # Pausing can span a state transition; do not use the stale
                        # pre-pause snapshot for automatic lobby actions.
                        loop_state = self.get_latest_state()

                if not self.picked_first_brawler and loop_state == "lobby":
                    if self.Stage_manager.brawlers_pick_data[0]['automatically_pick']:
                        next_brawler_name = self.Stage_manager.brawlers_pick_data[0]['brawler']
                        print("Picking brawler automatically")
                        if self.runtime_control:
                            self.runtime_control.mark_running()
                        select_brawler = self.lobby_automator.select_brawler(next_brawler_name, self.get_latest_state, runtime_control=self.runtime_control)

                        while select_brawler in ["failed", "error"]:
                            print("Automatic brawler selection failed.")
                            if self.ping_when_stuck:
                                screenshot = self.window_controller.screenshot()
                                notify_user("bot_failed_brawler_selection", screenshot, self.Stage_manager)
                            failed_brawler = self.Stage_manager.brawlers_pick_data.pop(0)
                            self.Stage_manager.brawlers_pick_data.append(failed_brawler)
                            next_brawler_name = self.Stage_manager.brawlers_pick_data[0]['brawler']
                            select_brawler = self.lobby_automator.select_brawler(next_brawler_name, self.get_latest_state, runtime_control=self.runtime_control)

                        if select_brawler == "aborted" or select_brawler == "stuck":
                            continue
                        self.picked_first_brawler = True
                        self.update_trophy_observer()
                    else:
                        self.picked_first_brawler = True
                t_now = time.time()
                if self.max_fps:
                    frame_start = time.perf_counter()

                if self.run_for_minutes > 0 and not self.in_cooldown:
                    elapsed_time = (t_now - self.start_time) / 60
                    if elapsed_time >= self.run_for_minutes:
                        cprint(f"timer is done, {self.run_for_minutes} is over. continuing for 3 minutes if in game", "#AAE5A4")
                        self.in_cooldown = True
                        self.cooldown_start_time = t_now
                        self.Stage_manager.states['lobby'] = lambda: 0

                if self.in_cooldown and t_now - self.cooldown_start_time >= self.cooldown_duration:
                    cprint("stopping bot fully", "#AAE5A4")
                    self.stop_gracefully()
                    break

                if abs(s_time - t_now) > 1:
                    elapsed = t_now - s_time
                    if elapsed > 0:
                        print(f"{c / elapsed:.2f} FPS")
                    s_time = t_now
                    c = 0
                self.check_and_handle_brawl_stars_crash()
                frame, last_ft = self.window_controller.wait_for_frame(
                    last_processed_frame_time, timeout=0.25
                )
                if frame is None:
                    continue
                if last_ft > 0 and (t_now - last_ft) > self.window_controller.FRAME_STALE_TIMEOUT:
                    stale_age = t_now - last_ft
                    self.Play.window_controller.release_movement()
                    if stale_age > 30:
                        print(f"Scrcpy feed stale for {stale_age:.0f}s -- attempting reconnect")
                        if not self.window_controller.reconnect_scrcpy():
                            print("Reconnect failed -- restarting Brawl Stars")
                            self.restart_brawl_stars()
                    else:
                        print("Stale frame detected -- pausing actions until feed resumes")
                        if self.sleep_interruptible(1) == "stop":
                            self.stop_gracefully()
                            break
                    continue

                # Do not run neural networks repeatedly on the same video frame.
                # Keep stale-feed recovery above this check active.
                if last_ft > 0 and last_ft <= last_processed_frame_time:
                    if self.should_stop():
                        self.stop_gracefully()
                        break
                    continue
                self.manage_time_tasks(frame)

                # State handlers may wait for menus, matchmaking or reconnects.
                # Their input frame is then obsolete: infer on the current feed.
                frame, last_ft = self.window_controller.screenshot(with_timestamp=True)
                if last_ft > 0 and time.time() - last_ft > self.window_controller.FRAME_STALE_TIMEOUT:
                    continue
                last_processed_frame_time = last_ft

                brawler = self.Stage_manager.brawlers_pick_data[0]['brawler']
                self.Play.current_brawler = brawler
                current_state = self.get_latest_state()
                if current_state != "match":
                    if current_state not in (None, "match_making"):
                        self.match_probe_confirmations = 0
                        continue
                    probe_now = time.monotonic()
                    if probe_now - self.last_match_probe < self.match_probe_interval:
                        continue
                    self.last_match_probe = probe_now
                    if self.Play.probe_match_started(frame):
                        self.match_probe_confirmations += 1
                    else:
                        self.match_probe_confirmations = 0
                    if self.match_probe_confirmations < 2:
                        continue
                    self.confirm_match_from_entities(last_ft)
                    self.match_probe_confirmations = 0
                inference_now = time.monotonic()
                enemy_seen_recently = (
                    time.time() - self.Play.time_since_detections.get("enemy", 0.0)
                    <= self.entity_combat_memory
                )
                active_interval = (
                    self.entity_inference_interval if enemy_seen_recently
                    else self.idle_entity_inference_interval
                )
                if inference_now - self.last_entity_inference < active_interval:
                    self.entity_skip_count += 1
                    continue
                self.last_entity_inference = inference_now
                self.entity_inference_count += 1
                self.Play.main(frame, brawler, self)
                c += 1

                if inference_now - self.last_performance_report >= 30.0:
                    entity_total = self.entity_inference_count + self.entity_skip_count
                    wall_total = self.Play.wall_inference_count + self.Play.wall_cache_count
                    planner = self.Play.path_planner
                    entity_saved = 100.0 * self.entity_skip_count / max(1, entity_total)
                    wall_saved = 100.0 * self.Play.wall_cache_count / max(1, wall_total)
                    path_saved = 100.0 * planner.path_cache_hits / max(1, planner.plan_requests)
                    poison_total = self.Play.poison_compute_count + self.Play.poison_cache_count
                    poison_saved = 100.0 * self.Play.poison_cache_count / max(1, poison_total)
                    print(
                        "Performance: "
                        f"entity frames avoided={entity_saved:.1f}%, "
                        f"wall inferences avoided={wall_saved:.1f}%, "
                        f"poison scans avoided={poison_saved:.1f}%, "
                        f"A* cache hits={path_saved:.1f}%, "
                        f"A* nodes={planner.expanded_nodes}"
                    )
                    self.last_performance_report = inference_now

                if self.max_fps:
                    target_period = 1 / self.max_fps
                    work_time = time.perf_counter() - frame_start
                    if work_time < target_period:
                        time.sleep(target_period - work_time)

    os.makedirs("debug_frames", exist_ok=True)
    main = Main()
    main.main()


all_brawlers = get_brawler_list()
if api_base_url != "localhost":
    update_missing_brawlers_info(all_brawlers)
    check_version()
    update_wall_model_classes()
    if not current_wall_model_is_latest():
        print("New Wall detection model found, downloading... (this might take a few minutes depending on your internet)")
        get_latest_wall_model_file()


def find_open_port(start_port=5185, host="127.0.0.1"):
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError("Could not find an open localhost port for the Flask UI.")


def open_browser_later(local_url):
    def _open():
        import webbrowser

        time.sleep(1.5)
        webbrowser.open(local_url)

    threading.Thread(target=_open, daemon=True, name="pyla-browser-launcher").start()

def stop_on_window_close(app):
    """Ask a running bot instance to stop when its desktop window closes."""
    def _on_close():
        print("PylaAI window closed, shutting down.")
        try:
            app.config["runtime_manager"].stop()
        except Exception as error:
            print(f"Could not stop the bot cleanly: {error}")

    return _on_close


def run_interface(app, local_url, interface_mode):
    """Present the loopback web UI using the selected startup interface."""
    if interface_mode == "desktop":
        from desktop import import_webview, run_webview

        webview_module, webview_error = import_webview()
        if webview_module is not None:
            try:
                run_webview(app, local_url, webview_module, on_close=stop_on_window_close(app))
                return
            except Exception as error:
                print(f"Could not start pywebview ({error}); opening the system browser instead.")
        else:
            print(f"pywebview is unavailable ({webview_error}); opening the system browser instead.")
        interface_mode = "browser"

    if interface_mode == "browser":
        open_browser_later(local_url)
        print("PylaAI is opening the local web UI in the system browser.")
    else:
        print(f"PylaAI is running headless. Open {local_url} manually to use the local web UI.")

    app.run(host="127.0.0.1", port=int(local_url.rsplit(":", 1)[1]), debug=False, use_reloader=False)

if __name__ == "__main__":
    print("Starting PylaAI, the best free and open source brawl stars bot")
    print("The only official discord is", get_discord_link())
    from webui import create_app

    port = find_open_port()
    app = create_app(pyla_main, start_discord_bot=True)
    local_url = f"http://127.0.0.1:{port}"
    print(f"Starting Pyla web UI at {local_url}")
    if CONSOLE_HIDDEN:
        print(f"Console output is written to {CONSOLE_LOG_FILE}")
    if CLI_ARGS.interface_mode is not None:
        print(f"{INTERFACE_MODE.capitalize()} interface mode was forced by a command-line argument.")
    run_interface(app, local_url, INTERFACE_MODE)
