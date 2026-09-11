import math
import random
import time
import ast
import traceback
import cv2
import numpy as np
import os
import threading
import weakref

from detect import Detect
from state_finder import get_state
from perception import DetectionStabilizer, SceneChangeGate
from pathfinding import LocalPathPlanner
from utils import load_toml_as_dict, count_hsv_pixels, load_brawlers_info, interpret_pyla_code, \
    JOYSTICK_RADIUS, clamp, config_bool, is_safe_ast, SAFE_GLOBALS


brawl_stars_width, brawl_stars_height = 1920, 1080
super_crop_area = load_toml_as_dict("./cfg/lobby_config.toml")['pixel_counter_crop_area']['super']
gadget_crop_area = load_toml_as_dict("./cfg/lobby_config.toml")['pixel_counter_crop_area']['gadget']
hypercharge_crop_area = load_toml_as_dict("./cfg/lobby_config.toml")['pixel_counter_crop_area']['hypercharge']
POISON_LOW_HSV = np.array((30, 90, 221), dtype=np.uint8)
POISON_HIGH_HSV = np.array((57, 114, 235), dtype=np.uint8)
POWER_CUBE_LOW_HSV = np.array((28, 155, 175), dtype=np.uint8)
POWER_CUBE_HIGH_HSV = np.array((82, 255, 255), dtype=np.uint8)
POWER_CUBE_CORE_LOW_HSV = np.array((34, 185, 205), dtype=np.uint8)
POWER_CUBE_CORE_HIGH_HSV = np.array((72, 255, 255), dtype=np.uint8)
class Play:

    def __init__(self, main_info_model, tile_detector_model, close_tile_detector_model, window_controller, pyla_code):
        bot_config = load_toml_as_dict("cfg/bot_config.toml")
        time_config = load_toml_as_dict("cfg/time_tresholds.toml")
        self.super_treshold = time_config["super"]
        self.gadget_treshold = time_config["gadget"]
        self.hypercharge_treshold = time_config["hypercharge"]
        self.walls_treshold = time_config["wall_detection"]
        self.last_walls_data = []
        self.last_bushes_data = []
        self._wall_tracks = []
        self._wall_motion_velocity = (0.0, 0.0)
        self._wall_motion_observed_at = 0.0
        self._projected_walls_key = None
        self._projected_walls = []
        self._wall_navigation_ready = False
        self._wall_navigation_initialized = False
        self._wall_navigation_grace_started_at = 0.0
        self.wall_navigation_startup_grace = max(
            0.25, float(bot_config.get("wall_navigation_startup_grace", 0.5))
        )
        self.keys_hold = []
        self.time_since_gadget_checked = time.time()
        self.is_gadget_ready = False
        self.time_since_hypercharge_checked = time.time()
        self.is_hypercharge_ready = False
        self.time_since_super_checked = time.time()
        self.is_super_ready = False
        self.ability_rearm_delay = max(
            0.25, float(bot_config.get("ability_rearm_delay", 1.0))
        )
        self._ability_used_at = {
            "gadget": 0.0,
            "hypercharge": 0.0,
            "super": 0.0,
        }
        self.window_controller = window_controller
        self.TILE_SIZE = bot_config.get("perceived_tile_size", 54)
        self.navigation_player_radius_tiles = max(
            0.35, min(0.75, float(
                bot_config.get("navigation_player_radius_tiles", 0.48)
            ))
        )
        self.navigation_player_radius = (
            self.TILE_SIZE * self.navigation_player_radius_tiles
        )
        self.centered_wall_detection = config_bool(bot_config.get("centered_wall_detection"), False)
        self.centered_wall_crop_size = 640

        self.verbose_debug = config_bool(load_toml_as_dict("cfg/debug_settings.toml").get('verbose_debug'), False)
        if self.verbose_debug:
            if not os.path.exists("debug_frames"):
                os.makedirs("debug_frames")
        self.Detect_main_info = Detect(
            main_info_model,
            classes=['enemy', 'teammate', 'player'],
            max_detections_per_class=bot_config.get("maximum_tracked_entities", 16),
        )
        self.tile_detector_model_classes = bot_config["wall_model_classes"]
        # Wall inference is not needed in menus. Keep its model off RAM/GPU
        # until a match contains a detected player.
        self.tile_detector_model_path = tile_detector_model
        self.close_tile_detector_model_path = close_tile_detector_model
        self.Detect_tile_detector = None
        self.Detect_centered_tile_detector = None
        self._wall_model_lock = threading.Lock()
        self._wall_model_loading = False
        self.wall_scene_gate = SceneChangeGate(
            threshold=bot_config.get("wall_scene_change_threshold", 4.0),
            maximum_age=bot_config.get("wall_detection_maximum_age", 1.25),
            minimum_age=bot_config.get("wall_detection_minimum_age", 0.45),
        )

        self.time_since_walls_checked = 0
        self.time_since_player_last_found = time.time()
        self.current_brawler = None
        self.brawlers_info = load_brawlers_info()
        self.brawler_ranges = None
        self.time_since_detections = {
            "player": time.time(),
            "enemy": time.time(),
        }
        self.time_since_last_proceeding = time.time()

        self.last_movement = ''
        self.last_movement_change_time = time.time()
        self.minimum_movement_delay = bot_config["minimum_movement_delay"]
        self.movement_direction_hysteresis = math.cos(math.radians(max(
            0.0, min(45.0, float(bot_config.get("movement_direction_hysteresis_degrees", 8.0)))
        )))
        self.no_detection_proceed_delay = time_config["no_detection_proceed"]
        self.gadget_pixels_minimum = bot_config["gadget_pixels_minimum"]
        self.hypercharge_pixels_minimum = bot_config["hypercharge_pixels_minimum"]
        self.super_pixels_minimum = bot_config["super_pixels_minimum"]
        self.ability_detection_scale = min(
            1.0, max(0.35, float(bot_config.get("ability_detection_scale", 0.5)))
        )
        self.wall_detection_confidence = bot_config["wall_detection_confidence"]
        self.wall_navigation_maximum_age = max(
            0.7, float(bot_config.get("wall_navigation_maximum_age", 1.2))
        )
        self.wall_obstacle_refresh_interval = max(
            0.15, float(bot_config.get("wall_obstacle_refresh_interval", 0.3))
        )
        self._last_obstacle_refresh_request = 0.0
        self.navigation_wall_padding = max(
            0.0, float(bot_config.get("navigation_wall_padding", 12.0))
        )
        self.entity_detection_confidence = bot_config["entity_detection_confidence"]
        self.seconds_to_hold_attack_after_reaching_max = bot_config["seconds_to_hold_attack_after_reaching_max"]
        self.minimum_attack_interval = max(
            0.0, float(bot_config.get("minimum_attack_interval", 0.08))
        )
        self.line_of_sight_wall_padding = max(
            4.0, float(bot_config.get("line_of_sight_wall_padding", 12.0))
        )
        self.last_attack_at = 0.0
        self._attack_authorized_until = 0.0
        self._super_authorized_until = 0.0
        self._current_enemy_data = ()
        self.persistent_data = {
            "time_since_holding_attack": None,
            "charged_attack_ready": False,
            "navigation_goal": "initializing",
            "combat_mode": None,
            "poison_heading": None,
        }
        self._playstyle_globals = None
        if isinstance(pyla_code, str):
            is_safe, error_msg = is_safe_ast(pyla_code)
            if not is_safe:
                print(f"Security/Syntax Validation Failed for playstyle: {error_msg}")
                self.pyla_code = compile("", "<string>", "exec")
            else:
                tree = ast.parse(pyla_code, filename="<pyla_script>", mode="exec")
                functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
                can_split = bool(functions) and all(
                    not node.decorator_list
                    and not node.args.defaults
                    and not node.args.kw_defaults
                    for node in functions
                )
                if can_split:
                    setup_tree = ast.Module(body=functions, type_ignores=[])
                    body_tree = ast.Module(
                        body=[node for node in tree.body if not isinstance(node, ast.FunctionDef)],
                        type_ignores=[],
                    )
                    ast.fix_missing_locations(setup_tree)
                    ast.fix_missing_locations(body_tree)
                    setup_code = compile(setup_tree, "<pyla_setup>", "exec")
                    self.pyla_code = compile(body_tree, "<pyla_tick>", "exec")
                    self._playstyle_globals = SAFE_GLOBALS.copy()
                    self._playstyle_globals['__builtins__'] = {}
                    exec(setup_code, self._playstyle_globals)
                else:
                    self.pyla_code = compile(tree, "<pyla_script>", "exec")
        else:
            self.pyla_code = pyla_code
        self.context = None
        self._dynamic_context = {}
        self._playstyle_context_initialized = False
        self._playstyle_api_callables = None
        self._last_playstyle_error = None
        self._last_playstyle_error_at = 0.0
        self.frame = None
        self._ability_crop_cache = {}
        self._ability_crop_scale = None
        self._ability_processing_buffers = {}
        self._decision_cache = {}
        self._wall_penetration_cache = {}
        self._prepared_frame_ref = None
        self._prepared_main_data = None
        self._target_memory = {}
        self.target_switch_ratio = max(1.0, float(bot_config.get("target_switch_ratio", 1.18)))
        self.power_cube_target_switch_ratio = max(
            1.0, float(bot_config.get("power_cube_target_switch_ratio", 1.06))
        )
        self.power_cube_target_match_tiles = max(
            0.35, float(bot_config.get("power_cube_target_match_tiles", 0.85))
        )
        self.target_match_distance = max(10.0, float(bot_config.get("target_match_distance", 180.0)))
        self.detection_stabilizer = DetectionStabilizer(
            smoothing=bot_config.get("detection_smoothing", 0.68),
            max_tracks=bot_config.get("maximum_tracked_entities", 16),
            prediction=bot_config.get("detection_prediction", 0.25),
            velocity_smoothing=bot_config.get("velocity_smoothing", 0.55),
        )
        self.objective_position = None
        self.objective_expires_at = 0.0
        self.objective_stability = 0
        self.objective_updated_at = 0.0
        self.objective_memory_duration = max(
            0.5, float(bot_config.get("objective_memory_duration", 4.0))
        )
        self.objective_arrival_distance = max(
            0.2, float(bot_config.get("objective_arrival_tiles", 0.35))
        )
        self.last_teammate_position = None
        self.last_teammate_direction = None
        self.last_teammate_distance = 0.0
        self.last_teammate_seen_at = 0.0
        self.last_teammate_search_direction = None
        self._teammate_search_changed_at = 0.0
        self.teammate_memory_duration = max(
            0.5, float(bot_config.get("teammate_memory_duration", 8.0))
        )
        self.path_planner = LocalPathPlanner(
            area_size=self.centered_wall_crop_size,
            cell_size=bot_config.get("pathfinding_cell_size", 36),
            cache_seconds=bot_config.get("pathfinding_cache_seconds", 0.32),
            wall_padding=self.navigation_wall_padding
                * (self.window_controller.scale_factor or 1.0),
        )
        self.fine_path_planner = LocalPathPlanner(
            area_size=self.centered_wall_crop_size,
            cell_size=bot_config.get("pathfinding_fine_cell_size", 24),
            cache_seconds=bot_config.get("pathfinding_cache_seconds", 0.32),
            wall_padding=self.navigation_wall_padding
                * (self.window_controller.scale_factor or 1.0),
        )
        self.fine_path_requests = 0
        self.fine_path_successes = 0
        self.fine_path_upgrades = 0
        self._failed_goal_heading = None
        self._failed_goal_kind = None
        self._failed_goal_until = 0.0
        self._route_failure_goal = None
        self._route_failure_count = 0
        self._route_failure_at = 0.0
        self._route_recovery_until = 0.0
        self.route_recoveries = 0
        self._failed_goal_alignment_cosine = math.cos(math.radians(25.0))
        self.unreachable_goal_avoids = 0
        self.pathfinding_horizon_tiles = max(
            3.0, min(8.0, float(
                bot_config.get("pathfinding_horizon_tiles", 5.8)
            ))
        )
        self.detour_commit_seconds = max(
            0.2, float(bot_config.get("detour_commit_seconds", 0.55))
        )
        self.combat_strafe_seconds = max(
            1.2, float(bot_config.get("combat_strafe_seconds", 2.2))
        )
        self.detour_goal_reset_cosine = math.cos(math.radians(max(
            20.0, min(90.0, float(bot_config.get("detour_goal_reset_degrees", 55.0)))
        )))
        self._detour_heading = None
        self._detour_goal_heading = None
        self._detour_expires_at = 0.0
        self._direct_clear_since = 0.0
        self._exploration_heading = (0.0, -JOYSTICK_RADIUS)
        self._exploration_expires_at = 0.0
        self._exploration_index = 0
        self._combat_strafe_side = 1
        self._combat_strafe_until = 0.0
        self._formation_side = -1
        self._obstacle_progress_center = None
        self._obstacle_progress_at = 0.0
        self._obstacle_escape_side = 1
        self._obstacle_stuck_count = 0
        self.wrong_way_corrections = 0
        self.poison_cache_interval = max(
            0.05, float(bot_config.get("poison_detection_interval", 0.2))
        )
        self._poison_cache = None
        self._poison_cache_at = 0.0
        self._poison_cache_player = None
        self._poison_buffer_shape = None
        self._poison_hsv_buffer = None
        self._poison_mask_buffer = None
        self._poison_filtered_mask_buffer = None
        self._latest_poison_gas = {
            "up": 0, "down": 0, "left": 0, "right": 0,
        }
        self._poison_danger_regions = ()
        self.power_cube_detection_interval = max(
            0.1, float(bot_config.get("power_cube_detection_interval", 0.22))
        )
        self._power_cube_cache = []
        self._power_cube_cache_at = 0.0
        self._power_cube_cache_player = None
        self._power_cube_buffer_shape = None
        self._power_cube_hsv_buffer = None
        self._power_cube_mask_buffer = None
        self._power_cube_core_mask_buffer = None
        self._power_cube_tracks = []
        self._last_power_cube_target = None
        self.power_cube_scans = 0
        self.power_cube_cache_hits = 0
        self.power_cube_candidates_rejected = 0
        self.wall_inference_count = 0
        self.wall_cache_count = 0
        self.wall_projection_uses = 0
        self.poison_compute_count = 0
        self.poison_cache_count = 0
        self.detour_reversals_prevented = 0
        self.stale_movement_overrides = 0
        self.opposite_goal_overrides = 0
        self.short_forward_overrides = 0
        self.overlap_escape_approvals = 0
        self._last_strategic_movement = (0.0, 0.0)
        self._last_output_movement = (0.0, 0.0)
        self.collision_corrections = 0
        self.collision_stops = 0
        self.progressive_escape_moves = 0
        self.camera_goal_compensations = 0
        self.navigation_decisions = 0
        self.navigation_goal_switches = 0
        self.navigation_idle_decisions = 0
        self.navigation_goal_counts = {}
        self.route_failure_counts = {}
        self._strategic_heading = None
        self._strategic_goal = "initializing"
        self._strategic_commit_until = 0.0
        self.goal_downgrade_holds = 0
        self._goal_priorities = {
            "poison": 4,
            "power_cube": 3,
            "fallback_power_cube": 3,
            "enemy_approach": 2,
            "enemy_retreat": 2,
            "enemy_strafe": 2,
            "objective": 2,
            "fallback_enemy": 2,
            "fallback_objective": 2,
            "teammate": 1,
            "teammate_formation": 1,
            "teammate_search": 1,
            "fallback_teammate": 1,
            "fallback_teammate_search": 1,
            "route_recovery_teammate": 1,
            "route_recovery_search": 1,
        }
        self.direct_navigation_moves = 0
        self.planned_navigation_moves = 0
        self._last_navigation_goal = "initializing"
        self.stale_wall_stops = 0
        self.unsafe_attack_requests_blocked = 0
        self.unsafe_super_requests_blocked = 0
        self.unknown_geometry_attacks_blocked = 0

    @staticmethod
    def get_entity_pos(entity):
        return (entity[0] + entity[2]) / 2, (entity[1] + entity[3]) / 2

    @staticmethod
    def get_distance(enemy_coords, player_coords):
        return math.hypot(enemy_coords[0] - player_coords[0], enemy_coords[1] - player_coords[1])

    @staticmethod
    def is_there_enemy(enemy_data):
        if not enemy_data:
            return False
        return True

    def attack(self, touch_up=True, touch_down=True):
        now = time.monotonic()
        if touch_down and now > self._attack_authorized_until:
            # Starting a shot requires a current line-of-sight confirmation.
            # A touch-up is still allowed so charged attacks cannot stick.
            self.unsafe_attack_requests_blocked += 1
            return False
        # Rate-limit only complete taps. Charge start and release are edge
        # events and must always reach the device immediately.
        if touch_up and touch_down:
            if now - self.last_attack_at < self.minimum_attack_interval:
                return False
            self.last_attack_at = now
        self.window_controller.press("attack", touch_up=touch_up, touch_down=touch_down)
        return True

    def use_hypercharge(self):
        print("Using hypercharge")
        self.window_controller.press("hypercharge")
        self.time_since_hypercharge_checked = time.time()
        self._ability_used_at["hypercharge"] = time.monotonic()
        self.is_hypercharge_ready = False

    def use_gadget(self):
        print("Using gadget")
        self.window_controller.press("gadget")
        self.time_since_gadget_checked = time.time()
        self._ability_used_at["gadget"] = time.monotonic()
        self.is_gadget_ready = False

    def use_super(self):
        if time.monotonic() > self._super_authorized_until:
            self.unsafe_super_requests_blocked += 1
            return False
        print("Using super")
        self.window_controller.press("super")
        self.time_since_super_checked = time.time()
        self._ability_used_at["super"] = time.monotonic()
        self.is_super_ready = False
        return True

    def ability_can_recheck(self, name):
        return time.monotonic() - self._ability_used_at[name] >= self.ability_rearm_delay

    def remember_objective(self, position):
        if position is None or len(position) < 2:
            return
        now = time.time()
        new_position = (float(position[0]), float(position[1]))
        scale = self.window_controller.scale_factor or 1.0
        stable_distance = self.TILE_SIZE * 0.25 * scale
        same_track_distance = self.TILE_SIZE * 1.5 * scale
        if (
            self.objective_position is not None
            and now - self.objective_updated_at <= 0.75
        ):
            displacement = self.get_distance(
                new_position, self.objective_position
            )
            if displacement <= stable_distance:
                self.objective_stability = min(
                    6, self.objective_stability + 1
                )
            elif displacement <= same_track_distance:
                self.objective_stability = max(
                    0, self.objective_stability - 2
                )
            else:
                self.objective_stability = 0
        else:
            self.objective_stability = 0
        self.objective_position = new_position
        self.objective_updated_at = now
        memory_duration = (
            self.objective_memory_duration
            if self.objective_stability >= 2
            else min(1.4, self.objective_memory_duration)
        )
        self.objective_expires_at = now + memory_duration

    def _shift_screen_space_memories(self, delta_x, delta_y):
        """Move remembered world targets with the detected camera motion."""
        if math.hypot(delta_x, delta_y) < 1.5:
            return
        maximum_delta = self.TILE_SIZE * 1.25 * (
            self.window_controller.scale_factor or 1.0
        )
        if math.hypot(delta_x, delta_y) > maximum_delta:
            return
        if self.objective_position is not None:
            self.objective_position = (
                self.objective_position[0] + delta_x,
                self.objective_position[1] + delta_y,
            )
        if self.last_teammate_position is not None:
            self.last_teammate_position = (
                self.last_teammate_position[0] + delta_x,
                self.last_teammate_position[1] + delta_y,
            )
        for target_type, position in tuple(self._target_memory.items()):
            self._target_memory[target_type] = (
                position[0] + delta_x, position[1] + delta_y
            )
        if self._power_cube_cache:
            self._power_cube_cache = [
                [
                    cube[0] + delta_x, cube[1] + delta_y,
                    cube[2] + delta_x, cube[3] + delta_y,
                ]
                for cube in self._power_cube_cache
            ]
        if self._power_cube_tracks:
            for track in self._power_cube_tracks:
                box = track["box"]
                track["box"] = [
                    box[0] + delta_x, box[1] + delta_y,
                    box[2] + delta_x, box[3] + delta_y,
                ]
        if self._last_power_cube_target is not None:
            self._last_power_cube_target = (
                self._last_power_cube_target[0] + delta_x,
                self._last_power_cube_target[1] + delta_y,
            )
        self.camera_goal_compensations += 1

    def get_recent_objective_movement(self, player_position):
        if self.objective_position is None or time.time() >= self.objective_expires_at:
            self.objective_position = None
            return None
        dx = self.objective_position[0] - player_position[0]
        dy = self.objective_position[1] - player_position[1]
        arrival_distance = (
            self.TILE_SIZE * self.window_controller.scale_factor
            * self.objective_arrival_distance
        )
        if math.hypot(dx, dy) <= arrival_distance:
            self.objective_position = None
            return None
        return dx, dy

    def get_teammate_search_movement(self, player_box, walls):
        """Keep searching in one meaningful direction until a teammate appears.

        A short teammate detector miss must not turn into a new random patrol
        heading.  Screen coordinates are camera-relative, so remember a
        direction rather than an old absolute point and let the navigator
        route that direction around terrain.
        """
        now = time.time()
        if self.last_teammate_search_direction is None:
            initial = self.get_exploration_movement(player_box, walls)
            length = math.hypot(*initial)
            if length >= 1:
                self.last_teammate_search_direction = (
                    initial[0] / length, initial[1] / length
                )
            self._teammate_search_changed_at = now
        elif (
            now - self._teammate_search_changed_at >= 4.0
            and self._obstacle_stuck_count >= 1
        ):
            # Sweep one neighbouring sector at a time. This is deterministic,
            # retains most forward progress, and eventually searches the map
            # instead of pushing against one edge forever.
            self.last_teammate_search_direction = self.rotate_movement(
                self.last_teammate_search_direction,
                self._formation_side * math.pi / 4,
            )
            self._teammate_search_changed_at = now
            self._detour_heading = None
            self._detour_goal_heading = None
            self._detour_expires_at = 0.0
        return self.normalize_move(*self.last_teammate_search_direction)

    def get_combat_strafe_movement(
        self, player_box, player_position, enemy_position, walls
    ):
        """Move laterally in firing range instead of freezing in the lane."""
        dx = enemy_position[0] - player_position[0]
        dy = enemy_position[1] - player_position[1]
        if math.hypot(dx, dy) < 1:
            return self.get_teammate_search_movement([], [])
        now = time.monotonic()
        preferred = self.normalize_move(
            -dy * self._combat_strafe_side,
            dx * self._combat_strafe_side,
        )
        opposite = (-preferred[0], -preferred[1])
        lookahead = self.TILE_SIZE * 0.8 * (
            self.window_controller.scale_factor or 1.0
        )
        preferred_blocked = self.is_path_blocked(
            player_box, preferred, walls, distance=lookahead
        )
        opposite_blocked = self.is_path_blocked(
            player_box, opposite, walls, distance=lookahead
        )
        preferred_risk = self._movement_poison_risk(preferred)
        opposite_risk = self._movement_poison_risk(opposite)
        should_flip = (
            not opposite_blocked
            and (
                preferred_blocked
                or opposite_risk + 0.05 < preferred_risk
                or (
                    now >= self._combat_strafe_until
                    and not preferred_blocked
                    and opposite_risk <= preferred_risk + 0.05
                )
            )
        )
        if should_flip:
            self._combat_strafe_side *= -1
            self._combat_strafe_until = now + self.combat_strafe_seconds
            return opposite
        if now >= self._combat_strafe_until:
            # The alternative is blocked or materially less safe. Retain the
            # clear lane instead of flipping merely because a timer elapsed.
            self._combat_strafe_until = now + self.combat_strafe_seconds
        return preferred

    def get_teammate_formation_movement(
        self, player_box, player_position, teammate_position, walls
    ):
        """Hold a close formation without approach/orbit oscillation."""
        dx = teammate_position[0] - player_position[0]
        dy = teammate_position[1] - player_position[1]
        distance = math.hypot(dx, dy)
        if distance < 1:
            return self.get_teammate_search_movement([], [])
        toward_x, toward_y = dx / distance, dy / distance
        target_distance = self.TILE_SIZE * 0.72 * (
            self.window_controller.scale_factor or 1.0
        )
        # Positive correction closes excessive spacing; negative correction
        # prevents both bots from occupying the same point. The tangent keeps
        # the bot active and useful without repeatedly crossing the distance
        # threshold and snapping back toward the teammate.
        radial_correction = max(-0.75, min(
            0.75,
            (distance - target_distance) / max(1.0, target_distance),
        ))
        tangent_x = -toward_y * self._formation_side
        tangent_y = toward_x * self._formation_side
        preferred = self.normalize_move(
            tangent_x * 0.62 + toward_x * radial_correction,
            tangent_y * 0.62 + toward_y * radial_correction,
        )
        alternate = self.normalize_move(
            -tangent_x * 0.62 + toward_x * radial_correction,
            -tangent_y * 0.62 + toward_y * radial_correction,
        )
        lookahead = self.TILE_SIZE * 0.75 * (
            self.window_controller.scale_factor or 1.0
        )
        preferred_blocked = self.is_path_blocked(
            player_box, preferred, walls, distance=lookahead
        )
        alternate_blocked = self.is_path_blocked(
            player_box, alternate, walls, distance=lookahead
        )
        if (
            not alternate_blocked
            and (
                preferred_blocked
                or self._movement_poison_risk(alternate) + 0.05
                    < self._movement_poison_risk(preferred)
            )
        ):
            self._formation_side *= -1
            return alternate
        if preferred_blocked and alternate_blocked and abs(radial_correction) > 0.12:
            radial = self.normalize_move(
                toward_x * radial_correction,
                toward_y * radial_correction,
            )
            if not self.is_path_blocked(
                player_box, radial, walls, distance=lookahead
            ):
                return radial
        return preferred

    def get_emergency_goal_movement(self, data):
        """Guarantee a useful goal when a playstyle yields no movement."""
        player_box = data["player"][0]
        player_position = self.get_player_position(player_box)
        walls = data.get("wall") or []
        enemies = data.get("enemy") or []
        cubes = data.get("power_cube") or []
        if cubes:
            cube_position, _ = self.find_closest_power_cube(
                cubes, player_position, walls
            )
            if cube_position is not None:
                self.persistent_data["navigation_goal"] = "fallback_power_cube"
                return (
                    cube_position[0] - player_position[0],
                    cube_position[1] - player_position[1],
                )
        if enemies:
            enemy_position, _ = self.find_closest_enemy(
                enemies, player_position, walls, "attack"
            )
            if enemy_position is not None:
                enemy_movement = (
                    enemy_position[0] - player_position[0],
                    enemy_position[1] - player_position[1],
                )
                if math.hypot(*enemy_movement) >= 1:
                    self.persistent_data["navigation_goal"] = "fallback_enemy"
                    return enemy_movement
        objective = self.get_recent_objective_movement(player_position)
        if objective is not None:
            self.persistent_data["navigation_goal"] = "fallback_objective"
            return objective
        teammate_position, teammate_distance = self.find_closest_teammate(
            data.get("teammate") or [], player_position, walls
        )
        if teammate_position is not None:
            self.persistent_data["navigation_goal"] = "fallback_teammate"
            if teammate_distance < self.TILE_SIZE * 1.05 * (
                self.window_controller.scale_factor or 1.0
            ):
                formation = self.get_teammate_formation_movement(
                    player_box, player_position, teammate_position, walls
                )
                if math.hypot(*formation) >= 1:
                    return formation
            teammate_movement = (
                teammate_position[0] - player_position[0],
                teammate_position[1] - player_position[1],
            )
            if math.hypot(*teammate_movement) >= 1:
                return teammate_movement
        self.persistent_data["navigation_goal"] = "fallback_teammate_search"
        return self.get_teammate_search_movement(player_box, walls)

    def find_closest_power_cube(self, cube_data, player_position, walls):
        """Select the quickest reachable dropped cube, not merely nearest."""
        if not cube_data:
            self._target_memory.pop("power_cube", None)
            self._last_power_cube_target = None
            return None, None
        positions = [self.get_entity_pos(cube) for cube in cube_data]
        route_costs = {
            position: self.estimate_navigation_cost(
                player_position, position, walls
            )
            for position in positions
        }
        best = min(
            positions,
            key=lambda position: (
                route_costs[position],
                self.get_distance(player_position, position),
            ),
        )
        selected = best
        previous = self._target_memory.get("power_cube")
        if previous is not None:
            matching = min(
                positions,
                key=lambda position: self.get_distance(position, previous),
            )
            if (
                self.get_distance(matching, previous)
                <= self.TILE_SIZE * self.power_cube_target_match_tiles
                    * (self.window_controller.scale_factor or 1.0)
                and route_costs[matching]
                    <= route_costs[best]
                        * self.power_cube_target_switch_ratio
            ):
                selected = matching
        self._target_memory["power_cube"] = selected
        self._last_power_cube_target = selected
        return selected, self.get_distance(player_position, selected)

    def detect_power_cubes(self, player_box, entity_boxes=(), walls=()):
        """Find bright, compact green dropped cubes near the player cheaply."""
        if self.frame is None or player_box is None:
            return []
        now = time.monotonic()
        player_position = self.get_player_position(player_box)
        if (
            now - self._power_cube_cache_at < self.power_cube_detection_interval
            and self._power_cube_cache_player is not None
            and self.get_distance(
                player_position, self._power_cube_cache_player
            ) <= 8.0 * (self.window_controller.scale_factor or 1.0)
        ):
            self.power_cube_cache_hits += 1
            return [cube[:] for cube in self._power_cube_cache]

        self.power_cube_scans += 1
        frame_height, frame_width = self.frame.shape[:2]
        half_size = min(
            self.centered_wall_crop_size * 0.5,
            frame_width * 0.46,
            frame_height * 0.46,
        )
        x1 = max(0, int(player_position[0] - half_size))
        y1 = max(0, int(player_position[1] - half_size))
        x2 = min(frame_width, int(player_position[0] + half_size))
        y2 = min(frame_height, int(player_position[1] + half_size))
        if x2 <= x1 or y2 <= y1:
            return []
        roi = self.frame[y1:y2, x1:x2]
        roi_shape = roi.shape[:2]
        if self._power_cube_buffer_shape != roi_shape:
            self._power_cube_buffer_shape = roi_shape
            self._power_cube_hsv_buffer = np.empty_like(roi)
            self._power_cube_mask_buffer = np.empty(
                roi_shape, dtype=np.uint8
            )
            self._power_cube_core_mask_buffer = np.empty(
                roi_shape, dtype=np.uint8
            )
        hsv = cv2.cvtColor(
            roi, cv2.COLOR_RGB2HSV, dst=self._power_cube_hsv_buffer
        )
        # Dropped cubes have a highly saturated lime/green core. Poison is
        # intentionally lower saturation and health bars are rejected below
        # by geometry.
        mask = cv2.inRange(
            hsv,
            POWER_CUBE_LOW_HSV,
            POWER_CUBE_HIGH_HSV,
            dst=self._power_cube_mask_buffer,
        )
        cv2.inRange(
            hsv,
            POWER_CUBE_CORE_LOW_HSV,
            POWER_CUBE_CORE_HIGH_HSV,
            dst=self._power_cube_core_mask_buffer,
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        scale = max(0.35, self.window_controller.scale_factor or 1.0)
        minimum_area = max(16, int(55 * scale * scale))
        maximum_area = max(220, int(2600 * scale * scale))
        minimum_side = max(4, int(7 * scale))
        maximum_side = max(24, int(82 * scale))
        cubes = []
        for index in range(1, count):
            local_x, local_y, width, height, area = stats[index]
            if area < minimum_area or area > maximum_area:
                continue
            if (
                width < minimum_side or height < minimum_side
                or width > maximum_side or height > maximum_side
            ):
                continue
            aspect = width / max(1.0, float(height))
            fill = area / max(1.0, float(width * height))
            if not 0.68 <= aspect <= 1.48 or fill < 0.24:
                continue
            component_core = self._power_cube_core_mask_buffer[
                local_y:local_y + height, local_x:local_x + width
            ]
            core_pixels = cv2.countNonZero(component_core)
            core_fraction = core_pixels / max(1.0, float(area))
            if (
                core_pixels < max(4, int(area * 0.09))
                or core_fraction > 0.90
            ):
                self.power_cube_candidates_rejected += 1
                continue
            candidate = [
                x1 + int(local_x), y1 + int(local_y),
                x1 + int(local_x + width), y1 + int(local_y + height),
            ]
            candidate_center = self.get_entity_pos(candidate)
            entity_margin = self.TILE_SIZE * 0.18 * scale
            if any(
                box[0] - entity_margin <= candidate_center[0]
                    <= box[2] + entity_margin
                and box[1] - entity_margin <= candidate_center[1]
                    <= box[3] + entity_margin
                for box in entity_boxes if box is not None and len(box) >= 4
            ):
                self.power_cube_candidates_rejected += 1
                continue
            if any(
                wall[0] <= candidate_center[0] <= wall[2]
                and wall[1] <= candidate_center[1] <= wall[3]
                for wall in walls if wall is not None and len(wall) >= 4
            ):
                # Pickups cannot occupy solid wall/water geometry. Bright map
                # details inside those detections are guaranteed false goals.
                self.power_cube_candidates_rejected += 1
                continue
            cubes.append(candidate)
        # Glow, bright core and cube faces can form several disconnected HSV
        # components. Merge nearby square components so navigation targets the
        # complete pickup centre instead of alternating between its edges.
        merge_distance = self.TILE_SIZE * 0.70 * scale
        merged = []
        for cube in cubes:
            cube_center = self.get_entity_pos(cube)
            destination = None
            for existing in merged:
                if self.get_distance(
                    cube_center, self.get_entity_pos(existing)
                ) <= merge_distance:
                    destination = existing
                    break
            if destination is None:
                merged.append(cube[:])
            else:
                destination[0] = min(destination[0], cube[0])
                destination[1] = min(destination[1], cube[1])
                destination[2] = max(destination[2], cube[2])
                destination[3] = max(destination[3], cube[3])
        track_radius = self.TILE_SIZE * 0.62 * scale
        close_radius = self.TILE_SIZE * 1.05 * scale
        next_tracks = []
        confirmed = []
        unmatched_previous = list(self._power_cube_tracks)
        for cube in merged:
            center = self.get_entity_pos(cube)
            match = None
            if unmatched_previous:
                match = min(
                    unmatched_previous,
                    key=lambda track: self.get_distance(
                        center, self.get_entity_pos(track["box"])
                    ),
                )
                if self.get_distance(
                    center, self.get_entity_pos(match["box"])
                ) > track_radius:
                    match = None
            hits = (match["hits"] + 1) if match is not None else 1
            if match is not None:
                unmatched_previous.remove(match)
            next_tracks.append({
                "box": cube, "hits": min(4, hits), "misses": 0,
            })
            if hits >= 2 or self.get_distance(center, player_position) <= close_radius:
                confirmed.append(cube)
        # Keep a confirmed stationary pickup through one missed HSV scan.
        # This avoids cube -> teammate -> cube goal flips from one noisy frame
        # while still expiring a collected cube in under half a second.
        for track in unmatched_previous:
            misses = track.get("misses", 0) + 1
            if track.get("hits", 0) < 2 or misses > 1:
                continue
            retained = {
                "box": track["box"], "hits": track["hits"],
                "misses": misses,
            }
            next_tracks.append(retained)
            confirmed.append(retained["box"])
        self._power_cube_tracks = next_tracks
        cubes = confirmed
        self._power_cube_cache = cubes
        self._power_cube_cache_at = now
        self._power_cube_cache_player = player_position
        return [cube[:] for cube in cubes]

    def get_route_recovery_movement(self, data):
        """Choose useful team movement while an objective route is invalid."""
        player_box = data["player"][0]
        player_position = self.get_player_position(player_box)
        walls = data.get("wall") or []
        teammate_position, teammate_distance = self.find_closest_teammate(
            data.get("teammate") or [], player_position, walls
        )
        if teammate_position is not None:
            self.persistent_data["navigation_goal"] = "route_recovery_teammate"
            if teammate_distance < self.TILE_SIZE * 1.05 * (
                self.window_controller.scale_factor or 1.0
            ):
                return self.get_teammate_formation_movement(
                    player_box, player_position, teammate_position, walls
                )
            return (
                teammate_position[0] - player_position[0],
                teammate_position[1] - player_position[1],
            )
        self.persistent_data["navigation_goal"] = "route_recovery_search"
        return self.get_teammate_search_movement(player_box, walls)

    @staticmethod
    def get_random_movement():
        random_movement = random.randint(-75, 75), random.randint(-75, 75)
        return random_movement

    def get_exploration_movement(self, player_box, walls):
        """Head toward the friendly-spawn-to-map-centre direction."""
        danger_up, danger_down, danger_left, danger_right = (
            self._directional_poison_danger()
        )
        # Gas closes from the map edge, so its opposite direction is direct
        # evidence of where the map centre lies. Retain that evidence after
        # the local gas pixels leave the scan instead of resuming a patrol.
        inferred_x = danger_left - danger_right
        inferred_y = danger_up - danger_down
        if math.hypot(inferred_x, inferred_y) >= 0.05:
            centre_heading = self.normalize_move(inferred_x, inferred_y)
            previous = self._exploration_heading
            if math.hypot(*previous) >= 1:
                centre_heading = self.normalize_move(
                    previous[0] * 0.35 + centre_heading[0] * 0.65,
                    previous[1] * 0.35 + centre_heading[1] * 0.65,
                )
            self._exploration_heading = centre_heading
        else:
            centre_heading = self._exploration_heading
        if self._movement_poison_risk(centre_heading) >= 0.08:
            headings = (
                centre_heading,
                (0.0, -JOYSTICK_RADIUS),
                (JOYSTICK_RADIUS, 0.0),
                (0.0, JOYSTICK_RADIUS),
                (-JOYSTICK_RADIUS, 0.0),
            )
            centre_heading = min(headings, key=self._movement_poison_risk)
        # Do not rotate merely because a wall is ahead. The central A*
        # navigator owns that decision and will preserve the map-centre goal
        # while routing around the obstacle.
        self._exploration_heading = centre_heading
        return centre_heading

    @staticmethod
    def movement_to_vector(movement):
        if not isinstance(movement, (tuple, list)) or len(movement) != 2:
            return None

        x, y = movement
        if x is None or y is None:
            return None

        try:
            return float(x), float(y)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def rotate_movement(movement, angle_radians):
        x, y = movement
        cos_angle = math.cos(angle_radians)
        sin_angle = math.sin(angle_radians)
        return (
            x * cos_angle - y * sin_angle,
            x * sin_angle + y * cos_angle,
        )

    def load_brawler_ranges(self, brawlers_info=None):
        if not brawlers_info:
            brawlers_info = load_brawlers_info()
        screen_size_ratio = self.window_controller.scale_factor
        ranges = {}
        for brawler, info in brawlers_info.items():
            attack_range = info['attack_range']
            safe_range = info['safe_range']
            super_range = info['super_range']
            v = [safe_range, attack_range, super_range]
            ranges[brawler] = [int(v[0] * screen_size_ratio), int(v[1] * screen_size_ratio), int(v[2] * screen_size_ratio)]
        return ranges

    @staticmethod
    def can_attack_through_walls(brawler, skill_type, brawlers_info=None):
        if not brawlers_info: brawlers_info = load_brawlers_info()
        if skill_type == "attack":
            return brawlers_info[brawler]['ignore_walls_for_attacks']
        elif skill_type == "super":
            return brawlers_info[brawler]['ignore_walls_for_supers']
        raise ValueError("skill_type must be either 'attack' or 'super'")

    @staticmethod
    def must_brawler_hold_attack(brawler, brawlers_info=None):
        if not brawlers_info: brawlers_info = load_brawlers_info()
        return brawlers_info[brawler]['hold_attack'] > 0

    @staticmethod
    def segment_intersects_rect(start, end, rect):
        """Liang-Barsky segment/AABB test without per-wall OpenCV calls."""
        x1, y1 = start
        dx, dy = end[0] - x1, end[1] - y1
        left, top, right, bottom = rect
        enter, leave = 0.0, 1.0

        p, q = -dx, x1 - left
        if abs(p) < 1e-9:
            if q < 0:
                return False
        else:
            ratio = q / p
            if p < 0:
                if ratio > leave:
                    return False
                enter = max(enter, ratio)
            else:
                if ratio < enter:
                    return False
                leave = min(leave, ratio)

        p, q = dx, right - x1
        if abs(p) < 1e-9:
            if q < 0:
                return False
        else:
            ratio = q / p
            if p < 0:
                if ratio > leave:
                    return False
                enter = max(enter, ratio)
            else:
                if ratio < enter:
                    return False
                leave = min(leave, ratio)

        p, q = -dy, y1 - top
        if abs(p) < 1e-9:
            if q < 0:
                return False
        else:
            ratio = q / p
            if p < 0:
                if ratio > leave:
                    return False
                enter = max(enter, ratio)
            else:
                if ratio < enter:
                    return False
                leave = min(leave, ratio)

        p, q = dy, bottom - y1
        if abs(p) < 1e-9:
            if q < 0:
                return False
        else:
            ratio = q / p
            if p < 0:
                if ratio > leave:
                    return False
                enter = max(enter, ratio)
            else:
                if ratio < enter:
                    return False
                leave = min(leave, ratio)
        return enter <= leave

    @staticmethod
    def walls_block_line_of_sight(p1, p2, walls, padding=0.0):
        if not walls:
            return False

        min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
        min_y, max_y = min(p1[1], p2[1]), max(p1[1], p2[1])
        for wall in walls:
            x1, y1, x2, y2 = wall[:4]

            if padding:
                x1 -= padding
                y1 -= padding
                x2 += padding
                y2 += padding

            if max_x < x1 or min_x > x2 or max_y < y1 or min_y > y2:
                continue

            if Play.segment_intersects_rect(p1, p2, (x1, y1, x2, y2)):
                return True
        return False

    def get_player_hit_circle(self, player_box):
        radius = self.navigation_player_radius * (
            self.window_controller.scale_factor or 1
        )
        if player_box and len(player_box) >= 4:
            x1, y1, x2, y2 = player_box[:4]
            return ((x1 + x2) / 2, y2 - radius), radius

        return None, radius

    def get_player_position(self, player_box):
        """Return the same physical origin used by collision and A*."""
        center, _ = self.get_player_hit_circle(player_box)
        if center is not None:
            return center
        return self.get_entity_pos(player_box)

    def get_actual_player_box(self, player_box):
        center, radius = self.get_player_hit_circle(player_box)
        if center is None:
            return None
        return [
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ]

    @staticmethod
    def point_rect_distance_sq(point, rect):
        x, y = point
        x1, y1, x2, y2 = rect
        dx = max(x1 - x, 0, x - x2)
        dy = max(y1 - y, 0, y - y2)
        return dx * dx + dy * dy

    @staticmethod
    def walls_block_swept_circle(p1, p2, radius, walls):
        if not walls:
            return False

        min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
        min_y, max_y = min(p1[1], p2[1]), max(p1[1], p2[1])
        radius = math.ceil(radius)
        radius_sq = radius * radius

        for wall in walls:
            wall_rect = wall[:4]
            x1, y1, x2, y2 = wall_rect
            expanded_x1 = x1 - radius
            expanded_y1 = y1 - radius
            expanded_x2 = x2 + radius
            expanded_y2 = y2 + radius

            if max_x < expanded_x1 or min_x > expanded_x2 or max_y < expanded_y1 or min_y > expanded_y2:
                continue

            if Play.segment_intersects_rect(
                p1, p2, (expanded_x1, expanded_y1, expanded_x2, expanded_y2)
            ):
                start_distance_sq = Play.point_rect_distance_sq(p1, wall_rect)
                if start_distance_sq <= radius_sq:
                    nearest_x = clamp(p1[0], x1, x2)
                    nearest_y = clamp(p1[1], y1, y2)
                    outward_x = p1[0] - nearest_x
                    outward_y = p1[1] - nearest_y
                    if abs(outward_x) + abs(outward_y) < 1e-6:
                        # Overlapping detector boxes can place the estimated
                        # player centre inside terrain. Choose its nearest edge
                        # as the deterministic escape direction.
                        edge = min(
                            (
                                (p1[0] - x1, -1.0, 0.0),
                                (x2 - p1[0], 1.0, 0.0),
                                (p1[1] - y1, 0.0, -1.0),
                                (y2 - p1[1], 0.0, 1.0),
                            ),
                            key=lambda item: item[0],
                        )
                        outward_x, outward_y = edge[1], edge[2]
                    movement_x = p2[0] - p1[0]
                    movement_y = p2[1] - p1[1]
                    # Positive is away, zero is tangent, negative enters the
                    # obstacle. End-distance alone misclassified tangents and
                    # could also approve a long segment crossing the wall.
                    if (
                        movement_x * outward_x + movement_y * outward_y
                        >= -1e-6
                    ):
                        continue
                return True

        return False

    def is_enemy_hittable(self, player_pos, enemy_pos, walls, skill_type):
        player_key = tuple(player_pos)
        enemy_key = tuple(enemy_pos)
        cache_key = (
            "line_of_sight", skill_type, self.current_brawler,
            player_key, enemy_key, id(walls),
        )
        if cache_key in self._decision_cache:
            return self._decision_cache[cache_key]
        penetration_key = (self.current_brawler, skill_type)
        can_ignore_walls = self._wall_penetration_cache.get(penetration_key)
        if can_ignore_walls is None:
            can_ignore_walls = self.can_attack_through_walls(
                self.current_brawler, skill_type, self.brawlers_info
            )
            self._wall_penetration_cache[penetration_key] = can_ignore_walls
        if can_ignore_walls:
            result = True
        elif not self._wall_navigation_ready:
            # An empty wall list before the first tile inference means
            # "unknown", not "clear line of sight".
            result = False
        elif time.time() - self.time_since_walls_checked > 0.70:
            # Wall boxes are screen-space coordinates. Once the camera moves,
            # an old box is not safe evidence for an auto-aim shot.
            result = False
        elif self.get_distance(player_pos, enemy_pos) > self._navigation_visible_radius():
            # The centred tile model only maps a local square around the
            # player.  A missing wall beyond that crop is unknown terrain,
            # not proof of clear line of sight (especially for long-range
            # brawlers such as Belle).
            self.unknown_geometry_attacks_blocked += 1
            result = False
        else:
            # Attack and super can have different wall-penetration rules, but
            # when either needs geometry the underlying segment/wall result is
            # identical. Share that potentially long wall scan within a frame.
            geometry_key = (
                "wall_line", player_key, enemy_key, id(walls)
            )
            blocked = self._decision_cache.get(geometry_key)
            if blocked is None:
                blocked = self.walls_block_line_of_sight(
                    player_pos, enemy_pos, walls,
                    padding=(
                        self.line_of_sight_wall_padding
                        * self.window_controller.scale_factor
                    ),
                )
                self._decision_cache[geometry_key] = blocked
            result = not blocked
        if skill_type in ("attack", "super") and result:
            # The attack button auto-aims at the nearest enemy, which may not
            # be the visible target selected by the playstyle. Do not approve
            # the tap when a closer enemy is hidden behind terrain.
            autoaim_is_safe = True
            if not can_ignore_walls:
                # Brawl Stars auto-aim chooses the nearest detected enemy,
                # not necessarily the visible target selected by a playstyle.
                # Validate that exact target. The old '< target distance'
                # loop missed equal-distance enemies and could consequently
                # authorize a shot which auto-aim sent into terrain.
                autoaim_pos = enemy_pos
                if self._current_enemy_data:
                    autoaim_pos = min(
                        (
                            self.get_player_position(candidate)
                            for candidate in self._current_enemy_data
                        ),
                        key=lambda position: self.get_distance(
                            player_pos, position
                        ),
                    )
                if (
                    self.get_distance(player_pos, autoaim_pos)
                    > self._navigation_visible_radius()
                ):
                    self.unknown_geometry_attacks_blocked += 1
                    autoaim_is_safe = False
                else:
                    autoaim_is_safe = not self.walls_block_line_of_sight(
                        player_pos, autoaim_pos, walls,
                        padding=(
                            self.line_of_sight_wall_padding
                            * self.window_controller.scale_factor
                        ),
                    )
            if autoaim_is_safe:
                authorized_until = time.monotonic() + 0.20
                if skill_type == "attack":
                    self._attack_authorized_until = authorized_until
                else:
                    self._super_authorized_until = authorized_until
            else:
                # "Hittable" describes the result of pressing auto-aim, not
                # merely whether one candidate has a clear centre ray.
                result = False
        self._decision_cache[cache_key] = result
        return result

    def estimate_navigation_cost(self, start, goal, walls):
        """Estimate route length without running A* for every visible target.

        Candidate bends are placed outside expanded wall corners.  Scoring the
        complete candidate route (instead of adding each wall's cheapest
        corner independently) prevents a close target behind several walls or
        water tiles from looking cheaper than an open, slightly farther one.
        """
        cache_key = (
            "navigation_cost", tuple(start), tuple(goal), id(walls)
        )
        cached = self._decision_cache.get(cache_key)
        if cached is not None:
            return cached
        direct = self.get_distance(start, goal)
        if not walls:
            self._decision_cache[cache_key] = direct
            return direct
        scale = self.window_controller.scale_factor or 1.0
        margin = (self.navigation_player_radius + 7.0) * scale
        geometry_key = (
            "navigation_geometry", id(walls), round(margin, 2)
        )
        expanded = self._decision_cache.get(geometry_key)
        if expanded is None:
            expanded = tuple(
                (
                    wall[0] - margin, wall[1] - margin,
                    wall[2] + margin, wall[3] + margin,
                )
                for wall in walls
            )
            self._decision_cache[geometry_key] = expanded
        candidate_corners = []
        for rect in expanded:
            # Only direct-route obstacles can provide a useful first bend.
            # This keeps target ranking linear on open maps.
            if self.segment_intersects_rect(start, goal, rect):
                clearance = 2.0 * scale
                candidate_corners.extend((
                    (rect[0] - clearance, rect[1] - clearance),
                    (rect[2] + clearance, rect[1] - clearance),
                    (rect[0] - clearance, rect[3] + clearance),
                    (rect[2] + clearance, rect[3] + clearance),
                ))

        def blockers(point_a, point_b):
            return sum(
                self.segment_intersects_rect(point_a, point_b, rect)
                for rect in expanded
            )

        direct_blockers = blockers(start, goal)
        if direct_blockers == 0:
            result = direct
        else:
            # A blocked segment is deliberately more expensive than a bend.
            # Thus an actually open target wins over a geometrically close one
            # which would require unresolved routing around multiple obstacles.
            blocked_penalty = max(120.0 * scale, direct * 0.75)
            best = direct + direct_blockers * blocked_penalty
            viable_first_bends = []
            for corner in candidate_corners:
                first_blockers = blockers(start, corner)
                if first_blockers:
                    continue
                viable_first_bends.append(corner)
                second_blockers = blockers(corner, goal)
                length = (
                    self.get_distance(start, corner)
                    + self.get_distance(corner, goal)
                )
                # Small turn cost makes equal-length choices prefer the
                # straight route and reduces target oscillation.
                best = min(
                    best,
                    length + 10.0 * scale
                    + second_blockers * blocked_penalty,
                )

            # Two bends handle U/L-shaped walls and narrow passages. Limit the
            # second-bend set to corners that can see the goal so this remains
            # much cheaper than one A* search per target.
            goal_visible_corners = [
                corner for corner in candidate_corners
                if blockers(corner, goal) == 0
            ]
            for first in viable_first_bends:
                first_length = self.get_distance(start, first)
                for second in goal_visible_corners:
                    if first == second or blockers(first, second):
                        continue
                    best = min(
                        best,
                        first_length
                        + self.get_distance(first, second)
                        + self.get_distance(second, goal)
                        + 20.0 * scale,
                    )
            result = best
        self._decision_cache[cache_key] = result
        return result

    def _recent_route_failure_penalty(self, start, goal, goal_kind):
        if (
            self._failed_goal_heading is None
            or self._failed_goal_kind != goal_kind
            or time.monotonic() >= self._failed_goal_until
        ):
            return 0.0
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1:
            return 0.0
        alignment = (
            dx * self._failed_goal_heading[0]
            + dy * self._failed_goal_heading[1]
        ) / length
        if alignment < self._failed_goal_alignment_cosine:
            return 0.0
        self.unreachable_goal_avoids += 1
        return self.TILE_SIZE * 5.0 * (
            self.window_controller.scale_factor or 1.0
        )

    def find_closest_enemy(self, enemy_data, player_coords, walls, skill_type):
        player_key = tuple(player_coords)
        cache_key = (
            "enemy", id(enemy_data), id(walls), player_key,
            skill_type, self.current_brawler,
        )
        if cache_key in self._decision_cache:
            return self._decision_cache[cache_key]
        closest_hittable_distance = float('inf')
        closest_hittable = None
        fastest_unhittable = None
        fastest_unhittable_cost = float('inf')
        matching_hittable = None
        matching_hittable_delta = float('inf')
        matching_unhittable = None
        matching_unhittable_delta = float('inf')
        previous_target = self._target_memory.get(skill_type)
        target_match_limit = self.target_match_distance * (
            self.window_controller.scale_factor or 1.0
        )
        for enemy in enemy_data:
            enemy_pos = self.get_player_position(enemy)
            distance = self.get_distance(enemy_pos, player_coords)
            previous_delta = (
                self.get_distance(enemy_pos, previous_target)
                if previous_target is not None else float('inf')
            )
            # A farther target cannot beat an already hittable target, and
            # blocked targets are only a fallback when none is hittable.
            # Preserve first-in-order tie handling without another wall scan.
            can_match_previous = (
                previous_target is not None
                and previous_delta <= target_match_limit
            )
            if (
                closest_hittable is not None
                and distance >= closest_hittable_distance
                and not can_match_previous
            ):
                continue
            if self.is_enemy_hittable(
                player_key, enemy_pos, walls, skill_type
            ):
                if distance < closest_hittable_distance:
                    closest_hittable_distance = distance
                    closest_hittable = [enemy_pos, distance]
                if can_match_previous and previous_delta < matching_hittable_delta:
                    matching_hittable_delta = previous_delta
                    matching_hittable = [enemy_pos, distance]
            else:
                navigation_cost = self.estimate_navigation_cost(
                    player_key, enemy_pos, walls
                ) + self._recent_route_failure_penalty(
                    player_key, enemy_pos, "enemy"
                )
                if navigation_cost < fastest_unhittable_cost:
                    fastest_unhittable_cost = navigation_cost
                    fastest_unhittable = [enemy_pos, distance]
                if can_match_previous and previous_delta < matching_unhittable_delta:
                    matching_unhittable_delta = previous_delta
                    matching_unhittable = [enemy_pos, distance]
        if closest_hittable:
            result = closest_hittable
            if (
                matching_hittable is not None
                and matching_hittable[1]
                    <= closest_hittable_distance * self.target_switch_ratio
            ):
                result = matching_hittable
            self._target_memory[skill_type] = result[0]
        elif fastest_unhittable:
            # Keep pursuing the same blocked target when its route remains
            # competitive. Without this, tiny detector changes can alternate
            # between two boxes/enemies on opposite sides of a wall.
            result = fastest_unhittable
            if matching_unhittable is not None:
                matching_cost = self.estimate_navigation_cost(
                    player_key, matching_unhittable[0], walls
                ) + self._recent_route_failure_penalty(
                    player_key, matching_unhittable[0], "enemy"
                )
                if matching_cost <= fastest_unhittable_cost * self.target_switch_ratio:
                    result = matching_unhittable
            self._target_memory[skill_type] = result[0]
        else:
            result = (None, None)
            self._target_memory.pop(skill_type, None)
        self._decision_cache[cache_key] = result
        return result

    def find_closest_teammate(self, teammate_data, player_coords, walls):
        cache_key = (
            "teammate", id(teammate_data), id(walls), tuple(player_coords)
        )
        if cache_key in self._decision_cache:
            return self._decision_cache[cache_key]
        teammate_anchor = None
        teammate_distance = float('inf')
        if teammate_data:
            positions = [
                self.get_player_position(teammate) for teammate in teammate_data
            ]
            if len(positions) == 1:
                closest = positions[0]
                closest_cost = self.get_distance(closest, player_coords)
                route_costs = {closest: closest_cost}
            else:
                route_costs = {
                    position: self.estimate_navigation_cost(
                        player_coords, position, walls
                    ) + self._recent_route_failure_penalty(
                        player_coords, position, "teammate"
                    )
                    for position in positions
                }
                closest = min(positions, key=route_costs.__getitem__)
                closest_cost = route_costs[closest]
            closest_distance = self.get_distance(closest, player_coords)
            teammate_anchor = closest
            teammate_distance = closest_distance
            if self.last_teammate_position is not None:
                matching = min(
                    positions,
                    key=lambda position: self.get_distance(
                        position, self.last_teammate_position
                    ),
                )
                match_delta = self.get_distance(
                    matching, self.last_teammate_position
                )
                matching_distance = self.get_distance(
                    matching, player_coords
                )
                matching_cost = route_costs[matching]
                match_limit = self.target_match_distance * (
                    self.window_controller.scale_factor or 1.0
                )
                if (
                    match_delta <= match_limit
                    and matching_cost
                        <= closest_cost * self.target_switch_ratio
                ):
                    teammate_anchor = matching
                    teammate_distance = matching_distance
            self.last_teammate_position = teammate_anchor
            direction_x = teammate_anchor[0] - player_coords[0]
            direction_y = teammate_anchor[1] - player_coords[1]
            direction_length = math.hypot(direction_x, direction_y)
            if direction_length >= 1:
                self.last_teammate_direction = (
                    direction_x / direction_length,
                    direction_y / direction_length,
                )
                self.last_teammate_search_direction = self.last_teammate_direction
                self.last_teammate_distance = direction_length
            self.last_teammate_seen_at = time.time()
            self._teammate_search_changed_at = self.last_teammate_seen_at
        elif (
            self.last_teammate_position is not None
            and time.time() - self.last_teammate_seen_at <= self.teammate_memory_duration
        ):
            # Wall-motion compensation keeps this screen-space world point
            # aligned while the camera follows us. Unlike rebuilding a fixed
            # distant anchor every tick, its distance decreases as we travel
            # and cannot make the bot overshoot for the full memory duration.
            teammate_distance = self.get_distance(
                self.last_teammate_position, player_coords
            )
            arrival_distance = self.TILE_SIZE * 0.58 * (
                self.window_controller.scale_factor or 1.0
            )
            if teammate_distance > arrival_distance:
                teammate_anchor = self.last_teammate_position
            else:
                # Search across the last approach lane instead of continuing
                # blindly through the point where the teammate disappeared.
                if self.last_teammate_direction is not None:
                    self.last_teammate_search_direction = self.normalize_move(
                        -self.last_teammate_direction[1] * self._formation_side,
                        self.last_teammate_direction[0] * self._formation_side,
                        radius=1.0,
                    )
                self._teammate_search_changed_at = time.time()
                teammate_anchor = None
                teammate_distance = float('inf')
        else:
            teammate_distance = float('inf')
        result = (teammate_anchor, teammate_distance)
        self._decision_cache[cache_key] = result
        return result

    def get_teammate_anchor(self, teammate_data):
        if not teammate_data:
            return None
        total_x = 0.0
        total_y = 0.0
        for teammate in teammate_data:
            position = self.get_player_position(teammate)
            total_x += position[0]
            total_y += position[1]
        count = len(teammate_data)
        return total_x / count, total_y / count

    def update_teammate_memory(self, teammate_data):
        teammate_anchor = self.get_teammate_anchor(teammate_data)
        if teammate_anchor is None:
            return
        self.last_teammate_position = teammate_anchor
        self.last_teammate_seen_at = time.time()

    def is_there_poison_gas(self, player_data, threshold=7000, area_from_player_checked=4.0):
        now = time.monotonic()
        player_center = self.get_player_position(player_data)
        if (
            self._poison_cache is not None
            and now - self._poison_cache_at < self.poison_cache_interval
            and self._poison_cache_player is not None
            and self.get_distance(player_center, self._poison_cache_player)
                <= 8.0 * self.window_controller.scale_factor
        ):
            self.poison_cache_count += 1
            self._latest_poison_gas = self._poison_cache
            return self._poison_cache.copy()
        self.poison_compute_count += 1
        cache_key = (
            "poison", tuple(player_data), float(threshold),
            float(area_from_player_checked),
        )
        cached = self._decision_cache.get(cache_key)
        if cached is not None:
            cached_result, cached_regions = cached
            self._latest_poison_gas = cached_result
            self._poison_danger_regions = cached_regions
            return cached_result.copy()
        actual_player_box = self.get_actual_player_box(player_data) or player_data
        px1, py1, px2, py2 = actual_player_box
        player_width = max(px2 - px1, 1)
        player_height = max(py2 - py1, 1)
        min_x = int(max(px1 - player_width*area_from_player_checked, 0))
        max_x = int(min(px2 + player_width*area_from_player_checked, self.window_controller.width))
        min_y = int(max(py1 - player_height*area_from_player_checked, 0))
        max_y = int(min(py2 + player_height*area_from_player_checked, self.window_controller.height))

        if min_x >= max_x or min_y >= max_y:
            result = {
                "up": 0,
                "down": 0,
                "left": 0,
                "right": 0,
            }
            self._latest_poison_gas = result
            self._poison_danger_regions = ()
            return result.copy()

        roi = self.frame[min_y:max_y, min_x:max_x]
        roi_shape = roi.shape[:2]
        if self._poison_buffer_shape != roi_shape:
            self._poison_buffer_shape = roi_shape
            self._poison_hsv_buffer = np.empty_like(roi)
            self._poison_mask_buffer = np.empty(roi_shape, dtype=np.uint8)
            self._poison_filtered_mask_buffer = np.empty(
                roi_shape, dtype=np.uint8
            )
        hsv_roi = cv2.cvtColor(
            roi, cv2.COLOR_RGB2HSV, dst=self._poison_hsv_buffer
        )
        mask = cv2.inRange(
            hsv_roi, POISON_LOW_HSV, POISON_HIGH_HSV,
            dst=self._poison_mask_buffer
        )
        x, y = self.get_entity_pos(actual_player_box)
        roi_w = int(max_x - min_x)
        roi_h = int(max_y - min_y)
        local_px = int(clamp(x - min_x, 0, roi_w))
        local_py = int(clamp(y - min_y, 0, roi_h))

        # Keep the actual gas geometry. Direction totals cannot distinguish a
        # safe opening beside a wall from poison covering the whole direction.
        minimum_storm_area = max(36.0, 0.0015 * float(roi_w * roi_h))
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        danger_regions = []
        storm_contours = []
        edge_margin = max(2, round(min(roi_w, roi_h) * 0.01))
        for contour in contours:
            if cv2.contourArea(contour) < minimum_storm_area:
                continue
            region_x, region_y, region_w, region_h = cv2.boundingRect(contour)
            if not (
                region_x <= edge_margin
                or region_y <= edge_margin
                or region_x + region_w >= roi_w - edge_margin
                or region_y + region_h >= roi_h - edge_margin
            ):
                # Real poison enters this player-centred scan from outside;
                # isolated matching map/UI colours inside the ROI are noise.
                continue
            storm_contours.append(contour)
            danger_regions.append((
                min_x + region_x,
                min_y + region_y,
                min_x + region_x + region_w,
                min_y + region_y + region_h,
            ))
        danger_regions = tuple(danger_regions)
        filtered_mask = self._poison_filtered_mask_buffer
        filtered_mask.fill(0)
        if storm_contours:
            cv2.drawContours(
                filtered_mask, storm_contours, -1, 255, thickness=cv2.FILLED
            )
        counts = {
            "up": cv2.countNonZero(filtered_mask[:local_py, :roi_w]),
            "down": cv2.countNonZero(filtered_mask[local_py:roi_h, :roi_w]),
            "left": cv2.countNonZero(filtered_mask[:roi_h, :local_px]),
            "right": cv2.countNonZero(filtered_mask[:roi_h, local_px:roi_w]),
        }
        result = {
            direction: count if count > threshold else 0
            for direction, count in counts.items()
        }

        if self.verbose_debug:
            print("Poison gas pixels:", counts)

            ts = int(time.time())

            debug_regions = {
                "up": roi[0:local_py, 0:roi_w],
                "down": roi[local_py:roi_h, 0:roi_w],
                "left": roi[0:roi_h, 0:local_px],
                "right": roi[0:roi_h, local_px:roi_w],
            }

            for direction, img in debug_regions.items():
                if img.size > 0:
                    cv2.imwrite(
                        f"debug_frames/poison_gas_{direction}_debug_{ts}.png",
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    )

        self._decision_cache[cache_key] = (result, danger_regions)
        # Callers always receive a copy, so both internal caches can safely
        # share the same read-only-by-convention result dictionary.
        self._poison_cache = result
        self._latest_poison_gas = result
        self._poison_danger_regions = danger_regions
        self._poison_cache_at = now
        self._poison_cache_player = player_center
        return result.copy()

    def _directional_poison_danger(self):
        poison = self._latest_poison_gas
        highest = max(poison.values(), default=0)
        if highest <= 0:
            return (0.0, 0.0, 0.0, 0.0)
        inverse = 1.0 / highest
        return (
            poison.get("up", 0) * inverse,
            poison.get("down", 0) * inverse,
            poison.get("left", 0) * inverse,
            poison.get("right", 0) * inverse,
        )

    def _movement_poison_risk(self, movement):
        x, y = movement
        magnitude = math.hypot(x, y)
        if magnitude < 1:
            return float("inf")
        up, down, left, right = self._directional_poison_danger()
        return (
            (left if x < 0 else right if x > 0 else 0.0)
            * abs(x) / magnitude
            + (up if y < 0 else down if y > 0 else 0.0)
            * abs(y) / magnitude
        )

    def get_main_data(self, frame, cache_for_reuse=False):
        prepared_frame = self._prepared_frame_ref() if self._prepared_frame_ref else None
        if prepared_frame is frame and self._prepared_main_data is not None:
            data = self._prepared_main_data
            self._prepared_frame_ref = None
            self._prepared_main_data = None
            return data
        data = self.Detect_main_info.detect_objects(frame, conf_tresh=self.entity_detection_confidence)
        data = self.detection_stabilizer.update(data)
        if cache_for_reuse:
            self._prepared_frame_ref = weakref.ref(frame)
            self._prepared_main_data = data
        return data

    def probe_match_started(self, frame):
        self.get_main_data(frame, cache_for_reuse=True)
        return self.detection_stabilizer.was_observed("player")

    def reset_perception(self):
        self.detection_stabilizer.reset()
        self.wall_scene_gate.reset()
        self._decision_cache.clear()
        self.last_walls_data = []
        self.last_bushes_data = []
        self._wall_tracks = []
        self._wall_motion_velocity = (0.0, 0.0)
        self._wall_motion_observed_at = 0.0
        self._projected_walls_key = None
        self._projected_walls = []
        self._wall_navigation_ready = False
        self._wall_navigation_initialized = False
        self._wall_navigation_grace_started_at = 0.0
        self._last_obstacle_refresh_request = 0.0
        self.time_since_walls_checked = 0
        self._target_memory.clear()
        self._prepared_frame_ref = None
        self._prepared_main_data = None
        self.last_attack_at = 0.0
        self.is_gadget_ready = False
        self.is_hypercharge_ready = False
        self.is_super_ready = False
        self.time_since_gadget_checked = 0.0
        self.time_since_hypercharge_checked = 0.0
        self.time_since_super_checked = 0.0
        self._attack_authorized_until = 0.0
        self._super_authorized_until = 0.0
        self._current_enemy_data = ()
        self.last_movement = ''
        self.last_movement_change_time = 0.0
        self._last_navigation_goal = "initializing"
        self.persistent_data["time_since_holding_attack"] = None
        self.persistent_data["charged_attack_ready"] = False
        self.persistent_data["navigation_goal"] = "initializing"
        self.persistent_data["combat_mode"] = None
        self.persistent_data["poison_heading"] = None
        self.objective_position = None
        self.objective_expires_at = 0.0
        self.objective_stability = 0
        self.objective_updated_at = 0.0
        self.last_teammate_position = None
        self.last_teammate_direction = None
        self.last_teammate_search_direction = None
        self._teammate_search_changed_at = 0.0
        self.last_teammate_distance = 0.0
        self.last_teammate_seen_at = 0.0
        self.path_planner.reset()
        self.fine_path_planner.reset()
        self._failed_goal_heading = None
        self._failed_goal_kind = None
        self._failed_goal_until = 0.0
        self._route_failure_goal = None
        self._route_failure_count = 0
        self._route_failure_at = 0.0
        self._route_recovery_until = 0.0
        self._detour_heading = None
        self._detour_goal_heading = None
        self._detour_expires_at = 0.0
        self._direct_clear_since = 0.0
        self._exploration_heading = (0.0, -JOYSTICK_RADIUS)
        self._exploration_expires_at = 0.0
        self._exploration_index = 0
        self._combat_strafe_side = 1
        self._combat_strafe_until = 0.0
        self._formation_side = -1
        self._obstacle_progress_center = None
        self._obstacle_progress_at = 0.0
        self._obstacle_escape_side = 1
        self._obstacle_stuck_count = 0
        self._poison_cache = None
        self._poison_cache_at = 0.0
        self._poison_cache_player = None
        self._latest_poison_gas = {
            "up": 0, "down": 0, "left": 0, "right": 0,
        }
        self._poison_danger_regions = ()
        self._power_cube_cache = []
        self._power_cube_cache_at = 0.0
        self._power_cube_cache_player = None
        self._power_cube_buffer_shape = None
        self._power_cube_hsv_buffer = None
        self._power_cube_mask_buffer = None
        self._power_cube_core_mask_buffer = None
        self._power_cube_tracks = []
        self._last_power_cube_target = None
        self._strategic_heading = None
        self._strategic_goal = "initializing"
        self._strategic_commit_until = 0.0
        for ability in self._ability_used_at:
            self._ability_used_at[ability] = 0.0

    def is_path_blocked(self, player_box, move_direction, walls, distance=None):
        if distance is None:
            distance = self.TILE_SIZE*self.window_controller.scale_factor
        movement = self.movement_to_vector(move_direction)
        if movement is None:
            return False

        magnitude = math.hypot(movement[0], movement[1])
        if magnitude < 1:
            return False

        cache_key = (
            "path_blocked", tuple(player_box[:4]),
            round(movement[0], 2), round(movement[1], 2),
            round(float(distance), 2), id(walls),
        )
        if cache_key in self._decision_cache:
            return self._decision_cache[cache_key]

        dx = movement[0] / magnitude * distance
        dy = movement[1] / magnitude * distance
        hit_circle_center, hit_circle_radius = self.get_player_hit_circle(player_box)
        if hit_circle_center is None:
            return False
        # Match the clearance used by A* smoothing. Without a small common
        # margin, a waypoint could be safe while the final joystick guard
        # still approved a detector-jittered wall-edge collision.
        hit_circle_radius += (
            self.navigation_wall_padding * self.window_controller.scale_factor
        )

        new_pos = (hit_circle_center[0] + dx, hit_circle_center[1] + dy)
        nearby_walls = self._nearby_walls(
            player_box, walls, distance + hit_circle_radius
        )
        collision_walls = nearby_walls
        for wall in nearby_walls:
            expanded = (
                wall[0] - hit_circle_radius,
                wall[1] - hit_circle_radius,
                wall[2] + hit_circle_radius,
                wall[3] + hit_circle_radius,
            )
            if not (
                expanded[0] <= hit_circle_center[0] <= expanded[2]
                and expanded[1] <= hit_circle_center[1] <= expanded[3]
            ):
                continue
            _, exit_x, exit_y = min(
                (
                    (hit_circle_center[0] - expanded[0], -1.0, 0.0),
                    (expanded[2] - hit_circle_center[0], 1.0, 0.0),
                    (hit_circle_center[1] - expanded[1], 0.0, -1.0),
                    (expanded[3] - hit_circle_center[1], 0.0, 1.0),
                ),
                key=lambda item: item[0],
            )
            if dx * exit_x + dy * exit_y <= 1e-6:
                continue
            # The swept-circle test necessarily reports a collision at t=0
            # for detector-padding overlap. Ignore only this one rectangle and
            # only while moving through its nearest exit; all other terrain
            # remains fully collision checked.
            collision_walls = [
                candidate for candidate in nearby_walls
                if candidate is not wall
            ]
            self.overlap_escape_approvals += 1
            break
        result = self.walls_block_swept_circle(
            hit_circle_center, new_pos, hit_circle_radius, collision_walls
        )
        self._decision_cache[cache_key] = result
        return result

    def _nearby_walls(self, player_box, walls, radius):
        if not walls:
            return []
        center, _ = self.get_player_hit_circle(player_box)
        if center is None:
            return walls
        cache_key = (
            "nearby_walls", id(walls), tuple(player_box[:4]), round(float(radius), 1)
        )
        cached = self._decision_cache.get(cache_key)
        if cached is not None:
            return cached
        x, y = center
        nearby = [
            wall for wall in walls
            if wall[2] >= x - radius and wall[0] <= x + radius
            and wall[3] >= y - radius and wall[1] <= y + radius
        ]
        self._decision_cache[cache_key] = nearby
        return nearby

    @staticmethod
    def _segment_blocked(start, end, radius, walls):
        return Play.walls_block_swept_circle(start, end, radius, walls)

    def _navigation_visible_radius(self):
        # centered_wall_crop_size is already measured in pixels of the current
        # captured frame. Multiplying it by scale_factor again shortened A*'s
        # lookahead disproportionately on low-resolution devices.
        visible_size = self.centered_wall_crop_size
        if self.window_controller.width:
            visible_size = min(visible_size, self.window_controller.width)
        if self.window_controller.height:
            visible_size = min(visible_size, self.window_controller.height)
        return visible_size * 0.45

    def _navigation_goal_distance(self, magnitude):
        """Do not plan beyond a visible, position-based objective."""
        scale = self.window_controller.scale_factor or 1.0
        precise_goals = {
            "objective", "power_cube", "fallback_power_cube",
            "teammate", "teammate_formation",
            "enemy_approach", "fallback_enemy", "fallback_objective",
            "fallback_teammate", "route_recovery_teammate",
        }
        if self.persistent_data.get("navigation_goal") in precise_goals:
            # A wall behind a nearby target is irrelevant. Looking several
            # tiles beyond it made the old navigator take a detour although
            # the direct segment to the target was completely clear.
            return min(
                max(magnitude, self.TILE_SIZE * 0.55 * scale),
                self._navigation_visible_radius(),
            )
        return min(
            max(
                magnitude,
                self.TILE_SIZE * self.pathfinding_horizon_tiles * scale,
            ),
            self._navigation_visible_radius(),
        )

    def _best_open_movement(self, movement, player_box, walls):
        """Choose the next waypoint on a shortest local A* route."""
        start, player_radius = self.get_player_hit_circle(player_box)
        magnitude = math.hypot(*movement)
        if start is None or magnitude < 1:
            return movement

        scale_factor = self.window_controller.scale_factor
        visible_navigation_area = min(
            self.centered_wall_crop_size,
            self.window_controller.width or self.centered_wall_crop_size,
            self.window_controller.height or self.centered_wall_crop_size,
        )
        self.path_planner.set_area_size(visible_navigation_area)
        self.fine_path_planner.set_area_size(visible_navigation_area)
        self.path_planner.set_wall_padding(
            self.navigation_wall_padding * scale_factor
        )
        self.fine_path_planner.set_wall_padding(
            self.navigation_wall_padding * scale_factor
        )
        known_distance = self._navigation_goal_distance(magnitude)
        goal = (
            start[0] + movement[0] / magnitude * known_distance,
            start[1] + movement[1] / magnitude * known_distance,
        )
        nearby = self._nearby_walls(player_box, walls, known_distance + player_radius)
        navigation_now = time.monotonic()
        poison_danger = self._directional_poison_danger()
        nearest_wall = None
        if nearby:
            nearest_wall = None
            corridor_padding = player_radius + (
                self.navigation_wall_padding * scale_factor
            )
            blocking_walls = [
                wall for wall in nearby
                if self.segment_intersects_rect(
                    start,
                    goal,
                    (
                        wall[0] - corridor_padding,
                        wall[1] - corridor_padding,
                        wall[2] + corridor_padding,
                        wall[3] + corridor_padding,
                    ),
                )
            ]
            obstacle_candidates = blocking_walls or nearby
            if self._obstacle_progress_center is not None:
                tracked_wall = min(
                    obstacle_candidates,
                    key=lambda wall: self.get_distance(
                        (
                            (wall[0] + wall[2]) * 0.5,
                            (wall[1] + wall[3]) * 0.5,
                        ),
                        self._obstacle_progress_center,
                    ),
                )
                tracked_center = (
                    (tracked_wall[0] + tracked_wall[2]) * 0.5,
                    (tracked_wall[1] + tracked_wall[3]) * 0.5,
                )
                if self.get_distance(
                    tracked_center, self._obstacle_progress_center
                ) <= self.TILE_SIZE * 0.85 * scale_factor:
                    nearest_wall = tracked_wall
            if nearest_wall is None:
                # Prefer the first obstacle in the actual goal corridor. A
                # physically closer side/behind wall must not choose the
                # tangent for an unrelated obstacle and send the bot in an
                # apparently random direction.
                nearest_wall = min(
                    obstacle_candidates,
                    key=lambda wall: self.point_rect_distance_sq(
                        start, wall[:4]
                    ),
                )
                self._obstacle_progress_center = None
            wall_center = (
                (nearest_wall[0] + nearest_wall[2]) * 0.5,
                (nearest_wall[1] + nearest_wall[3]) * 0.5,
            )
            forward_progress = 0.0
            progress_threshold = 10.0 * scale_factor
            wrong_way = False
            stuck_interval = 1.2
            if self._obstacle_progress_center is None:
                self._obstacle_progress_center = wall_center
                self._obstacle_progress_at = navigation_now
            else:
                progress_heading = (
                    self._detour_heading
                    if self._detour_heading is not None
                    and navigation_now < self._detour_expires_at
                    else movement
                )
                progress_length = math.hypot(*progress_heading)
                camera_delta_x = (
                    wall_center[0] - self._obstacle_progress_center[0]
                )
                camera_delta_y = (
                    wall_center[1] - self._obstacle_progress_center[1]
                )
                # The camera moves opposite to the controlled brawler. Only
                # motion along the chosen route proves useful progress;
                # lateral detector jitter and backwards oscillation do not.
                forward_progress = -(
                    camera_delta_x * progress_heading[0]
                    + camera_delta_y * progress_heading[1]
                ) / max(1.0, progress_length)
                wrong_way = forward_progress <= -progress_threshold
                stuck_interval = 0.65 if wrong_way else 1.2

            if (
                self._obstacle_progress_center is not None
                and forward_progress >= progress_threshold
            ):
                self._obstacle_progress_center = wall_center
                self._obstacle_progress_at = navigation_now
                self._obstacle_stuck_count = 0
            elif (
                self._obstacle_progress_center is not None
                and navigation_now - self._obstacle_progress_at
                    >= stuck_interval
            ):
                self._obstacle_stuck_count += 1
                if wrong_way:
                    self.wrong_way_corrections += 1
                # One slow/noisy wall refresh is not proof that the chosen
                # side is wrong. Flip only after two consecutive intervals
                # without camera-relative obstacle progress.
                if self._obstacle_stuck_count == 2:
                    self._obstacle_escape_side *= -1
                wall_width = nearest_wall[2] - nearest_wall[0]
                wall_height = nearest_wall[3] - nearest_wall[1]
                if self._obstacle_stuck_count >= 3:
                    self._detour_heading = self.normalize_move(
                        start[0] - wall_center[0],
                        start[1] - wall_center[1],
                    )
                    self._obstacle_stuck_count = 0
                elif wall_width >= wall_height:
                    left_distance = abs(start[0] - nearest_wall[0])
                    right_distance = abs(nearest_wall[2] - start[0])
                    tangent_side = (
                        -1 if left_distance < right_distance
                        else 1 if right_distance < left_distance
                        else self._obstacle_escape_side
                    )
                    self._detour_heading = (
                        JOYSTICK_RADIUS * tangent_side, 0.0
                    )
                else:
                    top_distance = abs(start[1] - nearest_wall[1])
                    bottom_distance = abs(nearest_wall[3] - start[1])
                    tangent_side = (
                        -1 if top_distance < bottom_distance
                        else 1 if bottom_distance < top_distance
                        else self._obstacle_escape_side
                    )
                    self._detour_heading = (
                        0.0, JOYSTICK_RADIUS * tangent_side
                    )
                self._detour_expires_at = (
                    navigation_now + self.detour_commit_seconds
                )
                self._obstacle_progress_center = wall_center
                self._obstacle_progress_at = navigation_now
                self.path_planner.reset()
                self.fine_path_planner.reset()
        else:
            self._obstacle_progress_center = None
            self._obstacle_progress_at = 0.0
            self._obstacle_stuck_count = 0
        if (
            self._detour_heading is not None
            and max(poison_danger) > 0
            and self._movement_poison_risk(self._detour_heading)
                > self._movement_poison_risk(movement) + 0.05
        ):
            # Storm safety overrides a side commitment made before the gas
            # reached this local area.
            self._detour_heading = None
            self._detour_goal_heading = None
            self._detour_expires_at = 0.0
        if self._detour_heading is not None and magnitude >= 1:
            detour_length = math.hypot(*self._detour_heading)
            detour_alignment = (
                self._detour_heading[0] * movement[0]
                + self._detour_heading[1] * movement[1]
            ) / max(1.0, detour_length * magnitude)
            short_forward = self.TILE_SIZE * 0.52 * scale_factor
            if (
                detour_alignment < 0.0
                and not self.is_path_blocked(
                    player_box, movement, walls, distance=short_forward
                )
                and self._movement_poison_risk(movement) < 0.08
            ):
                # Never let an old side commitment pull away from a goal when
                # useful forward ground is open right now.
                self._detour_heading = None
                self._detour_goal_heading = None
                self._detour_expires_at = 0.0
                self.short_forward_overrides += 1
        preferred_heading = (
            self._detour_heading
            if self._detour_heading is not None
            and navigation_now < self._detour_expires_at
            else movement
        )
        path = self.path_planner.plan(
            start, goal, nearby, player_radius, navigation_now,
            preferred_heading, poison_danger, self._poison_danger_regions,
        )
        needs_fine_path = not path
        if path:
            first_dx = path[0][0] - start[0]
            first_dy = path[0][1] - start[1]
            first_length = math.hypot(first_dx, first_dy)
            first_alignment = (
                first_dx * movement[0] + first_dy * movement[1]
            ) / max(1.0, first_length * magnitude)
            needs_fine_path = len(path) > 1 or first_alignment < 0.72
        if needs_fine_path:
            # A coarse route can be valid but unnecessarily long when a
            # narrow safe lane vanishes between 36 px cell centres. Refine
            # only complex/sideways routes, never ordinary open travel.
            self.fine_path_requests += 1
            fine_path = self.fine_path_planner.plan(
                start, goal, nearby, player_radius, navigation_now,
                preferred_heading, poison_danger,
                self._poison_danger_regions,
            )
            if fine_path:
                self.fine_path_successes += 1
                def route_length(route):
                    total = 0.0
                    previous = start
                    for point in route:
                        total += self.get_distance(previous, point)
                        previous = point
                    return total

                if (
                    not path
                    or route_length(fine_path)
                        <= route_length(path) * 1.03
                ):
                    if path:
                        self.fine_path_upgrades += 1
                    path = fine_path
        if path:
            self._route_failure_goal = None
            self._route_failure_count = 0
            if self._failed_goal_heading is not None:
                failed_alignment = (
                    movement[0] * self._failed_goal_heading[0]
                    + movement[1] * self._failed_goal_heading[1]
                ) / magnitude
                if failed_alignment >= self._failed_goal_alignment_cosine:
                    self._failed_goal_heading = None
                    self._failed_goal_kind = None
                    self._failed_goal_until = 0.0
            # The planner has already smoothed this into the farthest safe
            # waypoint while checking both walls and storm. A second shortcut
            # here used to ignore storm geometry and destabilize the route.
            waypoint = path[0]
            result = waypoint[0] - start[0], waypoint[1] - start[1]
            previous_detour = self._detour_heading
            if (
                previous_detour is not None
                and navigation_now < self._detour_expires_at
            ):
                previous_length = math.hypot(*previous_detour)
                result_length = math.hypot(*result)
                if previous_length >= 1 and result_length >= 1:
                    direction_cosine = (
                        previous_detour[0] * result[0]
                        + previous_detour[1] * result[1]
                    ) / (previous_length * result_length)
                    commitment_lookahead = (
                        self.TILE_SIZE * 1.5 * scale_factor
                    )
                    strategic_alignment = (
                        previous_detour[0] * movement[0]
                        + previous_detour[1] * movement[1]
                    ) / max(1.0, previous_length * magnitude)
                    short_forward_blocked = self.is_path_blocked(
                        player_box, movement, walls,
                        distance=self.TILE_SIZE * 0.52 * scale_factor,
                    )
                    if (
                        direction_cosine < 0.25
                        and (
                            strategic_alignment >= 0.0
                            or short_forward_blocked
                        )
                        and not self.is_path_blocked(
                            player_box, previous_detour, walls,
                            distance=commitment_lookahead,
                        )
                        and self._movement_poison_risk(previous_detour)
                            <= self._movement_poison_risk(result) + 0.05
                    ):
                        # Keep the original expiry. Extending it here would
                        # prevent a legitimate turn after reaching a corner.
                        self.detour_reversals_prevented += 1
                        return previous_detour
            self._detour_heading = result
            self._detour_goal_heading = (
                movement[0] / magnitude, movement[1] / magnitude
            )
            self._detour_expires_at = navigation_now + self.detour_commit_seconds
            return result

        failed_goal = self.persistent_data.get("navigation_goal", "unknown")
        self.route_failure_counts[failed_goal] = (
            self.route_failure_counts.get(failed_goal, 0) + 1
        )
        if (
            failed_goal == self._route_failure_goal
            and navigation_now - self._route_failure_at <= 2.0
        ):
            # Count at human reaction cadence, not once per inference frame.
            if navigation_now - self._route_failure_at >= 0.30:
                self._route_failure_count += 1
                self._route_failure_at = navigation_now
        else:
            self._route_failure_goal = failed_goal
            self._route_failure_count = 1
            self._route_failure_at = navigation_now
        if (
            self._route_failure_count >= 3
            and failed_goal in {
                "enemy_approach", "objective", "fallback_enemy",
                "fallback_objective",
            }
        ):
            # Reposition briefly, then retry the still-visible high-priority
            # objective from a different lane. This avoids indefinitely
            # repeating an impossible route while preserving enemy/cube
            # priority after the short recovery.
            self._route_recovery_until = navigation_now + 0.9
            self._route_failure_count = 0
            self.route_recoveries += 1

        self._failed_goal_heading = (
            movement[0] / magnitude, movement[1] / magnitude
        )
        if "enemy" in failed_goal:
            self._failed_goal_kind = "enemy"
        elif "teammate" in failed_goal:
            self._failed_goal_kind = "teammate"
        elif "power_cube" in failed_goal:
            self._failed_goal_kind = "power_cube"
        elif "objective" in failed_goal:
            self._failed_goal_kind = "objective"
        else:
            self._failed_goal_kind = failed_goal
        self._failed_goal_until = navigation_now + 1.5

        # Fully enclosed/noisy detections: deterministic angular fallback.
        side = self._obstacle_escape_side
        offsets = (
            side * math.pi / 8, -side * math.pi / 8,
            side * math.pi / 4, -side * math.pi / 4,
            side * math.pi / 2, -side * math.pi / 2,
            side * 3 * math.pi / 4, -side * 3 * math.pi / 4,
            math.pi,
        )
        lookahead = self.TILE_SIZE * 1.5 * self.window_controller.scale_factor
        candidates = [
            self.rotate_movement(movement, offset) for offset in offsets
        ]
        # Long water/wall strips can span the complete local A* window. Add
        # geometry-derived tangents and an outward normal so the fallback
        # follows the obstacle edge instead of repeatedly pushing into it.
        if nearest_wall is not None:
            wall_x1, wall_y1, wall_x2, wall_y2 = nearest_wall[:4]
            wall_width = wall_x2 - wall_x1
            wall_height = wall_y2 - wall_y1
            wall_center_x = (wall_x1 + wall_x2) * 0.5
            wall_center_y = (wall_y1 + wall_y2) * 0.5
            away = self.normalize_move(
                start[0] - wall_center_x,
                start[1] - wall_center_y,
            )
            # When a noisy box overlaps the estimated player circle, moving
            # away from the wall centre can still be diagonal to its closest
            # edge and collide with a neighbouring tile. Add the shortest
            # axis-aligned exit from the inflated wall first.
            clearance_radius = player_radius + (
                self.navigation_wall_padding * scale_factor
            )
            inflated = (
                wall_x1 - clearance_radius, wall_y1 - clearance_radius,
                wall_x2 + clearance_radius, wall_y2 + clearance_radius,
            )
            escape = None
            if (
                inflated[0] <= start[0] <= inflated[2]
                and inflated[1] <= start[1] <= inflated[3]
            ):
                _, escape_x, escape_y = min(
                    (
                        (start[0] - inflated[0], -1.0, 0.0),
                        (inflated[2] - start[0], 1.0, 0.0),
                        (start[1] - inflated[1], 0.0, -1.0),
                        (inflated[3] - start[1], 0.0, 1.0),
                    ),
                    key=lambda item: item[0],
                )
                escape = self.normalize_move(escape_x, escape_y)
            if wall_width >= wall_height:
                left_distance = abs(start[0] - wall_x1)
                right_distance = abs(wall_x2 - start[0])
                nearest_side = (
                    -1 if left_distance < right_distance
                    else 1 if right_distance < left_distance
                    else self._obstacle_escape_side
                )
                tangents = (
                    (JOYSTICK_RADIUS * nearest_side, 0.0),
                    (-JOYSTICK_RADIUS * nearest_side, 0.0),
                )
            else:
                top_distance = abs(start[1] - wall_y1)
                bottom_distance = abs(wall_y2 - start[1])
                nearest_side = (
                    -1 if top_distance < bottom_distance
                    else 1 if bottom_distance < top_distance
                    else self._obstacle_escape_side
                )
                tangents = (
                    (0.0, JOYSTICK_RADIUS * nearest_side),
                    (0.0, -JOYSTICK_RADIUS * nearest_side),
                )
            geometry_candidates = [away, *tangents]
            if escape is not None:
                geometry_candidates.insert(0, escape)
            # The exact overlap exit must precede generic angular guesses;
            # stable sorting by poison risk preserves this order when risks
            # are equal (the common non-storm case).
            candidates = geometry_candidates + candidates
        # Preserve order while removing equivalent vectors before collision
        # checks; this bounds fallback work when rotated and tangent choices
        # overlap.
        unique_candidates = []
        seen_candidates = set()
        for candidate in candidates:
            candidate_key = (round(candidate[0], 2), round(candidate[1], 2))
            if candidate_key in seen_candidates:
                continue
            seen_candidates.add(candidate_key)
            unique_candidates.append(candidate)
        candidates = unique_candidates
        desired_length = math.hypot(*movement)

        def fallback_score(candidate):
            candidate_length = math.hypot(*candidate)
            alignment = (
                (candidate[0] * movement[0] + candidate[1] * movement[1])
                / (candidate_length * desired_length)
                if candidate_length >= 1 and desired_length >= 1 else -1.0
            )
            # Risk matters, but tiny mask noise must not beat a direction that
            # actually advances toward the objective.
            return self._movement_poison_risk(candidate) * 3.0 - alignment

        candidates.sort(key=fallback_score)
        # Failure at a long lookahead does not mean the bot is physically
        # trapped. A human advances to the next safe decision point and plans
        # again. Try progressively shorter, still collision-checked horizons
        # so a crowded intersection does not become an unnecessary STOP.
        escape_horizons = (
            lookahead,
            self.TILE_SIZE * 0.9 * scale_factor,
            self.TILE_SIZE * 0.45 * scale_factor,
            self.TILE_SIZE * 0.20 * scale_factor,
        )
        for horizon_index, escape_horizon in enumerate(escape_horizons):
            open_candidates = [
                candidate for candidate in candidates
                if not self.is_path_blocked(
                    player_box, candidate, walls, distance=escape_horizon
                )
            ]
            forward_candidates = []
            for candidate in open_candidates:
                candidate_length = math.hypot(*candidate)
                if (
                    candidate_length >= 1
                    and (
                        candidate[0] * movement[0]
                        + candidate[1] * movement[1]
                    ) / max(1.0, candidate_length * magnitude) >= -0.05
                ):
                    forward_candidates.append(candidate)
            for candidate in forward_candidates or open_candidates:
                self._detour_heading = candidate
                self._detour_goal_heading = (
                    movement[0] / magnitude, movement[1] / magnitude
                )
                # A short-horizon escape must be reconsidered quickly; a full
                # detour can retain the normal side commitment.
                commitment = (
                    self.detour_commit_seconds
                    if horizon_index == 0
                    else min(0.25, self.detour_commit_seconds)
                )
                self._detour_expires_at = navigation_now + commitment
                if horizon_index:
                    self.progressive_escape_moves += 1
                return candidate
        # Never knowingly push the joystick through terrain. The previous
        # fallback returned a stale committed direction even after every
        # candidate failed collision checks, which caused wall/water walking.
        # Poison never authorizes movement through known terrain. If every
        # candidate is blocked, stop, refresh geometry, and retry next tick.
        # The former gas exception deliberately selected candidates[0] even
        # after it had failed collision checks, causing exactly the observed
        # wall/water pushing near the closing storm.
        fallback = (0.0, 0.0)
        self.collision_stops += 1
        self.path_planner.reset()
        self.fine_path_planner.reset()
        refresh_now = time.monotonic()
        if (
            refresh_now - self._last_obstacle_refresh_request
            >= self.wall_obstacle_refresh_interval
        ):
            self.wall_scene_gate.request_refresh()
            self._last_obstacle_refresh_request = refresh_now
        self._detour_heading = fallback
        self._detour_goal_heading = (
            movement[0] / magnitude, movement[1] / magnitude
        )
        self._detour_expires_at = navigation_now + self.detour_commit_seconds
        return fallback

    def navigation_correction(self, movement, player_box, walls):
        if not self._wall_navigation_ready:
            if not self._wall_navigation_initialized:
                now = time.monotonic()
                if self._wall_navigation_grace_started_at <= 0.0:
                    self._wall_navigation_grace_started_at = now
                if (
                    now - self._wall_navigation_grace_started_at
                    <= self.wall_navigation_startup_grace
                ):
                    return movement
            return (0.0, 0.0)
        magnitude = math.hypot(*movement)
        if magnitude >= 1 and self._detour_goal_heading is not None:
            goal_heading = movement[0] / magnitude, movement[1] / magnitude
            if (
                goal_heading[0] * self._detour_goal_heading[0]
                + goal_heading[1] * self._detour_goal_heading[1]
                < self.detour_goal_reset_cosine
            ):
                self._detour_heading = None
                self._detour_goal_heading = None
                self._detour_expires_at = 0.0
        scale_factor = self.window_controller.scale_factor
        direct_distance = self._navigation_goal_distance(magnitude)
        direct_is_blocked = self.is_path_blocked(
            player_box, movement, walls, distance=direct_distance
        )
        if direct_is_blocked:
            now = time.monotonic()
            if (
                now - self._last_obstacle_refresh_request
                >= self.wall_obstacle_refresh_interval
            ):
                self.wall_scene_gate.request_refresh()
                self._last_obstacle_refresh_request = now
        direct_poison_risk = self._movement_poison_risk(movement)
        # Straight movement wins only when it is both unobstructed and safe.
        # Previously a clear line bypassed A* even when it pointed into gas.
        if magnitude < 1 or (
            not direct_is_blocked and direct_poison_risk < 0.08
        ):
            # Move directly when it is clear, but retain the committed side
            # until its short expiry. If a wall box flickers back next frame,
            # A* will continue on the same side instead of oscillating.
            now = time.monotonic()
            if magnitude < 1:
                self._direct_clear_since = 0.0
            elif self._direct_clear_since <= 0:
                self._direct_clear_since = now
            if (
                now >= self._detour_expires_at
                or self._direct_clear_since > 0
                and now - self._direct_clear_since >= 0.25
            ):
                self._detour_heading = None
                self._detour_goal_heading = None
                self._detour_expires_at = 0.0
            if magnitude >= 1:
                self.direct_navigation_moves += 1
            return movement

        self._direct_clear_since = 0.0
        self.planned_navigation_moves += 1
        return self._best_open_movement(movement, player_box, walls)

    @staticmethod
    def validate_game_data(data):
        if not isinstance(data, dict):
            return False

        # Temporal perception intentionally exposes known classes with empty
        # lists. A key therefore no longer proves that an object was detected.
        for name in ("player", "enemy", "teammate", "wall", "bush"):
            boxes = data.get(name) or []
            # Detector/stabilizer output is normally already valid. Preserve
            # those lists so the hot loop allocates nothing here. Fall back to
            # the original filtering behavior only for malformed data.
            if isinstance(boxes, list):
                all_valid = True
                for box in boxes:
                    if box is None or len(box) < 4:
                        all_valid = False
                        break
                if all_valid:
                    data[name] = boxes
                    continue
            data[name] = [
                box for box in boxes if box is not None and len(box) >= 4
            ]

        return data if data["player"] else False

    def track_no_detections(self, data):
        if not data:
            data = {
                "enemy": None,
                "player": None
            }
        for key in self.time_since_detections:
            if key in data and data[key]:
                self.time_since_detections[key] = time.time()

    def do_movement(self, movement):
        movement_vector = self.movement_to_vector(movement)
        if movement_vector is None:
            self.window_controller.release_movement()
            return
        self.window_controller.move(*movement_vector)

    def get_brawler_range(self, brawler):
        if self.brawler_ranges is None:
            self.brawler_ranges = self.load_brawler_ranges(self.brawlers_info)
        return self.brawler_ranges[brawler]

    @staticmethod
    def normalize_move(x, y, radius=JOYSTICK_RADIUS):
        length = math.hypot(x, y)
        if length <= 0:
            return (0.0, 0.0)
        scale = radius / length
        return (x * scale, y * scale)

    def clamp_movement(self, movement):
        x, y = movement
        length = math.hypot(x, y)
        if length <= 0:
            return (0.0, 0.0)
        scale = JOYSTICK_RADIUS / length
        target_x = x * scale * self.window_controller.width_ratio
        target_y = y * scale * self.window_controller.height_ratio
        return target_x, target_y

    def movement_is_similar(self, first, second):
        first_vector = self.movement_to_vector(first)
        second_vector = self.movement_to_vector(second)
        if first_vector is None or second_vector is None:
            return False
        first_length = math.hypot(*first_vector)
        second_length = math.hypot(*second_vector)
        if first_length < 1 or second_length < 1:
            return first_length < 1 and second_length < 1
        cosine = (
            first_vector[0] * second_vector[0] + first_vector[1] * second_vector[1]
        ) / (first_length * second_length)
        return cosine >= self.movement_direction_hysteresis

    def loop(self, brawler, data, current_time):
        if not data or not data.get("player"):
            self.window_controller.release_movement()
            return None
        # Cache is valid only for this immutable frame's decision script.
        self._decision_cache.clear()
        self._attack_authorized_until = 0.0
        self._super_authorized_until = 0.0
        self._current_enemy_data = tuple(data['enemy'])
        data['power_cube'] = self.detect_power_cubes(
            data['player'][0],
            tuple(data.get('player') or ())
                + tuple(data.get('enemy') or ())
                + tuple(data.get('teammate') or ()),
            tuple(data.get('wall') or ()),
        )
        # Navigation safety must not depend on the selected playstyle opting
        # into poison detection.
        self.is_there_poison_gas(data['player'][0])
        if self.context is None:
            self.context = {
                'brawlers_info': self.brawlers_info,
                'must_brawler_hold_attack': self.must_brawler_hold_attack,
                'TILE_SIZE': self.TILE_SIZE*self.window_controller.scale_factor,
                'get_entity_pos': self.get_entity_pos,
                'get_player_position': self.get_player_position,
                'get_distance': self.get_distance,
                'get_actual_player_box': self.get_actual_player_box,
                'get_brawler_range': self.get_brawler_range,
                'is_there_enemy': self.is_there_enemy,
                'attack': self.attack,
                'use_hypercharge': self.use_hypercharge,
                'use_super': self.use_super,
                'use_gadget': self.use_gadget,
                'get_random_movement': self.get_random_movement,
                'get_exploration_movement': self.get_exploration_movement,
                'get_teammate_search_movement': self.get_teammate_search_movement,
                'get_combat_strafe_movement': self.get_combat_strafe_movement,
                'get_teammate_formation_movement': self.get_teammate_formation_movement,
                'remember_objective': self.remember_objective,
                'get_recent_objective_movement': self.get_recent_objective_movement,
                'seconds_to_hold_attack_after_reaching_max': self.seconds_to_hold_attack_after_reaching_max,
                "width": brawl_stars_width,
                "height": brawl_stars_height,
                'find_closest_enemy': self.find_closest_enemy,
                'find_closest_teammate': self.find_closest_teammate,
                'find_closest_power_cube': self.find_closest_power_cube,
                'is_there_poison_gas': self.is_there_poison_gas,
                'is_path_blocked': self.is_path_blocked,
                'is_enemy_hittable': self.is_enemy_hittable,
                'time': time,
                'random': random,
                "persistent_data": self.persistent_data,
                'JOYSTICK_RADIUS': JOYSTICK_RADIUS,
                'rotate_movement': self.rotate_movement,
                'normalize_move': self.normalize_move,
                'width_ratio': self.window_controller.width_ratio,
                'height_ratio': self.window_controller.height_ratio
            }
        dynamic = self._dynamic_context
        dynamic['player_data'] = data['player'][0]
        dynamic['enemy_data'] = data['enemy']
        dynamic['teammate_data'] = data['teammate']
        dynamic['power_cube_data'] = data['power_cube']
        dynamic['brawler'] = brawler
        dynamic['walls'] = data['wall']
        dynamic['bushes'] = data['bush']
        dynamic['is_gadget_ready'] = self.is_gadget_ready
        dynamic['is_hypercharge_ready'] = self.is_hypercharge_ready
        dynamic['is_super_ready'] = self.is_super_ready
        dynamic['current_brawler'] = self.current_brawler
        dynamic['last_movement'] = self.last_movement
        dynamic['last_movement_change_time'] = self.last_movement_change_time
        dynamic['debug'] = self.verbose_debug
        if self._playstyle_globals is None or not self._playstyle_context_initialized:
            self.context.update(self._dynamic_context)
        movement = self.get_movement()
        movement_vector = self.movement_to_vector(movement)
        proposed_goal = self.persistent_data.get(
            "navigation_goal", "unknown"
        )
        strategic_now = time.monotonic()
        if movement_vector is not None and math.hypot(*movement_vector) >= 1:
            if proposed_goal != self._strategic_goal:
                self._strategic_commit_until = strategic_now
            self._strategic_heading = movement_vector
            self._strategic_goal = proposed_goal
        if (
            time.monotonic() < self._route_recovery_until
            and self.persistent_data.get("navigation_goal") not in {
                "poison", "power_cube", "fallback_power_cube",
            }
        ):
            movement_vector = self.get_route_recovery_movement(data)
        elif self.persistent_data.get("navigation_goal") in {
            "power_cube", "fallback_power_cube",
        }:
            self._route_recovery_until = 0.0
        self.navigation_decisions += 1
        if movement_vector is None or math.hypot(*movement_vector) < 1:
            self.navigation_idle_decisions += 1
            movement_vector = self.get_emergency_goal_movement(data)
        navigation_goal = self.persistent_data.get(
            "navigation_goal", "unknown"
        )
        if navigation_goal != self._last_navigation_goal:
            self.navigation_goal_switches += 1
            self._last_navigation_goal = navigation_goal
        self.navigation_goal_counts[navigation_goal] = (
            self.navigation_goal_counts.get(navigation_goal, 0) + 1
        )
        if movement_vector is None:
            self.window_controller.release_movement()
            self.last_movement = ''
            return None
        strategic_movement = movement_vector
        self._last_strategic_movement = strategic_movement
        movement_vector = self.navigation_correction(
            movement_vector, data['player'][0], data['wall']
        )
        planned_movement = self.clamp_movement(movement_vector)
        if (
            not self._wall_navigation_ready
            and (
                self._wall_navigation_initialized
                or self._wall_navigation_grace_started_at <= 0.0
                or time.monotonic() - self._wall_navigation_grace_started_at
                    > self.wall_navigation_startup_grace
            )
        ):
            self.last_movement = (0.0, 0.0)
            self.last_movement_change_time = time.time()
            return self.last_movement
        movement = planned_movement
        current_time = time.time()
        if self.last_movement and self.movement_is_similar(movement, self.last_movement):
            movement = self.last_movement
        elif movement != self.last_movement:
            if current_time - self.last_movement_change_time >= self.minimum_movement_delay:
                self.last_movement = movement
                self.last_movement_change_time = current_time
            else:
                movement = self.last_movement

        # Do not let timing hysteresis keep a formerly clear direction after
        # A* has already seen an obstacle farther ahead. Waiting until the
        # short collision guard reached the wall caused late, alternating
        # left/right corrections. A valid planned detour takes precedence as
        # soon as the stale direction conflicts with the planning horizon.
        commitment_distance = self._navigation_goal_distance(
            math.hypot(*strategic_movement)
        )
        planned_validation_distance = min(
            commitment_distance,
            max(
                self.TILE_SIZE * 0.9 * self.window_controller.scale_factor,
                math.hypot(*movement_vector),
            ),
        )
        stale_vector = self.movement_to_vector(movement) or (0.0, 0.0)
        stale_length = math.hypot(*stale_vector)
        planned_length = math.hypot(*planned_movement)
        strategic_length_for_stale = math.hypot(*strategic_movement)
        stale_alignment = (
            stale_vector[0] * strategic_movement[0]
            + stale_vector[1] * strategic_movement[1]
        ) / max(1.0, stale_length * strategic_length_for_stale)
        planned_alignment = (
            planned_movement[0] * strategic_movement[0]
            + planned_movement[1] * strategic_movement[1]
        ) / max(1.0, planned_length * strategic_length_for_stale)
        stale_wrong_way = (
            stale_alignment < 0.0 and planned_alignment >= 0.0
        )
        if (
            movement != planned_movement
            and (
                stale_wrong_way
                or self.is_path_blocked(
                    data['player'][0], movement, data['wall'],
                    distance=commitment_distance,
                )
            )
            and not self.is_path_blocked(
                data['player'][0], planned_movement, data['wall'],
                distance=planned_validation_distance,
            )
        ):
            self.stale_movement_overrides += 1
            movement = planned_movement
            self.last_movement = movement
            self.last_movement_change_time = current_time

        # Hysteresis is never allowed to restore a stale direction through an
        # obstacle after navigation has selected a safe detour.
        safety_distance = (
            self.TILE_SIZE * 0.9 * self.window_controller.scale_factor
        )
        if self.is_path_blocked(
            data['player'][0], movement, data['wall'],
            distance=safety_distance,
        ):
            self.collision_corrections += 1
            refresh_now = time.monotonic()
            if (
                refresh_now - self._last_obstacle_refresh_request
                >= self.wall_obstacle_refresh_interval
            ):
                self.wall_scene_gate.request_refresh()
                self._last_obstacle_refresh_request = refresh_now
            if not self.is_path_blocked(
                data['player'][0], planned_movement, data['wall'],
                distance=safety_distance,
            ):
                movement = planned_movement
            else:
                side = self._obstacle_escape_side
                rescue_offsets = (
                    side * math.pi / 8,
                    -side * math.pi / 8,
                    side * 3 * math.pi / 8,
                    -side * 3 * math.pi / 8,
                    side * math.pi / 2,
                    -side * math.pi / 2,
                    side * math.pi / 4,
                    -side * math.pi / 4,
                    side * 3 * math.pi / 4,
                    -side * 3 * math.pi / 4,
                    math.pi,
                )
                rescue_candidates = [
                    self.rotate_movement(planned_movement, offset)
                    for offset in rescue_offsets
                ]
                planned_length = math.hypot(*planned_movement)

                def rescue_score(candidate):
                    candidate_length = math.hypot(*candidate)
                    alignment = (
                        (
                            candidate[0] * planned_movement[0]
                            + candidate[1] * planned_movement[1]
                        ) / (candidate_length * planned_length)
                        if candidate_length >= 1 and planned_length >= 1
                        else -1.0
                    )
                    return self._movement_poison_risk(candidate) * 3.0 - alignment

                rescue_candidates.sort(key=rescue_score)
                movement = (0.0, 0.0)
                rescue_horizons = (
                    safety_distance,
                    self.TILE_SIZE * 0.45
                        * self.window_controller.scale_factor,
                    self.TILE_SIZE * 0.20
                        * self.window_controller.scale_factor,
                )
                for horizon_index, rescue_horizon in enumerate(
                    rescue_horizons
                ):
                    open_candidates = [
                        candidate for candidate in rescue_candidates
                        if not self.is_path_blocked(
                            data['player'][0], candidate, data['wall'],
                            distance=rescue_horizon,
                        )
                    ]
                    forward_candidates = []
                    for candidate in open_candidates:
                        candidate_length = math.hypot(*candidate)
                        alignment = (
                            candidate[0] * strategic_movement[0]
                            + candidate[1] * strategic_movement[1]
                        ) / max(
                            1.0, candidate_length
                            * math.hypot(*strategic_movement)
                        )
                        if alignment >= -0.05:
                            forward_candidates.append(candidate)
                    usable_candidates = forward_candidates or open_candidates
                    movement = (
                        usable_candidates[0]
                        if usable_candidates else (0.0, 0.0)
                    )
                    if math.hypot(*movement) >= 1:
                        if horizon_index:
                            self.progressive_escape_moves += 1
                        break
                if math.hypot(*movement) < 1:
                    self.collision_stops += 1
                else:
                    move_length = math.hypot(*movement)
                    alignment = (
                        movement[0] * planned_movement[0]
                        + movement[1] * planned_movement[1]
                    ) / max(1.0, move_length * planned_length)
                    if alignment < 0.0:
                        # A necessary retreat must last long enough to create
                        # clearance. A one-frame reverse followed by immediate
                        # forward planning caused the visible back-forward
                        # twitch reported in real matches.
                        self._detour_heading = movement
                        strategic_escape_length = math.hypot(
                            *strategic_movement
                        )
                        self._detour_goal_heading = (
                            strategic_movement[0] / strategic_escape_length,
                            strategic_movement[1] / strategic_escape_length,
                        ) if strategic_escape_length >= 1 else None
                        self._detour_expires_at = time.monotonic() + 0.55
                self.path_planner.reset()
                self.fine_path_planner.reset()
            self.last_movement = movement
            self.last_movement_change_time = current_time

        # Final strategic invariant: a clear and storm-safe objective may not
        # leave the joystick pointing into the opposite half-plane. This
        # catches stale hysteresis/detour state after every lower-level guard,
        # while still permitting a genuine opposite-side wall detour.
        strategic_length = math.hypot(*strategic_movement)
        output_length = math.hypot(*movement)
        if strategic_length >= 1 and output_length >= 1:
            strategic_cosine = (
                strategic_movement[0] * movement[0]
                + strategic_movement[1] * movement[1]
            ) / (strategic_length * output_length)
            forward_probe_distance = min(
                commitment_distance,
                self.TILE_SIZE * 0.52
                    * (self.window_controller.scale_factor or 1.0),
            )
            if (
                strategic_cosine < 0.0
                and self._movement_poison_risk(strategic_movement) < 0.08
                and not self.is_path_blocked(
                    data['player'][0], strategic_movement, data['wall'],
                    distance=forward_probe_distance,
                )
            ):
                movement = self.clamp_movement(strategic_movement)
                self.last_movement = movement
                self.last_movement_change_time = current_time
                self._detour_heading = None
                self._detour_goal_heading = None
                self._detour_expires_at = 0.0
                self.opposite_goal_overrides += 1
                self.short_forward_overrides += 1

        # Absolute output invariant: no playstyle, poison override, stale
        # hysteresis, detour commitment, or future strategy branch may send a
        # vector through currently known terrain. Keep this after every other
        # movement rewrite so the controller only receives a verified vector.
        final_safety_distance = (
            self.TILE_SIZE * 0.9 * self.window_controller.scale_factor
        )
        if self.is_path_blocked(
            data['player'][0], movement, data['wall'],
            distance=final_safety_distance,
        ):
            # Preserve useful motion: try deterministic collision-free
            # alternatives ordered by poison safety and goal alignment before
            # accepting a full stop.
            side = self._obstacle_escape_side
            final_offsets = (
                side * math.pi / 8, -side * math.pi / 8,
                side * math.pi / 4, -side * math.pi / 4,
                side * math.pi / 2, -side * math.pi / 2,
                side * 3 * math.pi / 4, -side * 3 * math.pi / 4,
                math.pi,
            )
            alternatives = [
                self.rotate_movement(movement, offset)
                for offset in final_offsets
            ]
            desired_length = math.hypot(*movement)

            def final_alternative_score(candidate):
                candidate_length = math.hypot(*candidate)
                alignment = (
                    (candidate[0] * movement[0] + candidate[1] * movement[1])
                    / (candidate_length * desired_length)
                    if candidate_length >= 1 and desired_length >= 1
                    else -1.0
                )
                return self._movement_poison_risk(candidate) * 3.0 - alignment

            alternatives.sort(key=final_alternative_score)
            movement = (0.0, 0.0)
            final_horizons = (
                final_safety_distance,
                self.TILE_SIZE * 0.45 * self.window_controller.scale_factor,
                self.TILE_SIZE * 0.20 * self.window_controller.scale_factor,
            )
            for horizon_index, horizon in enumerate(final_horizons):
                open_candidates = [
                    candidate for candidate in alternatives
                    if not self.is_path_blocked(
                        data['player'][0], candidate, data['wall'],
                        distance=horizon,
                    )
                ]
                forward_candidates = []
                for candidate in open_candidates:
                    candidate_length = math.hypot(*candidate)
                    alignment = (
                        candidate[0] * strategic_movement[0]
                        + candidate[1] * strategic_movement[1]
                    ) / max(1.0, candidate_length * strategic_length)
                    if alignment >= -0.05:
                        forward_candidates.append(candidate)
                usable_candidates = forward_candidates or open_candidates
                movement = (
                    usable_candidates[0]
                    if usable_candidates else (0.0, 0.0)
                )
                if math.hypot(*movement) >= 1:
                    if horizon_index:
                        self.progressive_escape_moves += 1
                    break
            self.last_movement = movement
            self.last_movement_change_time = current_time
            if math.hypot(*movement) < 1:
                self.collision_stops += 1
            else:
                self._detour_heading = movement
                self._detour_goal_heading = (
                    strategic_movement[0] / strategic_length,
                    strategic_movement[1] / strategic_length,
                ) if strategic_length >= 1 else None
                self._detour_expires_at = (
                    time.monotonic() + self.detour_commit_seconds
                )
            self.path_planner.reset()
            self.fine_path_planner.reset()
            refresh_now = time.monotonic()
            if (
                refresh_now - self._last_obstacle_refresh_request
                >= self.wall_obstacle_refresh_interval
            ):
                self.wall_scene_gate.request_refresh()
                self._last_obstacle_refresh_request = refresh_now
        self._last_output_movement = movement
        return movement

    def _ability_crop(self, frame, name, area):
        wr, hr = self.window_controller.width_ratio, self.window_controller.height_ratio
        scale = (wr, hr)
        if scale != self._ability_crop_scale:
            self._ability_crop_cache.clear()
            self._ability_crop_scale = scale
        coordinates = self._ability_crop_cache.get(name)
        if coordinates is None:
            coordinates = tuple(
                int(value * (wr if index % 2 == 0 else hr))
                for index, value in enumerate(area)
            )
            self._ability_crop_cache[name] = coordinates
        x1, y1, x2, y2 = coordinates
        return frame[y1:y2, x1:x2]

    def _ability_ready(self, frame, name, area, low_hsv, high_hsv, minimum):
        screenshot = self._ability_crop(frame, name, area)
        scale = self.ability_detection_scale
        source_shape = screenshot.shape
        buffers = self._ability_processing_buffers.get(name)
        if buffers is not None and buffers[0] != source_shape:
            buffers = None
        if scale < 1.0 and screenshot.size:
            if buffers is None or buffers[1] is None:
                screenshot = cv2.resize(
                    screenshot, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            else:
                resized = buffers[1]
                screenshot = cv2.resize(
                    screenshot, (resized.shape[1], resized.shape[0]),
                    dst=resized, interpolation=cv2.INTER_AREA,
                )
            minimum *= scale * scale
        shape = screenshot.shape
        if buffers is None or buffers[2].shape != shape:
            resized_buffer = screenshot if scale < 1.0 else None
            hsv_buffer = np.empty(shape, dtype=np.uint8)
            mask_buffer = np.empty(shape[:2], dtype=np.uint8)
            buffers = (source_shape, resized_buffer, hsv_buffer, mask_buffer)
            self._ability_processing_buffers[name] = buffers
        pixels = count_hsv_pixels(
            screenshot, low_hsv, high_hsv, self.window_controller,
            hsv_buffer=buffers[2], mask_buffer=buffers[3]
        )
        if self.verbose_debug:
            print(f"{name} pixels:", pixels, "(if > ", minimum, f" then {name} is ready)")
            try:
                cv2.imwrite(
                    f"debug_frames/{name}_debug_{pixels}_{int(time.time())}.png",
                    cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR),
                )
            except Exception:
                pass
        return pixels > minimum

    def check_if_hypercharge_ready(self, frame):
        return self._ability_ready(
            frame, "hypercharge", hypercharge_crop_area,
            (137, 158, 159), (179, 255, 255), self.hypercharge_pixels_minimum,
        )

    def check_if_gadget_ready(self, frame):
        return self._ability_ready(
            frame, "gadget", gadget_crop_area,
            (57, 219, 165), (62, 255, 255), self.gadget_pixels_minimum,
        )

    def check_if_super_ready(self, frame):
        return self._ability_ready(
            frame, "super", super_crop_area,
            (17, 170, 200), (27, 255, 255), self.super_pixels_minimum,
        )

    def get_centered_wall_crop(self, frame, player_data=None):
        frame_height, frame_width = frame.shape[:2]
        # Low-end devices often capture at 960x540 or 1280x720. A fixed
        # 640px square is taller than a 540px frame; the old clamp then got a
        # negative upper bound and sliced only a thin strip from the bottom.
        # That made terrain effectively invisible to both A* and shot LOS.
        crop_size = min(
            self.centered_wall_crop_size, frame_width, frame_height
        )
        if crop_size <= 0:
            return frame, 0, 0

        if player_data:
            center_x, center_y = self.get_player_position(player_data[0])
        else:
            center_x, center_y = frame_width / 2, frame_height / 2

        crop_x1 = int(clamp(
            round(center_x - crop_size / 2), 0, frame_width - crop_size
        ))
        crop_y1 = int(clamp(
            round(center_y - crop_size / 2), 0, frame_height - crop_size
        ))
        crop_x2 = crop_x1 + crop_size
        crop_y2 = crop_y1 + crop_size

        return frame[crop_y1:crop_y2, crop_x1:crop_x2], crop_x1, crop_y1

    @staticmethod
    def offset_tile_data(tile_data, offset_x, offset_y):
        if not offset_x and not offset_y:
            return tile_data

        offset_data = {}
        for class_name, boxes in tile_data.items():
            offset_data[class_name] = [
                [box[0] + offset_x, box[1] + offset_y, box[2] + offset_x, box[3] + offset_y]
                for box in boxes
            ]
        return offset_data

    def get_tile_data(self, frame, player_data=None):
        if self.centered_wall_detection:
            if self.Detect_centered_tile_detector is None:
                self.preload_wall_model()
                return None
            crop, offset_x, offset_y = self.get_centered_wall_crop(frame, player_data)
            tile_data = self.Detect_centered_tile_detector.detect_objects(
                crop,
                conf_tresh=self.wall_detection_confidence
            )
            return self.offset_tile_data(tile_data, offset_x, offset_y)

        if self.Detect_tile_detector is None:
            self.preload_wall_model()
            return None
        tile_data = self.Detect_tile_detector.detect_objects(frame, conf_tresh=self.wall_detection_confidence)
        return tile_data

    def preload_wall_model(self):
        """Load the selected wall detector off the gameplay thread."""
        target_attribute = (
            "Detect_centered_tile_detector"
            if self.centered_wall_detection else "Detect_tile_detector"
        )
        if getattr(self, target_attribute) is not None or self._wall_model_loading:
            return
        with self._wall_model_lock:
            if getattr(self, target_attribute) is not None or self._wall_model_loading:
                return
            self._wall_model_loading = True

        model_path = (
            self.close_tile_detector_model_path
            if self.centered_wall_detection else self.tile_detector_model_path
        )
        load_started = time.monotonic()

        def load_detector():
            try:
                detector = Detect(
                    model_path,
                    classes=self.tile_detector_model_classes,
                )
                # Force provider kernels and reusable preprocessing buffers to
                # initialize during matchmaking, not on the first match frame.
                warm_height, warm_width = detector.input_size
                warm_frame = np.zeros(
                    (warm_height, warm_width, 3), dtype=np.uint8
                )
                detector.detect_objects(
                    warm_frame, conf_tresh=self.wall_detection_confidence
                )
                with self._wall_model_lock:
                    setattr(self, target_attribute, detector)
                print(
                    f"Wall navigation model ready in "
                    f"{time.monotonic() - load_started:.1f}s."
                )
            except Exception as error:
                print(f"Wall model preload failed: {error}")
            finally:
                with self._wall_model_lock:
                    self._wall_model_loading = False

        threading.Thread(
            target=load_detector,
            daemon=True,
            name="pyla-wall-model-loader",
        ).start()

    def stabilize_wall_boxes(self, walls):
        """Smooth wall detections and survive one missed inference."""
        observed_at = time.monotonic()
        observation_interval = (
            observed_at - self._wall_motion_observed_at
            if self._wall_motion_observed_at > 0.0 else 0.0
        )
        previous_tracks = self._wall_tracks
        if not previous_tracks:
            self._wall_tracks = [(list(wall[:4]), 0) for wall in walls]
            self._wall_motion_observed_at = observed_at
            return [track[0] for track in self._wall_tracks]

        unmatched_previous = set(range(len(previous_tracks)))
        updated_tracks = []
        matched_motion = []
        scale_factor = self.window_controller.scale_factor
        base_match_distance = self.TILE_SIZE * 1.5 * scale_factor

        for wall in walls:
            new_box = list(map(float, wall[:4]))
            new_center = (
                (new_box[0] + new_box[2]) * 0.5,
                (new_box[1] + new_box[3]) * 0.5,
            )
            new_width = max(1.0, new_box[2] - new_box[0])
            new_height = max(1.0, new_box[3] - new_box[1])
            best_index = None
            best_score = float("inf")
            for index in unmatched_previous:
                old_box, _ = previous_tracks[index]
                old_center = (
                    (old_box[0] + old_box[2]) * 0.5,
                    (old_box[1] + old_box[3]) * 0.5,
                )
                old_width = max(1.0, old_box[2] - old_box[0])
                old_height = max(1.0, old_box[3] - old_box[1])
                match_distance = max(
                    base_match_distance,
                    0.35 * max(new_width, new_height, old_width, old_height),
                )
                delta_x = new_center[0] - old_center[0]
                delta_y = new_center[1] - old_center[1]
                # Most old boxes are nowhere near this detection. Reject them
                # with two cheap comparisons before multiplication/sqrt.
                if (
                    abs(delta_x) > match_distance
                    or abs(delta_y) > match_distance
                ):
                    continue
                center_distance_sq = delta_x * delta_x + delta_y * delta_y
                if center_distance_sq > match_distance * match_distance:
                    continue
                center_distance = math.sqrt(center_distance_sq)
                size_error = (
                    abs(new_width - old_width) / max(new_width, old_width)
                    + abs(new_height - old_height) / max(new_height, old_height)
                )
                score = center_distance / match_distance + size_error * 0.35
                if score < best_score or (
                    score == best_score
                    and (best_index is None or index < best_index)
                ):
                    best_score = score
                    best_index = index

            if best_index is None:
                updated_tracks.append((new_box, 0))
                continue

            unmatched_previous.remove(best_index)
            old_box, _ = previous_tracks[best_index]
            # Prefer current geometry while suppressing a few pixels of model
            # jitter that would otherwise invalidate and flip an A* route.
            smoothed = [
                old_value * 0.25 + new_value * 0.75
                for old_value, new_value in zip(old_box, new_box)
            ]
            old_center_x = (old_box[0] + old_box[2]) * 0.5
            old_center_y = (old_box[1] + old_box[3]) * 0.5
            matched_motion.append((
                (smoothed[0] + smoothed[2]) * 0.5 - old_center_x,
                (smoothed[1] + smoothed[3]) * 0.5 - old_center_y,
            ))
            updated_tracks.append((smoothed, 0))

        # A completely empty inference is more likely a transient model miss
        # than every nearby wall vanishing simultaneously. Bridge two such
        # refreshes; when some current geometry exists, retain unmatched old
        # boxes for only one refresh to avoid long-lived ghost obstacles.
        maximum_misses = 2 if not walls else 1
        for index in sorted(unmatched_previous):
            old_box, misses = previous_tracks[index]
            if misses < maximum_misses:
                updated_tracks.append((old_box, misses + 1))

        updated_tracks.sort(key=lambda track: (
            round(track[0][1], 1), round(track[0][0], 1),
            round(track[0][3], 1), round(track[0][2], 1),
        ))
        self._wall_tracks = updated_tracks
        if len(matched_motion) >= 2:
            # Median motion rejects a wall that was associated with the wrong
            # repeated tile while retaining the shared camera translation.
            motion_x = sorted(delta[0] for delta in matched_motion)
            motion_y = sorted(delta[1] for delta in matched_motion)
            middle = len(matched_motion) // 2
            if len(matched_motion) % 2:
                camera_dx, camera_dy = motion_x[middle], motion_y[middle]
            else:
                camera_dx = (motion_x[middle - 1] + motion_x[middle]) * 0.5
                camera_dy = (motion_y[middle - 1] + motion_y[middle]) * 0.5
            consensus_radius = max(4.0, self.TILE_SIZE * 0.30 * scale_factor)
            consensus = [
                delta for delta in matched_motion
                if math.hypot(
                    delta[0] - camera_dx, delta[1] - camera_dy
                ) <= consensus_radius
            ]
            minimum_consensus = max(
                2, int(math.ceil(len(matched_motion) * 0.60))
            )
            if len(consensus) >= minimum_consensus:
                # Average only the median-consistent vectors. Repeated map
                # tiles can otherwise be cross-associated and move every
                # remembered goal in the wrong direction.
                camera_dx = sum(delta[0] for delta in consensus) / len(consensus)
                camera_dy = sum(delta[1] for delta in consensus) / len(consensus)
                if observation_interval > 0.05:
                    measured_velocity = (
                        camera_dx / observation_interval,
                        camera_dy / observation_interval,
                    )
                    old_velocity = self._wall_motion_velocity
                    self._wall_motion_velocity = (
                        old_velocity[0] * 0.35
                            + measured_velocity[0] * 0.65,
                        old_velocity[1] * 0.35
                            + measured_velocity[1] * 0.65,
                    )
                self._shift_screen_space_memories(camera_dx, camera_dy)
            else:
                self._wall_motion_velocity = (
                    self._wall_motion_velocity[0] * 0.45,
                    self._wall_motion_velocity[1] * 0.45,
                )
        else:
            self._wall_motion_velocity = (
                self._wall_motion_velocity[0] * 0.45,
                self._wall_motion_velocity[1] * 0.45,
            )
        self._wall_motion_observed_at = observed_at
        self._projected_walls_key = None
        return [track[0] for track in updated_tracks]

    def get_projected_walls(self):
        """Project cached screen-space walls between detector refreshes."""
        if not self.last_walls_data or self._wall_motion_observed_at <= 0.0:
            return self.last_walls_data
        elapsed = min(0.55, max(
            0.0, time.monotonic() - self._wall_motion_observed_at
        ))
        # Quantization avoids allocating a new box list on every entity tick.
        elapsed_bucket = round(elapsed / 0.06) * 0.06
        velocity_x, velocity_y = self._wall_motion_velocity
        displacement_x = velocity_x * elapsed_bucket
        displacement_y = velocity_y * elapsed_bucket
        maximum_shift = self.TILE_SIZE * 0.65 * (
            self.window_controller.scale_factor or 1.0
        )
        displacement_length = math.hypot(displacement_x, displacement_y)
        if displacement_length > maximum_shift:
            factor = maximum_shift / displacement_length
            displacement_x *= factor
            displacement_y *= factor
        if math.hypot(displacement_x, displacement_y) < 1.0:
            return self.last_walls_data
        key = (
            id(self.last_walls_data), elapsed_bucket,
            round(displacement_x, 1), round(displacement_y, 1),
        )
        if key == self._projected_walls_key:
            self.wall_projection_uses += 1
            return self._projected_walls
        self._projected_walls_key = key
        self._projected_walls = [
            [
                wall[0] + displacement_x, wall[1] + displacement_y,
                wall[2] + displacement_x, wall[3] + displacement_y,
            ]
            for wall in self.last_walls_data
        ]
        self.wall_projection_uses += 1
        return self._projected_walls

    def process_tile_data(self, tile_data):
        walls = []
        bushes = []
        for class_name, boxes in tile_data.items():
            if 'bush' not in class_name:
                walls.extend(boxes)
            else:
                bushes.extend(boxes)
        walls = self.merge_collinear_walls(walls)
        return self.stabilize_wall_boxes(walls), bushes

    def merge_collinear_walls(self, walls):
        """Merge straight tile runs with orientation-filtered sweep passes."""
        if len(walls) < 2:
            return walls
        gap_limit = self.TILE_SIZE * 0.3 * self.window_controller.scale_factor
        boxes = [list(map(float, wall[:4])) for wall in walls]
        dimensions = [
            (
                max(1.0, box[2] - box[0]),
                max(1.0, box[3] - box[1]),
            )
            for box in boxes
        ]

        def connected_components(indices, horizontal):
            parent = list(range(len(boxes)))
            ordered_indices = sorted(
                indices,
                key=lambda index: boxes[index][0 if horizontal else 1],
            )

            def find(index):
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                return index

            def union(first, second):
                first_root, second_root = find(first), find(second)
                if first_root != second_root:
                    parent[second_root] = first_root

            for offset, first_index in enumerate(ordered_indices):
                first = boxes[first_index]
                first_width, first_height = dimensions[first_index]
                primary_end = first[2 if horizontal else 3]
                for second_offset in range(offset + 1, len(ordered_indices)):
                    second_index = ordered_indices[second_offset]
                    second = boxes[second_index]
                    if second[0 if horizontal else 1] > primary_end + gap_limit:
                        break
                    second_width, second_height = dimensions[second_index]
                    if horizontal:
                        overlap = max(
                            0.0, min(first[3], second[3])
                            - max(first[1], second[1])
                        )
                        gap = max(first[0], second[0]) - min(first[2], second[2])
                        aligned = (
                            first_width >= first_height * 0.8
                            and second_width >= second_height * 0.8
                            and overlap >= min(first_height, second_height) * 0.72
                        )
                    else:
                        overlap = max(
                            0.0, min(first[2], second[2])
                            - max(first[0], second[0])
                        )
                        gap = max(first[1], second[1]) - min(first[3], second[3])
                        aligned = (
                            first_height >= first_width * 0.8
                            and second_height >= second_width * 0.8
                            and overlap >= min(first_width, second_width) * 0.72
                        )
                    if aligned and gap <= gap_limit:
                        union(first_index, second_index)

            groups = {}
            for index in ordered_indices:
                root = find(index)
                members = groups.get(root)
                if members is None:
                    members = []
                    groups[root] = members
                members.append(index)
            return list(groups.values())

        all_indices = range(len(boxes))
        # Avoid comparing impossible orientation pairs. Detector output often
        # contains dozens of tiles; the old implementation ran two full O(n²)
        # passes even though most pairs could never merge.
        horizontal_indices = [
            index for index in all_indices
            if dimensions[index][0] >= dimensions[index][1] * 0.8
        ]
        horizontal_groups = connected_components(
            horizontal_indices, horizontal=True
        )
        output = []
        remaining = set(range(len(boxes)))
        for group in horizontal_groups:
            if len(group) == 1:
                continue
            members = [boxes[index] for index in group]
            output.append([
                min(box[0] for box in members), min(box[1] for box in members),
                max(box[2] for box in members), max(box[3] for box in members),
            ])
            remaining.difference_update(group)

        vertical_indices = [
            index for index in remaining
            if dimensions[index][1] >= dimensions[index][0] * 0.8
        ]
        vertically_handled = set()
        for group in connected_components(vertical_indices, horizontal=False):
            members = [boxes[index] for index in group]
            output.append([
                min(box[0] for box in members), min(box[1] for box in members),
                max(box[2] for box in members), max(box[3] for box in members),
            ])
            vertically_handled.update(group)

        # Extremely elongated/noisy detections which match neither orientation
        # remain valid obstacles and must not disappear from navigation.
        for index in sorted(remaining - vertically_handled):
            output.append(boxes[index])

        return [[round(value) for value in wall] for wall in output]

    def get_movement(self):
        if self._playstyle_globals is not None:
            if not self._playstyle_context_initialized:
                self._playstyle_globals.update(self.context)
                # Tick variables persist in the optimized namespace. Preserve the
                # callable API so an assignment such as `attack = False` cannot
                # disable that function on every subsequent frame.
                self._playstyle_api_callables = {
                    name: value for name, value in self.context.items()
                    if callable(value)
                }
                self._playstyle_context_initialized = True
            else:
                self._playstyle_globals.update(self._dynamic_context)
                self._playstyle_globals.update(self._playstyle_api_callables)
            self._playstyle_globals.pop('movement', None)
            try:
                exec(self.pyla_code, self._playstyle_globals)
            except Exception as error:
                # Avoid spending the whole frame formatting the same traceback
                # when a custom playstyle has a persistent error.
                now = time.monotonic()
                error_key = (type(error), str(error))
                if error_key != self._last_playstyle_error or now - self._last_playstyle_error_at >= 5.0:
                    print("Error executing optimized .pyla code")
                    traceback.print_exc()
                    self._last_playstyle_error = error_key
                    self._last_playstyle_error_at = now
                return None
            self._last_playstyle_error = None
            return self._playstyle_globals.get('movement')
        movement, updated_globals = interpret_pyla_code(self.pyla_code, self.context)
        return movement

    def publish_debug_view(self, frame, data, state, movement=None):
        if not hasattr(self.window_controller, "debug_view"):
            return
        if not self.window_controller.debug_view.enabled:
            return

        self.frame = frame
        advanced_visuals = bool(getattr(self.window_controller.debug_view, "advanced_visuals", False))
        debug_data = {
            "state": state,
            "player": [],
            "enemy": [],
            "teammate": [],
            "power_cube": [],
            "wall": [],
            "attack_range": 0,
            "super_range": 0,
            "poison_gas": {},
            "movement": None,
            "joystick": [self.window_controller.movement_joystick_x, self.window_controller.movement_joystick_y],
            "advanced_visuals": advanced_visuals,
            "joystick_radius": int(JOYSTICK_RADIUS * (self.window_controller.scale_factor or 1)),
            "joystick_directions": [],
            "enemy_los_lines": [],
            "teammate_los_lines": [],
            "player_hit_circle": None,
        }

        if data:
            for key in ["player", "enemy", "teammate", "power_cube", "wall"]:
                debug_data[key] = [[int(v) for v in box[:4]] for box in (data.get(key) or []) if len(box) >= 4]
            try:
                _, attack_range, super_range = self.get_brawler_range(self.current_brawler)
                debug_data["attack_range"] = int(attack_range)
                debug_data["super_range"] = int(super_range)
            except Exception:
                pass
            if debug_data["player"]:
                try:
                    debug_data["poison_gas"] = self.is_there_poison_gas(debug_data["player"][0])
                except Exception:
                    pass

        if movement is not None:
            debug_data["movement"] = [float(movement[0]), float(movement[1])]

        self.window_controller.debug_view.publish(frame, debug_data)

    def main(self, frame, brawler, main):
        current_time = time.time()
        state = main.get_latest_state()
        data = self.get_main_data(frame)
        # Navigation cannot use walls without a player position. Defer that
        # network until a usable entity frame arrives; leave its timer due.
        if data.get("player") and current_time - self.time_since_walls_checked > self.walls_treshold:
            refresh_walls, wall_sample = self.wall_scene_gate.should_refresh(frame, current_time)
            if refresh_walls:
                self.wall_inference_count += 1
                tile_data = self.get_tile_data(frame, data.get("player"))
                if tile_data is None:
                    # The model is still warming during matchmaking. Keep the
                    # last geometry only briefly. Screen-space wall boxes
                    # become unsafe after camera movement, so stop navigation
                    # rather than steering through a stale obstacle map.
                    self.wall_inference_count -= 1
                    data['wall'] = self.get_projected_walls()
                    data['bush'] = self.last_bushes_data
                    if (
                        current_time - self.time_since_walls_checked
                        > self.wall_navigation_maximum_age
                    ):
                        if self._wall_navigation_ready:
                            self.stale_wall_stops += 1
                        self._wall_navigation_ready = False
                else:
                    walls, bushes = self.process_tile_data(tile_data)
                    self.time_since_walls_checked = current_time
                    self.last_walls_data = walls
                    data['wall'] = walls
                    self.last_bushes_data = bushes
                    data['bush'] = bushes
                    self._wall_navigation_ready = True
                    self._wall_navigation_initialized = True
                    self._wall_navigation_grace_started_at = 0.0
                    self.wall_scene_gate.accept(wall_sample, current_time)
            else:
                self.wall_cache_count += 1
                data['wall'] = self.get_projected_walls()
                data['bush'] = self.last_bushes_data
        else:
            data['wall'] = self.get_projected_walls()
            data['bush'] = self.last_bushes_data

        data = self.validate_game_data(data)
        self.track_no_detections(data)
        if data:
            self.time_since_player_last_found = time.time()
            if state != "match":
                data = None

        if not data:
            if current_time - self.time_since_player_last_found > 1.0:
                self.window_controller.release_movement()
            if current_time - self.time_since_last_proceeding > self.no_detection_proceed_delay:
                current_state = get_state(frame, previous_state=state)
                confirmed_state = main.observe_state(current_state, current_time)
                if confirmed_state is None:
                    pass
                elif confirmed_state != "match":
                    main.handle_detected_state(confirmed_state)
                    state = confirmed_state
                    self.time_since_last_proceeding = current_time
                else:
                    print("haven't detected the player in a while proceeding")
                    self.window_controller.press("proceed")
                    self.time_since_last_proceeding = time.time()
            self.publish_debug_view(frame, data, state)
            return
        self.time_since_last_proceeding = time.time()
        if (not self.is_hypercharge_ready
                and self.ability_can_recheck("hypercharge")
                and current_time - self.time_since_hypercharge_checked > self.hypercharge_treshold):
            self.is_hypercharge_ready = self.check_if_hypercharge_ready(frame)
            self.time_since_hypercharge_checked = current_time
        if (not self.is_gadget_ready
                and self.ability_can_recheck("gadget")
                and current_time - self.time_since_gadget_checked > self.gadget_treshold):
            self.is_gadget_ready = self.check_if_gadget_ready(frame)
            self.time_since_gadget_checked = current_time
        if (not self.is_super_ready
                and self.ability_can_recheck("super")
                and current_time - self.time_since_super_checked > self.super_treshold):
            self.is_super_ready = self.check_if_super_ready(frame)
            self.time_since_super_checked = current_time
        self.frame = frame
        movement = self.loop(brawler, data, current_time)
        self.publish_debug_view(frame, data, state, movement)
        if movement is not None:
            self.do_movement(movement)
