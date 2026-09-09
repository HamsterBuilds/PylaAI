import math
import random
import time
import ast
import traceback
import cv2
import numpy as np
import os
import weakref

from detect import Detect
from state_finder import get_state
from perception import DetectionStabilizer, SceneChangeGate
from pathfinding import LocalPathPlanner
from utils import load_toml_as_dict, count_hsv_pixels, load_brawlers_info, interpret_pyla_code, \
    count_mask_pixels, JOYSTICK_RADIUS, clamp, config_bool, is_safe_ast, SAFE_GLOBALS


brawl_stars_width, brawl_stars_height = 1920, 1080
super_crop_area = load_toml_as_dict("./cfg/lobby_config.toml")['pixel_counter_crop_area']['super']
gadget_crop_area = load_toml_as_dict("./cfg/lobby_config.toml")['pixel_counter_crop_area']['gadget']
hypercharge_crop_area = load_toml_as_dict("./cfg/lobby_config.toml")['pixel_counter_crop_area']['hypercharge']
POISON_LOW_HSV = np.array((30, 90, 221), dtype=np.uint8)
POISON_HIGH_HSV = np.array((57, 114, 235), dtype=np.uint8)
PLAYER_HIT_CIRCLE_RADIUS = 53

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
        self.keys_hold = []
        self.time_since_gadget_checked = time.time()
        self.is_gadget_ready = False
        self.time_since_hypercharge_checked = time.time()
        self.is_hypercharge_ready = False
        self.time_since_super_checked = time.time()
        self.is_super_ready = False
        self.window_controller = window_controller
        self.TILE_SIZE = bot_config.get("perceived_tile_size", 54)
        self.centered_wall_detection = config_bool(bot_config.get("centered_wall_detection"), False)
        self.centered_wall_crop_size = 640

        bot_config = load_toml_as_dict("cfg/bot_config.toml")
        time_config = load_toml_as_dict("cfg/time_tresholds.toml")
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
        self.entity_detection_confidence = bot_config["entity_detection_confidence"]
        self.seconds_to_hold_attack_after_reaching_max = load_toml_as_dict("cfg/bot_config.toml")["seconds_to_hold_attack_after_reaching_max"]
        self.minimum_attack_interval = max(
            0.0, float(bot_config.get("minimum_attack_interval", 0.08))
        )
        self.last_attack_at = 0.0
        self.persistent_data = {"time_since_holding_attack": None}
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
        self._dynamic_context = None
        self._playstyle_context_initialized = False
        self._playstyle_api_callables = None
        self._last_playstyle_error = None
        self._last_playstyle_error_at = 0.0
        self.frame = None
        self._ability_crop_cache = {}
        self._ability_crop_scale = None
        self._decision_cache = {}
        self._prepared_frame_ref = None
        self._prepared_main_data = None
        self._target_memory = {}
        self.target_switch_ratio = max(1.0, float(bot_config.get("target_switch_ratio", 1.18)))
        self.target_match_distance = max(10.0, float(bot_config.get("target_match_distance", 180.0)))
        self.detection_stabilizer = DetectionStabilizer(
            smoothing=bot_config.get("detection_smoothing", 0.68),
            max_tracks=bot_config.get("maximum_tracked_entities", 16),
            prediction=bot_config.get("detection_prediction", 0.25),
            velocity_smoothing=bot_config.get("velocity_smoothing", 0.55),
        )
        self.objective_position = None
        self.objective_expires_at = 0.0
        self.objective_memory_duration = max(
            0.5, float(bot_config.get("objective_memory_duration", 2.4))
        )
        self.objective_arrival_distance = max(
            0.25, float(bot_config.get("objective_arrival_tiles", 0.65))
        )
        self.last_teammate_position = None
        self.last_teammate_seen_at = 0.0
        self.teammate_memory_duration = max(
            0.5, float(bot_config.get("teammate_memory_duration", 3.0))
        )
        self.path_planner = LocalPathPlanner(
            area_size=self.centered_wall_crop_size,
            cell_size=bot_config.get("pathfinding_cell_size", 36),
            cache_seconds=bot_config.get("pathfinding_cache_seconds", 0.25),
        )
        self.poison_cache_interval = max(
            0.05, float(bot_config.get("poison_detection_interval", 0.2))
        )
        self._poison_cache = None
        self._poison_cache_at = 0.0
        self._poison_cache_player = None
        self.wall_inference_count = 0
        self.wall_cache_count = 0
        self.poison_compute_count = 0
        self.poison_cache_count = 0

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
        # Rate-limit only complete taps. Charge start and release are edge
        # events and must always reach the device immediately.
        if touch_up and touch_down:
            now = time.monotonic()
            if now - self.last_attack_at < self.minimum_attack_interval:
                return False
            self.last_attack_at = now
        self.window_controller.press("attack", touch_up=touch_up, touch_down=touch_down)
        return True

    def use_hypercharge(self):
        print("Using hypercharge")
        self.window_controller.press("hypercharge")
        self.time_since_hypercharge_checked = time.time()
        self.is_hypercharge_ready = False

    def use_gadget(self):
        print("Using gadget")
        self.window_controller.press("gadget")
        self.time_since_gadget_checked = time.time()
        self.is_gadget_ready = False

    def use_super(self):
        print("Using super")
        self.window_controller.press("super")
        self.time_since_super_checked = time.time()
        self.is_super_ready = False

    def remember_objective(self, position):
        if position is None or len(position) < 2:
            return
        self.objective_position = (float(position[0]), float(position[1]))
        self.objective_expires_at = time.time() + self.objective_memory_duration

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

    @staticmethod
    def get_random_movement():
        random_movement = random.randint(-75, 75), random.randint(-75, 75)
        return random_movement

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
        for p, q in (
            (-dx, x1 - left), (dx, right - x1),
            (-dy, y1 - top), (dy, bottom - y1),
        ):
            if abs(p) < 1e-9:
                if q < 0:
                    return False
                continue
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
    def walls_block_line_of_sight(p1, p2, walls):
        if not walls:
            return False

        min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
        min_y, max_y = min(p1[1], p2[1]), max(p1[1], p2[1])
        for wall in walls:
            x1, y1, x2, y2 = wall

            if max_x < x1 or min_x > x2 or max_y < y1 or min_y > y2:
                continue

            if Play.segment_intersects_rect(p1, p2, (x1, y1, x2, y2)):
                return True
        return False

    def get_player_hit_circle(self, player_box):
        radius = PLAYER_HIT_CIRCLE_RADIUS * (self.window_controller.scale_factor or 1)
        if player_box and len(player_box) >= 4:
            x1, y1, x2, y2 = player_box[:4]
            return ((x1 + x2) / 2, y2 - radius), radius

        return None, radius

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
            x1, y1, x2, y2 = wall[:4]
            wall_rect = (x1, y1, x2, y2)
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
                end_distance_sq = Play.point_rect_distance_sq(p2, wall_rect)
                if start_distance_sq <= radius_sq and end_distance_sq > start_distance_sq:
                    continue
                return True

        return False

    def is_enemy_hittable(self, player_pos, enemy_pos, walls, skill_type):
        if self.can_attack_through_walls(self.current_brawler, skill_type, self.brawlers_info):
            return True
        if self.walls_block_line_of_sight(player_pos, enemy_pos, walls):
            return False
        return True

    def estimate_navigation_cost(self, start, goal, walls):
        """Cheap wall-corner detour estimate used only for target ranking."""
        direct = self.get_distance(start, goal)
        if not walls:
            return direct
        radius = PLAYER_HIT_CIRCLE_RADIUS * (self.window_controller.scale_factor or 1)
        margin = radius + 6.0 * (self.window_controller.scale_factor or 1)
        extra = 0.0
        for wall in walls:
            if not self.segment_intersects_rect(start, goal, wall[:4]):
                continue
            x1, y1, x2, y2 = wall[:4]
            corners = (
                (x1 - margin, y1 - margin), (x2 + margin, y1 - margin),
                (x1 - margin, y2 + margin), (x2 + margin, y2 + margin),
            )
            corner_route = min(
                self.get_distance(start, corner) + self.get_distance(corner, goal)
                for corner in corners
            )
            extra += max(0.0, corner_route - direct)
        return direct + extra

    def find_closest_enemy(self, enemy_data, player_coords, walls, skill_type):
        cache_key = (
            "enemy", id(enemy_data), id(walls), tuple(player_coords),
            skill_type, self.current_brawler,
        )
        if cache_key in self._decision_cache:
            return self._decision_cache[cache_key]
        player_pos_x, player_pos_y = player_coords
        closest_hittable_distance = float('inf')
        closest_unhittable_distance = float('inf')
        closest_hittable = None
        closest_unhittable = None
        hittable_candidates = []
        unhittable_candidates = []
        previous_target = self._target_memory.get(skill_type)
        target_match_limit = self.target_match_distance * (
            self.window_controller.scale_factor or 1.0
        )
        for enemy in enemy_data:
            enemy_pos = self.get_entity_pos(enemy)
            distance = self.get_distance(enemy_pos, player_coords)
            # A farther target cannot beat an already hittable target, and
            # blocked targets are only a fallback when none is hittable.
            # Preserve first-in-order tie handling without another wall scan.
            can_match_previous = (
                previous_target is not None
                and self.get_distance(enemy_pos, previous_target) <= target_match_limit
            )
            if (
                closest_hittable is not None
                and distance >= closest_hittable_distance
                and not can_match_previous
            ):
                continue
            if self.is_enemy_hittable((player_pos_x, player_pos_y), enemy_pos, walls, skill_type):
                hittable_candidates.append((enemy_pos, distance))
                if distance < closest_hittable_distance:
                    closest_hittable_distance = distance
                    closest_hittable = [enemy_pos, distance]
            else:
                unhittable_candidates.append((enemy_pos, distance))
                if distance < closest_unhittable_distance:
                    closest_unhittable_distance = distance
                    closest_unhittable = [enemy_pos, distance]
        if closest_hittable:
            result = closest_hittable
            if previous_target is not None:
                matching = min(
                    hittable_candidates,
                    key=lambda item: self.get_distance(item[0], previous_target),
                    default=None,
                )
                if (
                    matching is not None
                    and self.get_distance(matching[0], previous_target) <= target_match_limit
                    and matching[1] <= closest_hittable_distance * self.target_switch_ratio
                ):
                    result = [matching[0], matching[1]]
            self._target_memory[skill_type] = result[0]
        elif closest_unhittable:
            fastest = min(
                unhittable_candidates,
                key=lambda item: self.estimate_navigation_cost(
                    player_coords, item[0], walls
                ),
            )
            # Keep pursuing the same blocked target when its route remains
            # competitive. Without this, tiny detector changes can alternate
            # between two boxes/enemies on opposite sides of a wall.
            if previous_target is not None:
                matching = min(
                    unhittable_candidates,
                    key=lambda item: self.get_distance(item[0], previous_target),
                    default=None,
                )
                if matching is not None and self.get_distance(
                    matching[0], previous_target
                ) <= target_match_limit:
                    fastest_cost = self.estimate_navigation_cost(
                        player_coords, fastest[0], walls
                    )
                    matching_cost = self.estimate_navigation_cost(
                        player_coords, matching[0], walls
                    )
                    if matching_cost <= fastest_cost * self.target_switch_ratio:
                        fastest = matching
            result = [fastest[0], fastest[1]]
            self._target_memory[skill_type] = result[0]
        else:
            result = (None, None)
            self._target_memory.pop(skill_type, None)
        self._decision_cache[cache_key] = result
        return result

    def find_closest_teammate(self, teammate_data, player_coords, walls):
        cache_key = ("teammate", id(teammate_data), tuple(player_coords))
        if cache_key in self._decision_cache:
            return self._decision_cache[cache_key]
        teammate_anchor = self.get_teammate_anchor(teammate_data)
        if teammate_anchor is not None:
            teammate_distance = self.get_distance(teammate_anchor, player_coords)
            self.last_teammate_position = teammate_anchor
            self.last_teammate_seen_at = time.time()
        elif (
            self.last_teammate_position is not None
            and time.time() - self.last_teammate_seen_at <= self.teammate_memory_duration
        ):
            teammate_anchor = self.last_teammate_position
            teammate_distance = self.get_distance(teammate_anchor, player_coords)
        else:
            teammate_distance = float('inf')
        result = (teammate_anchor, teammate_distance)
        self._decision_cache[cache_key] = result
        return result

    def get_teammate_anchor(self, teammate_data):
        if not teammate_data:
            return None
        positions = [self.get_entity_pos(teammate) for teammate in teammate_data]
        count = len(positions)
        return (
            sum(position[0] for position in positions) / count,
            sum(position[1] for position in positions) / count,
        )

    def update_teammate_memory(self, teammate_data):
        teammate_anchor = self.get_teammate_anchor(teammate_data)
        if teammate_anchor is None:
            return
        self.last_teammate_position = teammate_anchor
        self.last_teammate_seen_at = time.time()

    def is_there_poison_gas(self, player_data, threshold=7000, area_from_player_checked=1.5):
        now = time.monotonic()
        player_center = self.get_entity_pos(player_data)
        if (
            self._poison_cache is not None
            and now - self._poison_cache_at < self.poison_cache_interval
            and self._poison_cache_player is not None
            and self.get_distance(player_center, self._poison_cache_player)
                <= 8.0 * self.window_controller.scale_factor
        ):
            self.poison_cache_count += 1
            return self._poison_cache.copy()
        self.poison_compute_count += 1
        cache_key = (
            "poison", tuple(player_data), float(threshold),
            float(area_from_player_checked),
        )
        cached = self._decision_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        actual_player_box = self.get_actual_player_box(player_data) or player_data
        px1, py1, px2, py2 = actual_player_box
        player_width = max(px2 - px1, 1)
        player_height = max(py2 - py1, 1)
        min_x = int(max(px1 - player_width*area_from_player_checked, 0))
        max_x = int(min(px2 + player_width*area_from_player_checked, self.window_controller.width))
        min_y = int(max(py1 - player_height*area_from_player_checked, 0))
        max_y = int(min(py2 + player_height*area_from_player_checked, self.window_controller.height))

        if min_x >= max_x or min_y >= max_y:
            return {
                "up": 0,
                "down": 0,
                "left": 0,
                "right": 0,
            }

        roi = self.frame[min_y:max_y, min_x:max_x]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        mask = cv2.inRange(hsv_roi, POISON_LOW_HSV, POISON_HIGH_HSV)
        x, y = self.get_entity_pos(actual_player_box)
        roi_w = int(max_x - min_x)
        roi_h = int(max_y - min_y)
        local_px = int(clamp(x - min_x, 0, roi_w))
        local_py = int(clamp(y - min_y, 0, roi_h))

        counts = {
            "up": count_mask_pixels(mask, 0, 0, roi_w, local_py),
            "down": count_mask_pixels(mask, 0, local_py, roi_w, roi_h),
            "left": count_mask_pixels(mask, 0, 0, local_px, roi_h),
            "right": count_mask_pixels(mask, local_px, 0, roi_w, roi_h),
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

        self._decision_cache[cache_key] = result
        self._poison_cache = result.copy()
        self._poison_cache_at = now
        self._poison_cache_player = player_center
        return result.copy()

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
        self.last_walls_data = []
        self.last_bushes_data = []
        self.time_since_walls_checked = 0
        self._target_memory.clear()
        self._prepared_frame_ref = None
        self._prepared_main_data = None
        self.last_attack_at = 0.0
        self.objective_position = None
        self.objective_expires_at = 0.0
        self.last_teammate_position = None
        self.last_teammate_seen_at = 0.0
        self.path_planner.reset()
        self._poison_cache = None
        self._poison_cache_at = 0.0
        self._poison_cache_player = None

    def is_path_blocked(self, player_box, move_direction, walls, distance=None):
        if distance is None:
            distance = self.TILE_SIZE*self.window_controller.scale_factor
        movement = self.movement_to_vector(move_direction)
        if movement is None:
            return False

        magnitude = math.hypot(movement[0], movement[1])
        if magnitude < 1:
            return False

        dx = movement[0] / magnitude * distance
        dy = movement[1] / magnitude * distance
        hit_circle_center, hit_circle_radius = self.get_player_hit_circle(player_box)
        if hit_circle_center is None:
            return False

        new_pos = (hit_circle_center[0] + dx, hit_circle_center[1] + dy)
        nearby_walls = self._nearby_walls(
            player_box, walls, distance + hit_circle_radius
        )
        return self.walls_block_swept_circle(
            hit_circle_center, new_pos, hit_circle_radius, nearby_walls
        )

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

    def _best_open_movement(self, movement, player_box, walls):
        """Choose the next waypoint on a shortest local A* route."""
        start, player_radius = self.get_player_hit_circle(player_box)
        magnitude = math.hypot(*movement)
        if start is None or magnitude < 1 or not walls:
            return movement

        known_distance = min(
            magnitude, self.centered_wall_crop_size * 0.45 * self.window_controller.scale_factor
        )
        goal = (
            start[0] + movement[0] / magnitude * known_distance,
            start[1] + movement[1] / magnitude * known_distance,
        )
        nearby = self._nearby_walls(player_box, walls, known_distance + player_radius)
        preferred_heading = self.last_movement if self.last_movement else movement
        path = self.path_planner.plan(
            start, goal, nearby, player_radius, time.monotonic(), preferred_heading
        )
        if path:
            # Skip grid points already visible from the player. This removes
            # staircase movement while retaining A*'s obstacle choice.
            waypoint = path[0]
            for candidate in path[1:]:
                if self._segment_blocked(start, candidate, player_radius, nearby):
                    break
                waypoint = candidate
            return waypoint[0] - start[0], waypoint[1] - start[1]

        # Fully enclosed/noisy detections: deterministic angular fallback.
        offsets = (math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2, math.pi)
        lookahead = self.TILE_SIZE * 1.5 * self.window_controller.scale_factor
        for offset in offsets:
            candidate = self.rotate_movement(movement, offset)
            if not self.is_path_blocked(player_box, candidate, walls, distance=lookahead):
                return candidate
        return self.rotate_movement(movement, math.pi)

    def navigation_correction(self, movement, player_box, walls):
        magnitude = math.hypot(*movement)
        direct_distance = min(
            magnitude,
            self.centered_wall_crop_size * 0.45 * self.window_controller.scale_factor,
        )
        # Straight-line movement always wins. A stale detour must never
        # override a newly opened direct path to the current goal.
        if magnitude < 1 or not self.is_path_blocked(
            player_box, movement, walls, distance=direct_distance
        ):
            return movement

        return self._best_open_movement(movement, player_box, walls)

    @staticmethod
    def validate_game_data(data):
        if not isinstance(data, dict):
            return False

        # Temporal perception intentionally exposes known classes with empty
        # lists. A key therefore no longer proves that an object was detected.
        for name in ("player", "enemy", "teammate", "wall", "bush"):
            boxes = data.get(name) or []
            data[name] = [box for box in boxes if box is not None and len(box) >= 4]

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
        self.update_teammate_memory(data['teammate'])
        if self.context is None:
            self.context = {
                'brawlers_info': self.brawlers_info,
                'must_brawler_hold_attack': self.must_brawler_hold_attack,
                'TILE_SIZE': self.TILE_SIZE*self.window_controller.scale_factor,
                'get_entity_pos': self.get_entity_pos,
                'get_distance': self.get_distance,
                'get_actual_player_box': self.get_actual_player_box,
                'get_brawler_range': self.get_brawler_range,
                'is_there_enemy': self.is_there_enemy,
                'attack': self.attack,
                'use_hypercharge': self.use_hypercharge,
                'use_super': self.use_super,
                'use_gadget': self.use_gadget,
                'get_random_movement': self.get_random_movement,
                'remember_objective': self.remember_objective,
                'get_recent_objective_movement': self.get_recent_objective_movement,
                'seconds_to_hold_attack_after_reaching_max': self.seconds_to_hold_attack_after_reaching_max,
                "width": brawl_stars_width,
                "height": brawl_stars_height,
                'find_closest_enemy': self.find_closest_enemy,
                'find_closest_teammate': self.find_closest_teammate,
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
        self._dynamic_context = {
            'player_data': data['player'][0],
            'enemy_data': data['enemy'],
            'teammate_data': data['teammate'],
            'brawler': brawler,
            'walls': data['wall'],
            'bushes': data['bush'],
            'is_gadget_ready': self.is_gadget_ready,
            'is_hypercharge_ready': self.is_hypercharge_ready,
            'is_super_ready': self.is_super_ready,
            'current_brawler': self.current_brawler,
            'last_movement': self.last_movement,
            'last_movement_change_time': self.last_movement_change_time,
            'debug': self.verbose_debug,
        }
        if self._playstyle_globals is None or not self._playstyle_context_initialized:
            self.context.update(self._dynamic_context)
        movement = self.get_movement()
        movement_vector = self.movement_to_vector(movement)
        if movement_vector is None:
            self.window_controller.release_movement()
            self.last_movement = ''
            return None
        movement_vector = self.navigation_correction(
            movement_vector, data['player'][0], data['wall']
        )
        movement = self.clamp_movement(movement_vector)
        current_time = time.time()
        if self.last_movement and self.movement_is_similar(movement, self.last_movement):
            movement = self.last_movement
            self.last_movement_change_time = current_time
        elif movement != self.last_movement:
            if current_time - self.last_movement_change_time >= self.minimum_movement_delay:
                self.last_movement = movement
                self.last_movement_change_time = current_time
            else:
                movement = self.last_movement
        else:
            self.last_movement_change_time = current_time
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
        if scale < 1.0 and screenshot.size:
            screenshot = cv2.resize(
                screenshot, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA,
            )
            minimum *= scale * scale
        pixels = count_hsv_pixels(screenshot, low_hsv, high_hsv, self.window_controller)
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
        crop_size = self.centered_wall_crop_size

        if player_data:
            center_x, center_y = self.get_entity_pos(player_data[0])
        else:
            center_x, center_y = frame_width / 2, frame_height / 2

        crop_x1 = int(clamp(round(center_x - crop_size / 2), 0, frame_width - crop_size))
        crop_y1 = int(clamp(round(center_y - crop_size / 2), 0, frame_height - crop_size))
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
                self.Detect_centered_tile_detector = Detect(
                    self.close_tile_detector_model_path,
                    classes=self.tile_detector_model_classes,
                )
            crop, offset_x, offset_y = self.get_centered_wall_crop(frame, player_data)
            tile_data = self.Detect_centered_tile_detector.detect_objects(
                crop,
                conf_tresh=self.wall_detection_confidence
            )
            return self.offset_tile_data(tile_data, offset_x, offset_y)

        if self.Detect_tile_detector is None:
            self.Detect_tile_detector = Detect(
                self.tile_detector_model_path,
                classes=self.tile_detector_model_classes,
            )
        tile_data = self.Detect_tile_detector.detect_objects(frame, conf_tresh=self.wall_detection_confidence)
        return tile_data

    def process_tile_data(self, tile_data):
        walls = []
        bushes = []
        for class_name, boxes in tile_data.items():
            if 'bush' not in class_name:
                walls.extend(boxes)
            else:
                bushes.extend(boxes)
        return self.merge_collinear_walls(walls), bushes

    def merge_collinear_walls(self, walls):
        """Merge adjacent tiles only when they form the same straight wall."""
        if len(walls) < 2:
            return walls
        gap_limit = self.TILE_SIZE * 0.3 * self.window_controller.scale_factor
        merged = [list(map(float, wall[:4])) for wall in walls]
        changed = True
        while changed:
            changed = False
            output = []
            while merged:
                current = merged.pop()
                combined = False
                for index, other in enumerate(merged):
                    current_height = max(1.0, current[3] - current[1])
                    other_height = max(1.0, other[3] - other[1])
                    y_overlap = max(0.0, min(current[3], other[3]) - max(current[1], other[1]))
                    horizontal_gap = max(current[0], other[0]) - min(current[2], other[2])

                    current_width = max(1.0, current[2] - current[0])
                    other_width = max(1.0, other[2] - other[0])
                    x_overlap = max(0.0, min(current[2], other[2]) - max(current[0], other[0]))
                    vertical_gap = max(current[1], other[1]) - min(current[3], other[3])

                    same_row = (
                        current_width >= current_height * 0.8
                        and other_width >= other_height * 0.8
                        and
                        y_overlap >= min(current_height, other_height) * 0.72
                        and horizontal_gap <= gap_limit
                    )
                    same_column = (
                        current_height >= current_width * 0.8
                        and other_height >= other_width * 0.8
                        and
                        x_overlap >= min(current_width, other_width) * 0.72
                        and vertical_gap <= gap_limit
                    )
                    if not (same_row or same_column):
                        continue
                    merged[index] = [
                        min(current[0], other[0]), min(current[1], other[1]),
                        max(current[2], other[2]), max(current[3], other[3]),
                    ]
                    combined = True
                    changed = True
                    break
                if not combined:
                    output.append(current)
            merged = output
        return [[round(value) for value in wall] for wall in merged]

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
            for key in ["player", "enemy", "teammate", "wall"]:
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
                walls, bushes = self.process_tile_data(tile_data)
                self.time_since_walls_checked = current_time
                self.last_walls_data = walls
                data['wall'] = walls
                self.last_bushes_data = bushes
                data['bush'] = bushes
                self.wall_scene_gate.accept(wall_sample, current_time)
            else:
                self.wall_cache_count += 1
                data['wall'] = self.last_walls_data
                data['bush'] = self.last_bushes_data
        else:
            data['wall'] = self.last_walls_data
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
        if not self.is_hypercharge_ready and current_time - self.time_since_hypercharge_checked > self.hypercharge_treshold:
            self.is_hypercharge_ready = self.check_if_hypercharge_ready(frame)
            self.time_since_hypercharge_checked = current_time
        if not self.is_gadget_ready and current_time - self.time_since_gadget_checked > self.gadget_treshold:
            self.is_gadget_ready = self.check_if_gadget_ready(frame)
            self.time_since_gadget_checked = current_time
        if not self.is_super_ready and current_time - self.time_since_super_checked > self.super_treshold:
            self.is_super_ready = self.check_if_super_ready(frame)
            self.time_since_super_checked = current_time
        self.frame = frame
        movement = self.loop(brawler, data, current_time)
        self.publish_debug_view(frame, data, state, movement)
        if movement is not None:
            self.do_movement(movement)
