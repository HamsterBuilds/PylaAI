"""Fast local A* navigation for screen-space detections."""

from __future__ import annotations

import heapq
import math


class LocalPathPlanner:
    _SQRT2 = math.sqrt(2)
    _INF = float("inf")
    _NEIGHBORS = (
        (-1, 0, 1.0, math.atan2(0, -1)),
        (1, 0, 1.0, math.atan2(0, 1)),
        (0, -1, 1.0, math.atan2(-1, 0)),
        (0, 1, 1.0, math.atan2(1, 0)),
        (-1, -1, _SQRT2, math.atan2(-1, -1)),
        (1, -1, _SQRT2, math.atan2(-1, 1)),
        (-1, 1, _SQRT2, math.atan2(1, -1)),
        (1, 1, _SQRT2, math.atan2(1, 1)),
    )
    _UNIQUE_EDGES = ((1, 0), (0, 1), (1, 1), (1, -1))

    def __init__(self, area_size=640, cell_size=36, cache_seconds=0.25,
                 wall_padding=None):
        self.area_size = max(160, int(area_size))
        self.cell_size = max(16, int(cell_size))
        self.cache_seconds = max(0.0, float(cache_seconds))
        self.wall_padding = (
            self.cell_size * 0.18
            if wall_padding is None
            else max(0.0, float(wall_padding))
        )
        self.reset()
        self.plan_requests = 0
        self.path_cache_hits = 0
        self.stale_path_rejections = 0
        self.occupancy_rebuilds = 0
        self.expanded_nodes = 0
        self.routes_found = 0
        self.routes_failed = 0
        self.raw_waypoints = 0
        self.smoothed_waypoints = 0
        self.overlap_escape_cells_opened = 0
        self.total_route_distance = 0.0
        self.total_direct_distance = 0.0
        self.maximum_route_stretch = 1.0
        self.backward_first_steps = 0

    def reset(self):
        self._cache_key = None
        self._cache_until = 0.0
        self._cache_path = []
        self._cache_traverses_danger = False
        self._occupancy_key = None
        self._occupancy = frozenset()
        self._near_wall_cells = frozenset()
        self._blocked_edges = frozenset()
        self._danger_occupancy_key = None
        self._danger_cells = frozenset()
        self._signature_source = None
        self._signature_value = ()

    def set_wall_padding(self, wall_padding):
        """Update resolution-scaled clearance without retaining stale grids."""
        value = max(0.0, float(wall_padding))
        if abs(value - self.wall_padding) < 1e-6:
            return
        self.wall_padding = value
        self.reset()

    def set_area_size(self, area_size):
        """Match the grid to the currently visible wall-model crop."""
        value = max(160, int(area_size))
        if value == self.area_size:
            return
        self.area_size = value
        self.reset()

    @staticmethod
    def _distance(first, second):
        return math.hypot(first[0] - second[0], first[1] - second[1])

    @classmethod
    def _octile_distance(cls, first, second):
        dx = abs(first[0] - second[0])
        dy = abs(first[1] - second[1])
        diagonal = min(dx, dy)
        return max(dx, dy) + (cls._SQRT2 - 1.0) * diagonal

    @staticmethod
    def _segment_intersects_rect(start, end, rect):
        """Allocation-free Liang-Barsky segment/rectangle test."""
        x, y = start
        dx, dy = end[0] - x, end[1] - y
        enter, leave = 0.0, 1.0

        p, q = -dx, x - rect[0]
        if abs(p) < 1e-9:
            if q < 0:
                return False
        else:
            ratio = q / p
            if p < 0:
                if ratio > leave:
                    return False
                if ratio > enter:
                    enter = ratio
            else:
                if ratio < enter:
                    return False
                if ratio < leave:
                    leave = ratio

        p, q = dx, rect[2] - x
        if abs(p) < 1e-9:
            if q < 0:
                return False
        else:
            ratio = q / p
            if p < 0:
                if ratio > leave:
                    return False
                if ratio > enter:
                    enter = ratio
            else:
                if ratio < enter:
                    return False
                if ratio < leave:
                    leave = ratio

        p, q = -dy, y - rect[1]
        if abs(p) < 1e-9:
            if q < 0:
                return False
        else:
            ratio = q / p
            if p < 0:
                if ratio > leave:
                    return False
                if ratio > enter:
                    enter = ratio
            else:
                if ratio < enter:
                    return False
                if ratio < leave:
                    leave = ratio

        p, q = dy, rect[3] - y
        if abs(p) < 1e-9:
            if q < 0:
                return False
        else:
            ratio = q / p
            if p < 0:
                if ratio > leave:
                    return False
                if ratio > enter:
                    enter = ratio
            else:
                if ratio < enter:
                    return False
                if ratio < leave:
                    leave = ratio
        return enter <= leave

    @classmethod
    def _segment_clear(cls, start, end, expanded_walls):
        for wall in expanded_walls:
            if cls._segment_intersects_rect(start, end, wall):
                return False
        return True

    @classmethod
    def _segment_clear_while_exiting(cls, start, end, expanded_walls):
        """Allow the first step to leave detector padding that overlaps us.

        Wall detections wobble around the player's hit circle.  Treating an
        inflated rectangle which already contains ``start`` as an ordinary
        collision makes every possible first segment invalid.  We only relax
        that one case, and only for motion toward the nearest edge; crossing
        or moving deeper into the rectangle remains forbidden.
        """
        move_x, move_y = end[0] - start[0], end[1] - start[1]
        for rect in expanded_walls:
            if not cls._segment_intersects_rect(start, end, rect):
                continue
            if not (rect[0] <= start[0] <= rect[2]
                    and rect[1] <= start[1] <= rect[3]):
                return False
            edge = min(
                (
                    (start[0] - rect[0], -1.0, 0.0),
                    (rect[2] - start[0], 1.0, 0.0),
                    (start[1] - rect[1], 0.0, -1.0),
                    (rect[3] - start[1], 0.0, 1.0),
                ),
                key=lambda item: item[0],
            )
            if move_x * edge[1] + move_y * edge[2] <= 1e-6:
                return False
        return True

    @classmethod
    def _path_clear(cls, start, path, expanded_obstacles):
        """Validate every segment of a short, already-smoothed path."""
        previous = start
        for index, waypoint in enumerate(path):
            checker = (
                cls._segment_clear_while_exiting
                if index == 0 else cls._segment_clear
            )
            if not checker(previous, waypoint, expanded_obstacles):
                return False
            previous = waypoint
        return True

    @classmethod
    def _smooth_path(cls, start, path, walls, padding,
                     directional_danger=None, danger_regions=(),
                     expanded_walls=None, expanded_danger=None):
        """Replace grid staircases with the farthest collision-free waypoints."""
        if not path:
            return path
        if expanded_walls is None:
            expanded_walls = tuple(
                (
                    wall[0] - padding, wall[1] - padding,
                    wall[2] + padding, wall[3] + padding,
                )
                for wall in walls
            )
        if expanded_danger is None:
            expanded_danger = tuple(
                (
                    region[0] - padding, region[1] - padding,
                    region[2] + padding, region[3] + padding,
                )
                for region in danger_regions
            )
        points = [start, *path]
        smoothed = []
        anchor_index = 0
        last_index = len(points) - 1
        segment_clear = cls._segment_clear
        if directional_danger is None:
            danger_up = danger_down = danger_left = danger_right = 0.0
        else:
            danger_up, danger_down, danger_left, danger_right = directional_danger

        def shortcut_enters_danger(first, second):
            dx = second[0] - first[0]
            dy = second[1] - first[1]
            return (
                (dx < 0 and danger_left > 0)
                or (dx > 0 and danger_right > 0)
                or (dy < 0 and danger_up > 0)
                or (dy > 0 and danger_down > 0)
            )

        while anchor_index < last_index:
            next_index = last_index
            while next_index > anchor_index:
                adjacent = next_index == anchor_index + 1
                wall_checker = (
                    cls._segment_clear_while_exiting
                    if anchor_index == 0 else segment_clear
                )
                wall_clear = wall_checker(
                    points[anchor_index], points[next_index], expanded_walls
                )
                # A* may intentionally traverse storm when no safe route
                # exists, so poison blocks shortcuts but not its validated
                # adjacent grid steps. Terrain is never optional: even an
                # adjacent centre-to-centre segment must be collision-free.
                danger_clear = adjacent or (
                    not shortcut_enters_danger(
                        points[anchor_index], points[next_index]
                    )
                    and segment_clear(
                        points[anchor_index], points[next_index],
                        expanded_danger,
                    )
                )
                if wall_clear and danger_clear:
                    break
                next_index -= 1
            if next_index == anchor_index:
                # Cell-centre occupancy can miss a very thin obstacle between
                # two legal cells. Never emit that unsafe adjacent segment.
                return []
            smoothed.append(points[next_index])
            anchor_index = next_index
        return smoothed

    def _wall_signature(self, walls):
        if walls is self._signature_source:
            return self._signature_value
        # Detector boxes commonly wobble by several pixels even when neither
        # the player nor the obstacle changed. Quantize at half a navigation
        # cell so that harmless vision noise does not invalidate a good route.
        quantum = max(8.0, self.cell_size * 0.5)
        signature = tuple(sorted(
            (round(w[0] / quantum), round(w[1] / quantum),
             round(w[2] / quantum), round(w[3] / quantum))
            for w in walls
        ))
        self._signature_source = walls
        self._signature_value = signature
        return signature

    def plan(self, start, goal, walls, player_radius, now,
             preferred_heading=None, directional_danger=None,
             danger_regions=()):
        self.plan_requests += 1
        half = self.area_size * 0.5
        size = max(5, int(math.ceil(self.area_size / self.cell_size)))
        # Anchor the local grid to world/screen cell boundaries. Small detector
        # jitter then reuses the same occupancy map instead of rebuilding it.
        origin_x = math.floor((start[0] - half) / self.cell_size) * self.cell_size
        origin_y = math.floor((start[1] - half) / self.cell_size) * self.cell_size
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        scale = min(1.0, half / max(abs(dx), abs(dy), 1.0))
        local_goal = (start[0] + dx * scale, start[1] + dy * scale)

        def to_cell(point):
            return (
                min(size - 1, max(0, int((point[0] - origin_x) / self.cell_size))),
                min(size - 1, max(0, int((point[1] - origin_y) / self.cell_size))),
            )

        def to_point(cell):
            return (
                origin_x + (cell[0] + 0.5) * self.cell_size,
                origin_y + (cell[1] + 0.5) * self.cell_size,
            )

        start_cell, goal_cell = to_cell(start), to_cell(local_goal)
        wall_signature = self._wall_signature(walls)
        planning_padding = player_radius + self.wall_padding
        start_overlaps_padding = any(
            wall[0] - planning_padding <= start[0]
            <= wall[2] + planning_padding
            and wall[1] - planning_padding <= start[1]
            <= wall[3] + planning_padding
            for wall in walls
        )
        # Normal movement keeps the coarse, camera-stable occupancy key. When
        # detector padding overlaps the player, however, the legal escape
        # neighbour depends on the player's position inside this cell. Track
        # that exceptional position at quarter-cell precision so a cached
        # opening on the wrong side cannot be reused.
        escape_position_key = None
        if start_overlaps_padding:
            quarter_cell = self.cell_size * 0.25
            escape_position_key = (
                round(start[0] / quarter_cell),
                round(start[1] / quarter_cell),
            )
        occupancy_key = (
            int(origin_x), int(origin_y), size, round(player_radius / 2),
            wall_signature, escape_position_key,
        )
        preferred_angle = None
        if preferred_heading is not None:
            px, py = preferred_heading
            if math.hypot(px, py) >= 1:
                preferred_angle = math.atan2(py, px)
        heading_key = None if preferred_angle is None else round(preferred_angle / (math.pi / 8))
        if directional_danger is None:
            danger_up = danger_down = danger_left = danger_right = 0.0
        else:
            danger_up, danger_down, danger_left, danger_right = (
                max(0.0, min(1.0, float(value)))
                for value in directional_danger
            )
        danger_key = (
            round(danger_up, 2), round(danger_down, 2),
            round(danger_left, 2), round(danger_right, 2),
        )
        danger_quantum = max(8.0, self.cell_size * 0.5)
        danger_region_key = tuple(sorted(
            (
                round(region[0] / danger_quantum),
                round(region[1] / danger_quantum),
                round(region[2] / danger_quantum),
                round(region[3] / danger_quantum),
            )
            for region in danger_regions
        ))
        key = (
            start_cell, goal_cell, occupancy_key, heading_key, danger_key,
            danger_region_key,
        )
        # Geometry is expressed in screen coordinates. Even if quantized
        # cells/signatures stay equal, camera movement gradually makes an old
        # waypoint stale, so the configured short expiry remains necessary.
        if key == self._cache_key and now <= self._cache_until:
            cached_path = self._cache_path
            padding = planning_padding
            expanded_walls = tuple(
                (
                    wall[0] - padding, wall[1] - padding,
                    wall[2] + padding, wall[3] + padding,
                )
                for wall in walls
            )
            expanded_danger = tuple(
                (
                    region[0] - padding, region[1] - padding,
                    region[2] + padding, region[3] + padding,
                )
                for region in danger_regions
            )
            wall_geometry_clear = (
                not cached_path
                or self._path_clear(start, cached_path, expanded_walls)
            )
            wall_path_clear = wall_geometry_clear
            if wall_path_clear and cached_path:
                first_dx = cached_path[0][0] - start[0]
                first_dy = cached_path[0][1] - start[1]
                goal_dx = local_goal[0] - start[0]
                goal_dy = local_goal[1] - start[1]
                first_length = math.hypot(first_dx, first_dy)
                goal_length = math.hypot(goal_dx, goal_dy)
                if (
                    first_length >= 1.0
                    and goal_length >= 1.0
                    and first_dx * goal_dx + first_dy * goal_dy < 0.0
                ):
                    probe_length = min(
                        self.cell_size * 0.65, goal_length
                    )
                    forward_probe = (
                        start[0] + goal_dx / goal_length * probe_length,
                        start[1] + goal_dy / goal_length * probe_length,
                    )
                    if self._segment_clear_while_exiting(
                        start, forward_probe, expanded_walls
                    ):
                        # The player has passed a cached first waypoint while
                        # remaining in the same quantized cell. Reusing it is
                        # a literal backward step, not route stability.
                        wall_path_clear = False
            danger_path_clear = (
                not cached_path
                or self._cache_traverses_danger
                or self._path_clear(start, cached_path, expanded_danger)
            )
            if wall_path_clear and danger_path_clear:
                self.path_cache_hits += 1
                return list(cached_path)
            self.stale_path_rejections += 1
            # A quantized signature intentionally tolerates detector jitter.
            # Once exact geometry proves that jitter intersects the route,
            # however, retaining the matching occupancy cache would simply
            # reproduce that unsafe route on the next A* pass.
            if not wall_geometry_clear:
                self._occupancy_key = None
            if not danger_path_clear:
                self._danger_occupancy_key = None
            self._cache_key = None

        if occupancy_key == self._occupancy_key:
            blocked = self._occupancy
            near_wall_cells = self._near_wall_cells
            blocked_edges = self._blocked_edges
        else:
            self.occupancy_rebuilds += 1
            padding = planning_padding
            blocked = set()
            expanded_wall_rects = []
            # Convert each expanded wall directly to a grid range. This avoids
            # testing every grid cell against every detected wall.
            for wall in walls:
                expanded_rect = (
                    wall[0] - padding, wall[1] - padding,
                    wall[2] + padding, wall[3] + padding,
                )
                expanded_wall_rects.append(expanded_rect)
                min_x = max(0, math.ceil((wall[0] - padding - origin_x) / self.cell_size - 0.5))
                max_x = min(size - 1, math.floor((wall[2] + padding - origin_x) / self.cell_size - 0.5))
                min_y = max(0, math.ceil((wall[1] - padding - origin_y) / self.cell_size - 0.5))
                max_y = min(size - 1, math.floor((wall[3] + padding - origin_y) / self.cell_size - 0.5))
                if min_x > max_x or min_y > max_y:
                    continue
                blocked.update(
                    (gx, gy)
                    for gx in range(min_x, max_x + 1)
                    for gy in range(min_y, max_y + 1)
                )
            # If noisy detector padding contains the player, expose only
            # neighbouring cells which genuinely move toward an exit.  This
            # prevents the common all-neighbours-blocked A* failure without
            # opening any route into or across real terrain.
            for ox, oy, _, _ in self._NEIGHBORS:
                candidate = (start_cell[0] + ox, start_cell[1] + oy)
                if not (0 <= candidate[0] < size and 0 <= candidate[1] < size):
                    continue
                if self._segment_clear_while_exiting(
                    start, to_point(candidate), expanded_wall_rects
                ):
                    if candidate in blocked:
                        self.overlap_escape_cells_opened += 1
                    blocked.discard(candidate)

            self._occupancy_key = occupancy_key
            self._occupancy = frozenset(blocked)
            blocked = self._occupancy
            near_wall = set()
            for blocked_x, blocked_y in blocked:
                for offset_x, offset_y, _, _ in self._NEIGHBORS:
                    adjacent = blocked_x + offset_x, blocked_y + offset_y
                    if (
                        0 <= adjacent[0] < size
                        and 0 <= adjacent[1] < size
                        and adjacent not in blocked
                    ):
                        near_wall.add(adjacent)
            self._near_wall_cells = frozenset(near_wall)
            near_wall_cells = self._near_wall_cells

            # A thin obstacle can sit between two legal cell centres without
            # owning either cell. Cache those forbidden centre-to-centre edges
            # once per occupancy rebuild so A* cannot cross it.
            blocked_edge_set = set()
            for rect in expanded_wall_rects:
                edge_min_x = max(
                    0, int(math.floor((rect[0] - origin_x) / self.cell_size)) - 1
                )
                edge_max_x = min(
                    size - 1,
                    int(math.floor((rect[2] - origin_x) / self.cell_size)) + 1,
                )
                edge_min_y = max(
                    0, int(math.floor((rect[1] - origin_y) / self.cell_size)) - 1
                )
                edge_max_y = min(
                    size - 1,
                    int(math.floor((rect[3] - origin_y) / self.cell_size)) + 1,
                )
                for cell_x in range(edge_min_x, edge_max_x + 1):
                    for cell_y in range(edge_min_y, edge_max_y + 1):
                        cell = (cell_x, cell_y)
                        if cell != start_cell and cell in blocked:
                            continue
                        cell_point = to_point(cell)
                        for edge_x, edge_y in self._UNIQUE_EDGES:
                            neighbor = cell_x + edge_x, cell_y + edge_y
                            if not (
                                0 <= neighbor[0] < size
                                and 0 <= neighbor[1] < size
                                and (
                                    neighbor == start_cell
                                    or neighbor not in blocked
                                )
                            ):
                                continue
                            if cell == start_cell or neighbor == start_cell:
                                # Detector padding may overlap the player.
                                # Keep an escape edge available; the caller's
                                # swept-circle guard chooses the outward one.
                                continue
                            if self._segment_intersects_rect(
                                cell_point, to_point(neighbor), rect
                            ):
                                blocked_edge_set.add((cell, neighbor))
            self._blocked_edges = frozenset(blocked_edge_set)
            blocked_edges = self._blocked_edges

        # Storm remains traversable because the player may already be inside
        # it, but A* strongly prefers any available safe corridor.
        danger_occupancy_key = (
            int(origin_x), int(origin_y), size,
            round(player_radius / 2), danger_region_key,
        )
        if danger_occupancy_key == self._danger_occupancy_key:
            danger_cells = self._danger_cells
        else:
            danger_cells_set = set()
            danger_padding = player_radius * 0.8
            for region in danger_regions:
                min_x = max(0, math.ceil(
                    (region[0] - danger_padding - origin_x) / self.cell_size - 0.5
                ))
                max_x = min(size - 1, math.floor(
                    (region[2] + danger_padding - origin_x) / self.cell_size - 0.5
                ))
                min_y = max(0, math.ceil(
                    (region[1] - danger_padding - origin_y) / self.cell_size - 0.5
                ))
                max_y = min(size - 1, math.floor(
                    (region[3] + danger_padding - origin_y) / self.cell_size - 0.5
                ))
                if min_x <= max_x and min_y <= max_y:
                    danger_cells_set.update(
                        (x, y)
                        for x in range(min_x, max_x + 1)
                        for y in range(min_y, max_y + 1)
                    )
            self._danger_occupancy_key = danger_occupancy_key
            self._danger_cells = frozenset(danger_cells_set)
            danger_cells = self._danger_cells

        def is_blocked(cell):
            # Detector padding may overlap the player's current cell. It must
            # remain a valid starting point without copying/mutating occupancy.
            return cell != start_cell and cell in blocked

        # Legal cells immediately beside inflated terrain remain available in
        # narrow corridors, but cost more. This keeps normal routes away from
        # collision boundaries where detector jitter caused wall rubbing.
        if danger_cells:
            # A cell is useful as a safe replacement only when it is both out
            # of gas and traversable. Counting danger cells alone treated wall
            # and water cells as possible safe goals; in an enclosed pocket
            # that produced no replacement and made the bot stand still.
            safe_goal_exists = any(
                (x, y) != start_cell
                and (x, y) not in blocked
                and (x, y) not in danger_cells
                for x in range(size)
                for y in range(size)
            )
        else:
            safe_goal_exists = True
        goal_cells = frozenset((goal_cell,))
        goal_was_replaced = False
        if is_blocked(goal_cell) or (
            safe_goal_exists and goal_cell in danger_cells
        ):
            goal_was_replaced = True
            # Search expanding perimeters instead of allocating/scoring every
            # cell in the grid. Keep every legal cell on the first available
            # ring: selecting only one before A* could choose an unreachable
            # side of a wall and report failure despite an open opposite side.
            replacements = None
            gx, gy = goal_cell
            for radius in range(1, size):
                candidates = set()
                min_x, max_x = max(0, gx - radius), min(size - 1, gx + radius)
                min_y, max_y = max(0, gy - radius), min(size - 1, gy + radius)
                for x in range(min_x, max_x + 1):
                    for y in (min_y, max_y):
                        cell = (x, y)
                        if cell != start_cell and not is_blocked(cell) and (
                            not safe_goal_exists or cell not in danger_cells
                        ):
                            candidates.add(cell)
                for y in range(min_y + 1, max_y):
                    for x in (min_x, max_x):
                        cell = (x, y)
                        if cell != start_cell and not is_blocked(cell) and (
                            not safe_goal_exists or cell not in danger_cells
                        ):
                            candidates.add(cell)
                if candidates:
                    # A blocked goal commonly lies inside a long wall. Cells
                    # on the nearest ring are equally close to that goal, but
                    # some can be behind the player. Prefer replacements that
                    # still make forward progress; otherwise A* may choose a
                    # technically short route in the opposite direction.
                    goal_dx = local_goal[0] - start[0]
                    goal_dy = local_goal[1] - start[1]
                    forward_candidates = {
                        cell for cell in candidates
                        if (
                            (to_point(cell)[0] - start[0]) * goal_dx
                            + (to_point(cell)[1] - start[1]) * goal_dy
                        ) >= 0
                    }
                    if forward_candidates:
                        candidates = forward_candidates
                    replacements = frozenset(candidates)
                    break
            if replacements is None:
                return []
            goal_cells = replacements

        came_from, cost = {}, {start_cell: 0.0}
        reached = None
        heappop = heapq.heappop
        heappush = heapq.heappush
        neighbors = self._NEIGHBORS
        infinite = self._INF
        octile_distance = self._octile_distance
        if len(goal_cells) == 1:
            single_goal = next(iter(goal_cells))

            def goal_heuristic(cell):
                return octile_distance(single_goal, cell)
        else:
            # Keep the exact multi-goal heuristic, but compute it lazily only
            # for cells A* actually queues. The previous eager size² table did
            # size² × replacement-count calculations even when the target was
            # reached after exploring a small fraction of the local grid.
            ordered_goals = tuple(sorted(goal_cells))
            goal_heuristic_cache = {}

            def goal_heuristic(cell):
                cached = goal_heuristic_cache.get(cell)
                if cached is not None:
                    return cached
                value = min(
                    octile_distance(candidate, cell)
                    for candidate in ordered_goals
                )
                goal_heuristic_cache[cell] = value
                return value
        start_heuristic = goal_heuristic(start_cell)
        # On equal f-scores prefer the state nearer the goal. The old queue
        # used g as its second key, which preferred cells near the start and
        # needlessly fanned the search sideways across much of the local map.
        # This tie-break preserves A* optimality while normally expanding far
        # fewer cells and producing less arbitrary left/right route choices.
        goal_vector_x = goal_cell[0] - start_cell[0]
        goal_vector_y = goal_cell[1] - start_cell[1]
        goal_vector_length = max(
            1.0, math.hypot(goal_vector_x, goal_vector_y)
        )

        def route_deviation(cell):
            # Tie-break equally short alternatives toward the centre line.
            # This removes grid-induced zigzags without sacrificing distance.
            relative_x = cell[0] - start_cell[0]
            relative_y = cell[1] - start_cell[1]
            return abs(
                relative_x * goal_vector_y
                - relative_y * goal_vector_x
            ) / goal_vector_length

        queue = [(
            start_heuristic, start_heuristic,
            route_deviation(start_cell), 0.0, start_cell,
        )]
        while queue:
            _, _, _, queued_cost, current = heappop(queue)
            if queued_cost != cost.get(current):
                continue
            self.expanded_nodes += 1
            if current in goal_cells:
                reached = current
                break
            current_cost = cost[current]
            for ox, oy, step_cost, direction_angle in neighbors:
                nxt = current[0] + ox, current[1] + oy
                if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                    continue
                if nxt != start_cell and nxt in blocked:
                    continue
                edge = (
                    (current, nxt) if current < nxt else (nxt, current)
                )
                if edge in blocked_edges:
                    continue
                if ox and oy:
                    side_x = (current[0] + ox, current[1])
                    side_y = (current[0], current[1] + oy)
                    if (
                        (side_x != start_cell and side_x in blocked)
                        or (side_y != start_cell and side_y in blocked)
                    ):
                        continue
                turn_cost = 0.0
                if current == start_cell and preferred_angle is not None:
                    # Commit to the chosen side of an obstacle unless the
                    # alternative is materially shorter. This prevents the
                    # left/right indecision caused by nearly tied A* routes.
                    # A route must be materially better before switching to
                    # the opposite side of an obstacle. The old 0.55 weight
                    # allowed detector jitter to flip near-tied routes.
                    turn_cost = (
                        1.0 - math.cos(direction_angle - preferred_angle)
                    ) * 0.65
                danger_cost = 6.0 * (
                    (danger_left if ox < 0 else danger_right if ox > 0 else 0.0)
                    + (danger_up if oy < 0 else danger_down if oy > 0 else 0.0)
                )
                storm_cost = 18.0 if nxt in danger_cells else 0.0
                # Physical player-radius + padding already guarantees safety.
                # Keep this as a tie-break only: a multi-cell penalty made A*
                # choose visibly long loops instead of the shortest safe lane.
                clearance_cost = 0.55 if nxt in near_wall_cells else 0.0
                new_cost = (
                    current_cost + step_cost + turn_cost
                    + danger_cost + storm_cost + clearance_cost
                )
                if new_cost >= cost.get(nxt, infinite):
                    continue
                cost[nxt], came_from[nxt] = new_cost, current
                heuristic = goal_heuristic(nxt)
                heappush(
                    queue,
                    (
                        new_cost + heuristic, heuristic,
                        route_deviation(nxt), new_cost, nxt,
                    ),
                )

        path = []
        if reached is not None:
            cursor = reached
            while cursor != start_cell:
                path.append(to_point(cursor))
                cursor = came_from[cursor]
            path.reverse()
            reusable_expanded_walls = None
            reusable_expanded_danger = None
            # Grid centres are search states, not the strategic destination.
            # Finish at the exact local goal when that final segment is safe;
            # otherwise preserve the reachable replacement beside terrain.
            if not goal_was_replaced:
                last_point = path[-1] if path else start
                exact_goal_padding = (
                    player_radius + self.wall_padding
                )
                exact_goal_walls = tuple(
                    (
                        wall[0] - exact_goal_padding,
                        wall[1] - exact_goal_padding,
                        wall[2] + exact_goal_padding,
                        wall[3] + exact_goal_padding,
                    )
                    for wall in walls
                )
                exact_goal_danger = tuple(
                    (
                        region[0] - exact_goal_padding,
                        region[1] - exact_goal_padding,
                        region[2] + exact_goal_padding,
                        region[3] + exact_goal_padding,
                    )
                    for region in danger_regions
                )
                reusable_expanded_walls = exact_goal_walls
                reusable_expanded_danger = exact_goal_danger
                if (
                    self._segment_clear(
                        last_point, local_goal, exact_goal_walls
                    )
                    and self._segment_clear(
                        last_point, local_goal, exact_goal_danger
                    )
                ):
                    if not path or self._distance(path[-1], local_goal) >= 1:
                        path.append(local_goal)
            raw_count = len(path)
            path = self._smooth_path(
                start, path, walls, player_radius + self.wall_padding,
                danger_key, danger_regions,
                reusable_expanded_walls, reusable_expanded_danger,
            )
            self.raw_waypoints += raw_count
            self.smoothed_waypoints += len(path)
            if path:
                route_distance = 0.0
                previous_point = start
                for point in path:
                    route_distance += self._distance(previous_point, point)
                    previous_point = point
                direct_distance = max(
                    1.0, self._distance(start, local_goal)
                )
                self.total_route_distance += route_distance
                self.total_direct_distance += direct_distance
                self.maximum_route_stretch = max(
                    self.maximum_route_stretch,
                    route_distance / direct_distance,
                )
                first_dx = path[0][0] - start[0]
                first_dy = path[0][1] - start[1]
                goal_dx = local_goal[0] - start[0]
                goal_dy = local_goal[1] - start[1]
                if first_dx * goal_dx + first_dy * goal_dy < 0.0:
                    self.backward_first_steps += 1
                self.routes_found += 1
            else:
                self.routes_failed += 1
        else:
            self.routes_failed += 1
        self._cache_key, self._cache_until = key, now + self.cache_seconds
        self._cache_path = path
        if path and danger_regions:
            danger_padding = player_radius + self.wall_padding
            final_expanded_danger = (
                reusable_expanded_danger
                if reached is not None
                and reusable_expanded_danger is not None
                else tuple(
                    (
                        region[0] - danger_padding,
                        region[1] - danger_padding,
                        region[2] + danger_padding,
                        region[3] + danger_padding,
                    )
                    for region in danger_regions
                )
            )
            self._cache_traverses_danger = not self._path_clear(
                start, path, final_expanded_danger
            )
        else:
            self._cache_traverses_danger = False
        return list(path)
