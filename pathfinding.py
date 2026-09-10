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

    def __init__(self, area_size=640, cell_size=36, cache_seconds=0.25):
        self.area_size = max(160, int(area_size))
        self.cell_size = max(16, int(cell_size))
        self.cache_seconds = max(0.0, float(cache_seconds))
        # Every non-initial A* turn is between two of the eight fixed grid
        # directions. Compute those identical cosine penalties once instead
        # of repeating atan2/cos for every candidate of every expanded node.
        self._turn_costs = {}
        for previous_x, previous_y, _, previous_angle in self._NEIGHBORS:
            for next_x, next_y, _, next_angle in self._NEIGHBORS:
                self._turn_costs[
                    (previous_x, previous_y, next_x, next_y)
                ] = (1.0 - math.cos(next_angle - previous_angle)) * 0.08
        self.reset()
        self.plan_requests = 0
        self.path_cache_hits = 0
        self.occupancy_rebuilds = 0
        self.expanded_nodes = 0
        self.routes_found = 0
        self.routes_failed = 0
        self.raw_waypoints = 0
        self.smoothed_waypoints = 0

    def reset(self):
        self._cache_key = None
        self._cache_until = 0.0
        self._cache_path = []
        self._occupancy_key = None
        self._occupancy = frozenset()
        self._signature_source = None
        self._signature_value = ()

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
    def _smooth_path(cls, start, path, walls, padding,
                     directional_danger=None, danger_regions=()):
        """Replace grid staircases with the farthest collision-free waypoints."""
        if len(path) < 2:
            return path
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
            while next_index > anchor_index + 1 and (
                shortcut_enters_danger(
                    points[anchor_index], points[next_index]
                )
                or not segment_clear(
                    points[anchor_index], points[next_index], expanded_walls
                )
                or not segment_clear(
                    points[anchor_index], points[next_index], expanded_danger
                )
            ):
                next_index -= 1
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
        occupancy_key = (
            int(origin_x), int(origin_y), size, round(player_radius / 2), wall_signature
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
            padding = player_radius + self.cell_size * 0.18
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
            if not cached_path or (
                self._segment_clear(start, cached_path[0], expanded_walls)
                and self._segment_clear(
                    start, cached_path[0], expanded_danger
                )
            ):
                self.path_cache_hits += 1
                return list(cached_path)
            self._cache_key = None

        if occupancy_key == self._occupancy_key:
            blocked = self._occupancy
        else:
            self.occupancy_rebuilds += 1
            padding = player_radius + self.cell_size * 0.18
            blocked = set()
            # Convert each expanded wall directly to a grid range. This avoids
            # testing every grid cell against every detected wall.
            for wall in walls:
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
            self._occupancy_key = occupancy_key
            self._occupancy = frozenset(blocked)
            blocked = self._occupancy

        # Storm remains traversable because the player may already be inside
        # it, but A* strongly prefers any available safe corridor.
        danger_cells = set()
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
                danger_cells.update(
                    (x, y)
                    for x in range(min_x, max_x + 1)
                    for y in range(min_y, max_y + 1)
                )

        def is_blocked(cell):
            # Detector padding may overlap the player's current cell. It must
            # remain a valid starting point without copying/mutating occupancy.
            return cell != start_cell and cell in blocked

        # Legal cells immediately beside inflated terrain remain available in
        # narrow corridors, but cost more. This keeps normal routes away from
        # collision boundaries where detector jitter caused wall rubbing.
        near_wall_cells = set()
        for blocked_x, blocked_y in blocked:
            for offset_x, offset_y, _, _ in self._NEIGHBORS:
                adjacent = blocked_x + offset_x, blocked_y + offset_y
                if (
                    0 <= adjacent[0] < size
                    and 0 <= adjacent[1] < size
                    and adjacent not in blocked
                ):
                    near_wall_cells.add(adjacent)

        safe_goal_exists = len(danger_cells) < size * size
        goal_cells = frozenset((goal_cell,))
        if is_blocked(goal_cell) or (
            safe_goal_exists and goal_cell in danger_cells
        ):
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

        queue = [(0.0, 0.0, start_cell)]
        came_from, cost = {}, {start_cell: 0.0}
        reached = None
        heappop = heapq.heappop
        heappush = heapq.heappush
        neighbors = self._NEIGHBORS
        turn_costs = self._turn_costs
        infinite = self._INF
        octile_distance = self._octile_distance
        if len(goal_cells) == 1:
            single_goal = next(iter(goal_cells))

            def goal_heuristic(cell):
                return octile_distance(single_goal, cell)
        else:
            ordered_goals = tuple(sorted(goal_cells))
            goal_heuristics = {
                (x, y): min(
                    octile_distance(candidate, (x, y))
                    for candidate in ordered_goals
                )
                for x in range(size)
                for y in range(size)
            }
            goal_heuristic = goal_heuristics.__getitem__
        while queue:
            _, queued_cost, current = heappop(queue)
            if queued_cost != cost.get(current):
                continue
            self.expanded_nodes += 1
            if current in goal_cells:
                reached = current
                break
            current_cost = cost[current]
            previous_step = None
            previous = came_from.get(current)
            if previous is not None:
                previous_step = (
                    current[0] - previous[0], current[1] - previous[1]
                )
            for ox, oy, step_cost, direction_angle in neighbors:
                nxt = current[0] + ox, current[1] + oy
                if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                    continue
                if nxt != start_cell and nxt in blocked:
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
                    ) * 2.4
                elif previous_step is not None:
                    turn_cost = turn_costs[
                        (previous_step[0], previous_step[1], ox, oy)
                    ]
                danger_cost = 6.0 * (
                    (danger_left if ox < 0 else danger_right if ox > 0 else 0.0)
                    + (danger_up if oy < 0 else danger_down if oy > 0 else 0.0)
                )
                storm_cost = 18.0 if nxt in danger_cells else 0.0
                clearance_cost = 1.15 if nxt in near_wall_cells else 0.0
                new_cost = (
                    current_cost + step_cost + turn_cost
                    + danger_cost + storm_cost + clearance_cost
                )
                if new_cost >= cost.get(nxt, infinite):
                    continue
                cost[nxt], came_from[nxt] = new_cost, current
                heuristic = goal_heuristic(nxt)
                heappush(queue, (new_cost + heuristic, new_cost, nxt))

        path = []
        if reached is not None:
            cursor = reached
            while cursor != start_cell:
                path.append(to_point(cursor))
                cursor = came_from[cursor]
            path.reverse()
            raw_count = len(path)
            path = self._smooth_path(
                start, path, walls, player_radius + self.cell_size * 0.18,
                danger_key, danger_regions,
            )
            self.routes_found += 1
            self.raw_waypoints += raw_count
            self.smoothed_waypoints += len(path)
        else:
            self.routes_failed += 1
        self._cache_key, self._cache_until = key, now + self.cache_seconds
        self._cache_path = path
        return list(path)
