"""Fast local A* navigation for screen-space detections."""

from __future__ import annotations

import heapq
import math


class LocalPathPlanner:
    _SQRT2 = math.sqrt(2)
    _NEIGHBORS = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, _SQRT2), (1, -1, _SQRT2),
        (-1, 1, _SQRT2), (1, 1, _SQRT2),
    )

    def __init__(self, area_size=640, cell_size=36, cache_seconds=0.25):
        self.area_size = max(160, int(area_size))
        self.cell_size = max(16, int(cell_size))
        self.cache_seconds = max(0.0, float(cache_seconds))
        self.reset()
        self.plan_requests = 0
        self.path_cache_hits = 0
        self.occupancy_rebuilds = 0
        self.expanded_nodes = 0

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
        for p, q in (
            (-dx, x - rect[0]), (dx, rect[2] - x),
            (-dy, y - rect[1]), (dy, rect[3] - y),
        ):
            if abs(p) < 1e-9:
                if q < 0:
                    return False
                continue
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
    def _segment_clear(cls, start, end, walls, padding):
        for wall in walls:
            expanded = (
                wall[0] - padding, wall[1] - padding,
                wall[2] + padding, wall[3] + padding,
            )
            if cls._segment_intersects_rect(start, end, expanded):
                return False
        return True

    @classmethod
    def _smooth_path(cls, start, path, walls, padding):
        """Replace grid staircases with the farthest collision-free waypoints."""
        if len(path) < 2:
            return path
        points = [start, *path]
        smoothed = []
        anchor_index = 0
        last_index = len(points) - 1
        while anchor_index < last_index:
            next_index = last_index
            while next_index > anchor_index + 1 and not cls._segment_clear(
                points[anchor_index], points[next_index], walls, padding
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

    def plan(self, start, goal, walls, player_radius, now, preferred_heading=None):
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
        key = (start_cell, goal_cell, occupancy_key, heading_key)
        if key == self._cache_key and now < self._cache_until:
            self.path_cache_hits += 1
            return list(self._cache_path)

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

        def is_blocked(cell):
            # Detector padding may overlap the player's current cell. It must
            # remain a valid starting point without copying/mutating occupancy.
            return cell != start_cell and cell in blocked

        if is_blocked(goal_cell):
            free = [(x, y) for x in range(size) for y in range(size)
                    if not is_blocked((x, y))]
            if not free:
                return []
            goal_cell = min(free, key=lambda cell: self._distance(to_point(cell), local_goal))

        queue = [(0.0, 0.0, start_cell)]
        came_from, cost = {}, {start_cell: 0.0}
        reached = None
        while queue:
            _, queued_cost, current = heapq.heappop(queue)
            if queued_cost != cost.get(current):
                continue
            self.expanded_nodes += 1
            if current == goal_cell:
                reached = current
                break
            for ox, oy, step_cost in self._NEIGHBORS:
                nxt = current[0] + ox, current[1] + oy
                if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or is_blocked(nxt):
                    continue
                if ox and oy and (
                    is_blocked((current[0] + ox, current[1]))
                    or is_blocked((current[0], current[1] + oy))
                ):
                    continue
                turn_cost = 0.0
                direction_angle = math.atan2(oy, ox)
                if current == start_cell and preferred_angle is not None:
                    # Commit to the chosen side of an obstacle unless the
                    # alternative is materially shorter. This prevents the
                    # left/right indecision caused by nearly tied A* routes.
                    turn_cost = (1.0 - math.cos(direction_angle - preferred_angle)) * 0.55
                elif current in came_from:
                    previous = came_from[current]
                    previous_angle = math.atan2(
                        current[1] - previous[1], current[0] - previous[0]
                    )
                    turn_cost = (1.0 - math.cos(direction_angle - previous_angle)) * 0.08
                new_cost = cost[current] + step_cost + turn_cost
                if new_cost >= cost.get(nxt, float("inf")):
                    continue
                cost[nxt], came_from[nxt] = new_cost, current
                heuristic = self._octile_distance(goal_cell, nxt)
                heapq.heappush(queue, (new_cost + heuristic, new_cost, nxt))

        path = []
        if reached is not None:
            cursor = reached
            while cursor != start_cell:
                path.append(to_point(cursor))
                cursor = came_from[cursor]
            path.reverse()
            path = self._smooth_path(
                start, path, walls, player_radius + self.cell_size * 0.18
            )
        self._cache_key, self._cache_until = key, now + self.cache_seconds
        self._cache_path = path
        return list(path)
