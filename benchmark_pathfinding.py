"""Offline benchmark for PylaAI's local A* planner. Does not launch the game."""

import argparse
import math
import time

from pathfinding import LocalPathPlanner


SCENARIOS = {
    "open_straight": ((320, 320), (580, 320), []),
    "single_wall": ((180, 320), (460, 320), [[300, 220, 350, 420]]),
    "l_wall": ((170, 180), (450, 500), [[280, 160, 330, 420], [280, 370, 470, 420]]),
    "corridor": ((170, 320), (470, 320), [[180, 180, 520, 245], [180, 395, 520, 460]]),
    "offset_barriers": (
        (180, 320), (500, 320),
        [[240, 110, 290, 370], [380, 270, 430, 530]],
    ),
    "long_water_strip": (
        (320, 540), (320, 100), [[80, 280, 560, 350]],
    ),
    "narrow_gate": (
        (120, 320), (530, 320),
        [[290, 20, 350, 225], [290, 415, 350, 620]],
    ),
}


def distance(first, second):
    return math.hypot(second[0] - first[0], second[1] - first[1])


def route_length(points):
    return sum(distance(first, second) for first, second in zip(points, points[1:]))


def complete_route(start, path, goal):
    """Append the exact goal only when the planner did not already do so."""
    points = [start, *path]
    if not points or distance(points[-1], goal) >= 1e-6:
        points.append(goal)
    return points


def direction_changes(points, threshold_degrees=12):
    changes = 0
    previous = None
    threshold = math.radians(threshold_degrees)
    for first, second in zip(points, points[1:]):
        angle = math.atan2(second[1] - first[1], second[0] - first[0])
        if previous is not None:
            delta = abs((angle - previous + math.pi) % (2 * math.pi) - math.pi)
            changes += delta >= threshold
        previous = angle
    return changes


def segment_intersects_rect(start, end, rect):
    x, y = start
    dx, dy = end[0] - x, end[1] - y
    enter, leave = 0.0, 1.0
    for p, q in ((-dx, x - rect[0]), (dx, rect[2] - x),
                 (-dy, y - rect[1]), (dy, rect[3] - y)):
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


def collision_free(points, walls, radius):
    expanded = [
        (wall[0] - radius, wall[1] - radius,
         wall[2] + radius, wall[3] + radius)
        for wall in walls
    ]
    return not any(
        segment_intersects_rect(first, second, wall)
        for first, second in zip(points, points[1:])
        for wall in expanded
    )


def legacy_greedy_route(start, goal, walls, radius, step=36, max_steps=80):
    """Approximate the old direct/X/Y/cardinal movement policy."""
    route = [start]
    current = start
    for _ in range(max_steps):
        dx, dy = goal[0] - current[0], goal[1] - current[1]
        remaining = math.hypot(dx, dy)
        if remaining <= step and collision_free([current, goal], walls, radius):
            route.append(goal)
            return route, True
        candidates = [(dx, dy), (dx, 0), (0, dy)]
        candidates.extend(((step, 0), (-step, 0), (0, step), (0, -step)))
        moved = False
        for move_x, move_y in candidates:
            magnitude = math.hypot(move_x, move_y)
            if magnitude < 1:
                continue
            nxt = (
                current[0] + move_x / magnitude * min(step, magnitude),
                current[1] + move_y / magnitude * min(step, magnitude),
            )
            if collision_free([current, nxt], walls, radius):
                route.append(nxt)
                current = nxt
                moved = True
                break
        if not moved:
            break
    return route, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args()
    planner = LocalPathPlanner(
        area_size=640, cell_size=36, cache_seconds=0.25, wall_padding=12
    )
    radius = 53
    required_clearance = radius + planner.wall_padding
    successful = 0
    legacy_successful = 0
    new_turns = 0
    legacy_turns = 0

    print("Scenario             Legacy  A*     LegacyLen  A*Len   LegacyTurns  A*Turns")
    print("------------------------------------------------------------------------")
    for name, (start, goal, walls) in SCENARIOS.items():
        planner.reset()
        path = planner.plan(start, goal, walls, radius, time.monotonic(), (1, 0))
        points = complete_route(start, path, goal)
        found = bool(path) and collision_free(
            points, walls, required_clearance
        )
        legacy_points, legacy_found = legacy_greedy_route(start, goal, walls, radius)
        successful += int(found)
        legacy_successful += int(legacy_found)
        scenario_new_turns = direction_changes(points)
        scenario_legacy_turns = direction_changes(legacy_points)
        new_turns += scenario_new_turns
        legacy_turns += scenario_legacy_turns
        print(
            f"{name:20} {str(legacy_found):6}  {str(found):5}  "
            f"{route_length(legacy_points):9.1f}  {route_length(points):6.1f}  "
            f"{scenario_legacy_turns:11d}  {scenario_new_turns:7d}"
        )

    start, goal, walls = SCENARIOS["offset_barriers"]
    cached_start = time.perf_counter()
    for _ in range(max(1, args.repetitions)):
        planner.plan(start, goal, walls, radius, time.monotonic(), (1, 0))
    cached_ms = (time.perf_counter() - cached_start) * 1000
    success_rate = successful / len(SCENARIOS) * 100
    legacy_success_rate = legacy_successful / len(SCENARIOS) * 100
    relative_success_gain = (
        (success_rate - legacy_success_rate) / max(legacy_success_rate, 1.0) * 100
    )
    turn_reduction = (legacy_turns - new_turns) / max(legacy_turns, 1) * 100
    waypoint_reduction = 100.0 * (
        planner.raw_waypoints - planner.smoothed_waypoints
    ) / max(1, planner.raw_waypoints)
    hit_rate = planner.path_cache_hits / max(1, planner.plan_requests) * 100

    # Force repeated replans while a detected wall jitters by a few pixels.
    # The selected side should remain stable instead of alternating each tick.
    jitter_planner = LocalPathPlanner(
        area_size=640, cell_size=36, cache_seconds=0.25, wall_padding=12
    )
    jitter_start, jitter_goal = (180, 320), (470, 320)
    jitter_signs = []
    jitter_collision_free = True
    preferred = (1.0, 0.0)
    jitter_time = time.monotonic()
    for index, jitter in enumerate((0, 4, -3, 5, -4, 2, -2, 3)):
        jitter_walls = [[300 + jitter, 220, 350 + jitter, 420]]
        jitter_path = jitter_planner.plan(
            jitter_start, jitter_goal, jitter_walls, radius,
            jitter_time + index * 0.3, preferred,
        )
        jitter_points = complete_route(
            jitter_start, jitter_path, jitter_goal
        )
        jitter_collision_free &= bool(jitter_path) and collision_free(
            jitter_points, jitter_walls,
            radius + jitter_planner.wall_padding,
        )
        if jitter_path:
            first_heading = (
                jitter_path[0][0] - jitter_start[0],
                jitter_path[0][1] - jitter_start[1],
            )
            preferred = first_heading
            if abs(first_heading[1]) >= 1:
                jitter_signs.append(1 if first_heading[1] > 0 else -1)
    jitter_side_changes = sum(
        first != second
        for first, second in zip(jitter_signs, jitter_signs[1:])
    )
    print("------------------------------------------------------------------------")
    print(f"Legacy route success: {legacy_success_rate:.1f}%")
    print(f"A* route success: {success_rate:.1f}%")
    print(f"Relative success improvement: {relative_success_gain:.1f}%")
    print(f"Direction-change reduction: {turn_reduction:.1f}%")
    print(f"A* waypoint reduction after smoothing: {waypoint_reduction:.1f}%")
    print(f"A* routes found/failed: {planner.routes_found}/{planner.routes_failed}")
    print(f"Cached {args.repetitions} requests: {cached_ms:.2f} ms")
    print(f"A* cache hit rate: {hit_rate:.1f}%")
    print(f"A* occupancy rebuilds: {planner.occupancy_rebuilds}")
    print(f"Current blocked grid edges: {len(planner._blocked_edges)}")
    print(f"Wall-jitter side changes: {jitter_side_changes}")
    print(f"Wall-jitter routes collision-free: {jitter_collision_free}")
    path_target_met = (
        successful == len(SCENARIOS)
        and max(relative_success_gain, turn_reduction) >= 80.0
        and jitter_collision_free
        and jitter_side_changes == 0
    )
    print(
        "PATHFINDING TARGET: "
        + ("PASS" if path_target_met else "NOT YET")
        + " (requires collision-free success in every scenario and >=80% "
          "improvement in route success or direction changes, plus zero "
          "wall-jitter side changes)"
    )


if __name__ == "__main__":
    main()
