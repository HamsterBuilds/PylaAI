"""Small, bounded temporal filters for noisy frame-by-frame detections."""

from __future__ import annotations

from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(slots=True)
class _Track:
    box: list[float]
    velocity: list[float] | None = None
    missed: int = 0


def _iou(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1.0)


def _center_distance_sq(first, second):
    first_x = (first[0] + first[2]) * 0.5
    first_y = (first[1] + first[3]) * 0.5
    second_x = (second[0] + second[2]) * 0.5
    second_y = (second[1] + second[3]) * 0.5
    dx, dy = first_x - second_x, first_y - second_y
    return dx * dx + dy * dy


class DetectionStabilizer:
    """Smooth matched boxes and bridge one missed player frame.

    Enemy boxes are never retained after a miss, so the bot cannot attack a
    target which has disappeared. Storage is bounded by max_tracks per class.
    """

    _MAX_MISSES = {"player": 1, "teammate": 1, "enemy": 0}

    def __init__(self, smoothing=0.68, max_tracks=16, prediction=0.25, velocity_smoothing=0.55):
        self.smoothing = min(max(float(smoothing), 0.0), 1.0)
        self.max_tracks = max(1, int(max_tracks))
        self.prediction = min(max(float(prediction), 0.0), 0.75)
        self.velocity_smoothing = min(max(float(velocity_smoothing), 0.0), 1.0)
        self._tracks = {name: [] for name in self._MAX_MISSES}
        self._fresh = {name: False for name in self._MAX_MISSES}

    def reset(self):
        for tracks in self._tracks.values():
            tracks.clear()
        for name in self._fresh:
            self._fresh[name] = False

    @staticmethod
    def _match_score(track_box, detection):
        overlap = _iou(track_box, detection)
        if overlap >= 0.15:
            return 2.0 + overlap
        width = max(track_box[2] - track_box[0], detection[2] - detection[0], 1.0)
        height = max(track_box[3] - track_box[1], detection[3] - detection[1], 1.0)
        distance_limit = max(width, height) * 1.25
        distance_sq = _center_distance_sq(track_box, detection)
        if distance_sq <= distance_limit * distance_limit:
            return 1.0 - math.sqrt(distance_sq) / distance_limit
        return None

    def _update_class(self, name, detections):
        tracks = self._tracks[name]
        available = set(range(len(tracks)))
        updated = []

        for raw_box in detections[:self.max_tracks]:
            box = [float(value) for value in raw_box[:4]]
            index = None
            best_score = -1.0
            for candidate_index in available:
                score = self._match_score(tracks[candidate_index].box, box)
                if score is not None and score > best_score:
                    index = candidate_index
                    best_score = score
            if index is not None:
                available.remove(index)
                previous = tracks[index].box
                alpha = self.smoothing
                smoothed = [previous[i] + (box[i] - previous[i]) * alpha for i in range(4)]
                old_velocity = tracks[index].velocity or [0.0] * 4
                velocity_alpha = self.velocity_smoothing
                velocity = [
                    old_velocity[i] * (1.0 - velocity_alpha)
                    + (box[i] - previous[i]) * velocity_alpha
                    for i in range(4)
                ]
                updated.append(_Track(smoothed, velocity))
            else:
                updated.append(_Track(box))

        max_misses = self._MAX_MISSES[name]
        for index in available:
            track = tracks[index]
            track.missed += 1
            if track.missed <= max_misses and len(updated) < self.max_tracks:
                updated.append(track)

        self._tracks[name] = updated
        return [
            [
                round(value + (track.velocity[i] if track.velocity else 0.0) * self.prediction)
                for i, value in enumerate(track.box)
            ]
            for track in updated
        ]

    def update(self, detections):
        stable = dict(detections)
        for name in self._tracks:
            self._fresh[name] = bool(detections.get(name))
            stable[name] = self._update_class(name, detections.get(name, []))
        return stable

    def was_observed(self, name):
        return self._fresh.get(name, False)


class SceneChangeGate:
    """Decide whether an expensive scene detector needs a refresh."""

    def __init__(self, threshold=4.0, maximum_age=1.25, minimum_age=0.45,
                 sample_size=(96, 54)):
        self.threshold = max(0.0, float(threshold))
        self.maximum_age = max(0.05, float(maximum_age))
        self.minimum_age = min(
            self.maximum_age, max(0.0, float(minimum_age))
        )
        self.sample_size = tuple(int(max(8, value)) for value in sample_size)
        self._accepted_sample = None
        self._accepted_at = 0.0

    def reset(self):
        self._accepted_sample = None
        self._accepted_at = 0.0

    def sample(self, frame):
        reduced = cv2.resize(frame, self.sample_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(reduced, cv2.COLOR_RGB2GRAY)

    def should_refresh(self, frame, now):
        if self._accepted_sample is None:
            candidate = self.sample(frame)
            return True, candidate
        age = now - self._accepted_at
        if age < self.minimum_age:
            return False, None
        candidate = self.sample(frame)
        if age >= self.maximum_age:
            return True, candidate
        difference = cv2.absdiff(candidate, self._accepted_sample)
        return float(np.mean(difference)) >= self.threshold, candidate

    def accept(self, sample, now):
        self._accepted_sample = sample
        self._accepted_at = now


class StateConsensus:
    """Reject isolated state classifications without delaying stable states."""

    def __init__(self, confirmations=2):
        self.confirmations = max(1, int(confirmations))
        self.current = None
        self.candidate = None
        self.candidate_count = 0

    def reset(self):
        self.current = None
        self.candidate = None
        self.candidate_count = 0

    def observe(self, state):
        if state is None:
            return self.current
        if state == self.current:
            self.candidate = None
            self.candidate_count = 0
            return self.current
        if state == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = state
            self.candidate_count = 1
        if self.candidate_count >= self.confirmations:
            self.current = self.candidate
            self.candidate = None
            self.candidate_count = 0
        return self.current

    def force(self, state):
        self.current = state
        self.candidate = None
        self.candidate_count = 0
        return state
