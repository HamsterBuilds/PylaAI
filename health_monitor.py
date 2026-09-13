"""Non-blocking, conservative player-health recognition."""

import os
import re
import subprocess
import tempfile
import threading
import time
from collections import Counter, OrderedDict

import cv2
import numpy as np

from utils import resolve_project_path


_DIGITS = re.compile(r"(?<![A-Za-z0-9])([0-9OoIlSsBbZzGg]{3,5})(?![A-Za-z0-9])")
_TRANSLATION = str.maketrans({
    "O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "s": "5",
    "B": "8", "b": "8", "Z": "2", "z": "2", "G": "6", "g": "6",
})
_WORKER = resolve_project_path("scripts", "windows_ocr_worker.ps1")


class HealthMonitor:
    """OCR health off the gameplay thread and retain only fresh evidence."""

    def __init__(self, interval=0.45, low=1500, healed=3300):
        self.interval = max(0.25, float(interval))
        self.low_threshold = int(low)
        self.healed_threshold = int(healed)
        self._condition = threading.Condition()
        self._pending = None
        self._last_submit = 0.0
        self.value = None
        self.observed_at = 0.0
        self.low_confirmed = False
        self.healed_confirmed = False
        self._low_votes = 0
        self._healed_votes = 0
        self.requests = 0
        self.readings = 0
        self._process = None
        self._generation = 0
        self._closed = False
        temporary_prefix = os.path.join(
            tempfile.gettempdir(), f"pyla-health-{os.getpid()}-{id(self)}"
        )
        self._temporary_paths = (
            temporary_prefix + "-0.bmp", temporary_prefix + "-1.bmp"
        )
        self._temporary_path = self._temporary_paths[0]
        self._temporary_slot = 0
        self._prepare_source_shape = None
        self._prepare_gray = None
        self._prepare_resized = None
        self._prepare_equalized = None
        self._prepare_bright = None
        self._prepare_adaptive = None
        self._prepare_borders = None
        self._prepare_output = None
        self._prepare_cache = OrderedDict()
        self._worker = threading.Thread(
            target=self._run, daemon=True, name="health-ocr"
        )
        self._worker.start()

    def close(self):
        """Stop the persistent OCR helper and release its temporary file."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        process, self._process = self._process, None
        if process is not None:
            try:
                process.stdin.close()
            except (AttributeError, OSError):
                pass
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        if self._worker is not threading.current_thread():
            self._worker.join(timeout=0.5)
        for temporary_path in self._temporary_paths:
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass

    def reset(self):
        with self._condition:
            self._pending = None
            self._generation += 1
            self.value = None
            self.observed_at = 0.0
            self.low_confirmed = False
            self.healed_confirmed = False
            self._low_votes = 0
            self._healed_votes = 0

    def submit(self, frame, player_box, now=None):
        if self._closed:
            return
        now = time.monotonic() if now is None else now
        if now - self._last_submit < self.interval:
            return
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in player_box[:4])
        box_w, box_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        center_x = (x1 + x2) * 0.5
        # The health number is immediately above the detected player. Keep the
        # region narrow enough that damage numbers and nearby players cannot vote.
        crop_x1 = max(0, int(center_x - max(105.0, box_w * 1.8)))
        crop_x2 = min(width, int(center_x + max(105.0, box_w * 1.8)))
        crop_y1 = max(0, int(y1 - max(115.0, box_h * 2.0)))
        crop_y2 = min(height, int(y1 + box_h * 0.18))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return
        # Copying this small crop is cheap and makes it safe after scrcpy
        # replaces/reuses the source frame.
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        with self._condition:
            self._pending = (crop, self._generation)
            self._last_submit = now
            self.requests += 1
            self._condition.notify()

    def snapshot(self, maximum_age=1.8):
        fresh = time.monotonic() - self.observed_at <= maximum_age
        return {
            "value": self.value if fresh else None,
            "fresh": fresh,
            "low": self.low_confirmed,
            "healed": self.healed_confirmed,
        }

    def _start_worker(self):
        self._process = subprocess.Popen(
            [
                "powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
                "-ExecutionPolicy", "Bypass", "-File", str(_WORKER),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
            ),
        )

    def _recognize(self, image_path):
        with self._condition:
            if self._closed:
                return ""
            if self._process is None or self._process.poll() is not None:
                self._start_worker()
            process = self._process
        try:
            process.stdin.write(str(image_path) + "\n")
            process.stdin.flush()
            return process.stdout.readline()
        except (BrokenPipeError, OSError):
            with self._condition:
                if self._process is process:
                    self._process = None
            return ""

    def _prepare(self, crop):
        source_shape = crop.shape[:2]
        if self._prepare_source_shape != source_shape:
            self._prepare_source_shape = source_shape
            buffers = self._prepare_cache.get(source_shape)
            if buffers is None:
                gray = np.empty(source_shape, dtype=np.uint8)
                scale = min(3.0, 1300.0 / max(source_shape[1], 1))
                # Let OpenCV choose the exact fx/fy rounding once, then retain
                # that shape for every equal-sized player crop.
                initial = cv2.resize(
                    gray, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                resized_shape = initial.shape
                resized = np.empty(resized_shape, dtype=np.uint8)
                equalized = np.empty(resized_shape, dtype=np.uint8)
                bright = np.empty(resized_shape, dtype=np.uint8)
                adaptive = np.empty(resized_shape, dtype=np.uint8)
                bordered_shape = (
                    resized_shape[0] + 36, resized_shape[1] + 36
                )
                borders = tuple(
                    np.empty(bordered_shape, dtype=np.uint8)
                    for _ in range(3)
                )
                output = np.empty(
                    (bordered_shape[0] * 3, bordered_shape[1]),
                    dtype=np.uint8,
                )
                buffers = (
                    gray, resized, equalized, bright, adaptive,
                    borders, output,
                )
                self._prepare_cache[source_shape] = buffers
                if len(self._prepare_cache) > 3:
                    self._prepare_cache.popitem(last=False)
            else:
                self._prepare_cache.move_to_end(source_shape)
            (
                self._prepare_gray,
                self._prepare_resized,
                self._prepare_equalized,
                self._prepare_bright,
                self._prepare_adaptive,
                self._prepare_borders,
                self._prepare_output,
            ) = buffers

        cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY, dst=self._prepare_gray)
        cv2.resize(
            self._prepare_gray,
            (self._prepare_resized.shape[1], self._prepare_resized.shape[0]),
            dst=self._prepare_resized, interpolation=cv2.INTER_CUBIC,
        )
        cv2.equalizeHist(
            self._prepare_resized, dst=self._prepare_equalized
        )
        cv2.threshold(
            self._prepare_equalized, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            dst=self._prepare_bright,
        )
        cv2.adaptiveThreshold(
            self._prepare_equalized, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7,
            dst=self._prepare_adaptive,
        )
        first, second, third = self._prepare_borders
        cv2.copyMakeBorder(
            self._prepare_bright, 18, 18, 18, 18,
            cv2.BORDER_CONSTANT, dst=first, value=255,
        )
        cv2.bitwise_not(self._prepare_bright, dst=self._prepare_resized)
        cv2.copyMakeBorder(
            self._prepare_resized, 18, 18, 18, 18,
            cv2.BORDER_CONSTANT, dst=second, value=255,
        )
        cv2.copyMakeBorder(
            self._prepare_adaptive, 18, 18, 18, 18,
            cv2.BORDER_CONSTANT, dst=third, value=255,
        )
        cv2.vconcat(
            (first, second, third), dst=self._prepare_output
        )
        return self._prepare_output

    def _accept(self, text, generation):
        if generation != self._generation:
            return
        candidates = [int(value.translate(_TRANSLATION))
                      for value in _DIGITS.findall(text or "")]
        candidates = [value for value in candidates if 100 <= value <= 20000]
        if not candidates:
            return
        # HP is the stable large number in this player-only crop. Prefer a
        # candidate near the previous reading when OCR exposes multiple variants.
        counts = Counter(candidates)
        value = min(
            counts,
            key=lambda item: (
                -counts[item],
                abs(item - self.value) if self.value is not None else -item,
            ),
        )
        self.value = value
        self.observed_at = time.monotonic()
        self.readings += 1
        if value < self.low_threshold:
            self._low_votes += 1
            self._healed_votes = 0
        elif value >= self.healed_threshold:
            self._healed_votes += 1
            self._low_votes = 0
        else:
            self._low_votes = self._healed_votes = 0
        if self._low_votes >= 2:
            self.low_confirmed = True
            self.healed_confirmed = False
        if self._healed_votes >= 2:
            self.healed_confirmed = True
            self.low_confirmed = False

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                pending, self._pending = self._pending, None
                crop, generation = pending
            try:
                prepared = self._prepare(crop)
                temporary_path = self._temporary_paths[self._temporary_slot]
                self._temporary_slot ^= 1
                if cv2.imwrite(temporary_path, prepared):
                    self._accept(
                        self._recognize(temporary_path), generation
                    )
            except (OSError, subprocess.SubprocessError, cv2.error):
                self._process = None
