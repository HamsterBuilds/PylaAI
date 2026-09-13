from __future__ import annotations

import collections
import ctypes
import re
import sys
import threading
import time
import traceback
from typing import Any, Callable


GLOBAL_LOGS = collections.deque(maxlen=2000)
GLOBAL_LOGS_LOCK = threading.Lock()


class ThreadFilterStream:
    ANSI_CLEAN_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

    def __init__(self, original_stream, prefix_filter="pyla-", is_stderr=False):
        self.original_stream = original_stream
        self.prefix_filter = prefix_filter
        self.is_stderr = is_stderr
        self.thread_buffers: dict[str, list[str]] = {}

    def write(self, text):
        self.original_stream.write(text)
        if not text:
            return

        thread_name = threading.current_thread().name
        if not thread_name.startswith(self.prefix_filter):
            return

        with GLOBAL_LOGS_LOCK:
            self.thread_buffers.setdefault(thread_name, []).append(text)
            combined = "".join(self.thread_buffers[thread_name])
            if "\n" not in combined:
                return

            lines = combined.split("\n")
            self.thread_buffers[thread_name] = [lines[-1]]
            for line in lines[:-1]:
                log_line = self.ANSI_CLEAN_RE.sub("", line)
                if self.is_stderr:
                    log_line = f"[stderr] {log_line}"
                GLOBAL_LOGS.append(log_line)

    def flush(self):
        self.original_stream.flush()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)


if not getattr(sys.stdout, "_is_pyla_redirected", False):
    sys.stdout = ThreadFilterStream(sys.stdout, prefix_filter="pyla-")
    sys.stdout._is_pyla_redirected = True

if not getattr(sys.stderr, "_is_pyla_redirected", False):
    sys.stderr = ThreadFilterStream(sys.stderr, prefix_filter="pyla-", is_stderr=True)
    sys.stderr._is_pyla_redirected = True


class RuntimeControl:
    def __init__(self, state_callback: Callable[[str], None]):
        self._state_callback = state_callback
        self._stop_event = threading.Event()
        self._pause_requested = threading.Event()
        self._force_stop_callback = None

    def request_pause(self):
        self._pause_requested.set()

    def resume(self):
        self._pause_requested.clear()

    def request_stop(self):
        self._stop_event.set()
        self._pause_requested.clear()

    def set_force_stop_callback(self, callback):
        self._force_stop_callback = callback

    def force_stop(self):
        self.request_stop()
        callback = self._force_stop_callback
        if callback:
            threading.Thread(
                target=callback,
                daemon=True,
                name="pyla-force-cleanup",
            ).start()

    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def should_pause(self) -> bool:
        return self._pause_requested.is_set() and not self._stop_event.is_set()

    def mark_running(self):
        self._state_callback("running")

    def mark_paused(self):
        self._state_callback("paused")


class RuntimeManager:
    def __init__(self, pyla_main):
        self.pyla_main = pyla_main
        self._thread: threading.Thread | None = None
        self.rt_control: RuntimeControl | None = None
        self._lock = threading.Lock()
        self._state = "idle"
        self._last_error = ""
        self._session_started_at: float | None = None
        self.queue_provider: Callable[[], list[dict[str, Any]]] | None = None
        self._auth_provider: Callable[[], dict[str, Any]] | None = None

    def _set_state(self, state: str):
        with self._lock:
            self._state = state

    def configure_start_gate(
            self,
            queue_provider: Callable[[], list[dict[str, Any]]],
            auth_provider: Callable[[], dict[str, Any]],
    ):
        self.queue_provider = queue_provider
        self._auth_provider = auth_provider

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            thread_alive = self._thread.is_alive() if self._thread else False
            if not thread_alive and self._state != "error":
                self._state = "idle"
                self._thread = None
                self.rt_control = None
                self._session_started_at = None
            return {
                "state": self._state,
                "is_running": thread_alive,
                "last_error": self._last_error,
                "session_started_at": self._session_started_at if thread_alive else None,
            }

    def start(self, queue_data: list[dict[str, Any]], discord_bot) -> dict[str, Any]:
        with self._lock:
            thread_alive = self._thread.is_alive() if self._thread else False

            if thread_alive:
                if self._state == "paused" and self.rt_control:
                    self.rt_control.resume()
                    self._state = "running"
                    self._last_error = ""
                    return {"ok": True, "message": "Pyla resumed."}
                return {"ok": False, "message": f"Pyla cannot start while state is {self._state}."}

            self.rt_control = RuntimeControl(self._set_state)
            self._state = "running"
            self._last_error = ""
            self._session_started_at = time.time()
            self._thread = threading.Thread(
                target=self._run_worker,
                args=(queue_data, self.rt_control, discord_bot),
                daemon=True,
                name="pyla-runtime",
            )
            self._thread.start()
            return {"ok": True, "message": "Pyla started."}

    def start_current_queue(self, discord_bot) -> dict[str, Any]:
        if not self.queue_provider or not self._auth_provider:
            return {
                "ok": False,
                "message": "Runtime start gate is not configured.",
                "code": "START_GATE_NOT_CONFIGURED",
            }

        runtime_state = self.get_status()["state"]
        queue_data = self.queue_provider()
        if runtime_state != "paused" and not queue_data:
            return {"ok": False, "message": "Queue is empty.", "code": "EMPTY_QUEUE"}

        auth_state = self._auth_provider()
        if auth_state.get("required") and not auth_state.get("authenticated"):
            return {
                "ok": False,
                "message": auth_state.get("message") or "Login required before starting.",
                "code": auth_state.get("code") or "LOGIN_REQUIRED",
                "auth": auth_state,
            }

        return self.start(queue_data, discord_bot)

    def _run_worker(self, queue_data: list[dict[str, Any]], control: RuntimeControl, discord_bot):
        try:
            self.pyla_main(discord_bot, queue_data, runtime_control=control)
            with self._lock:
                if self._state != "error":
                    self._state = "idle"
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0
            with self._lock:
                if code in (0, None):
                    self._state = "idle"
                    self._last_error = ""
                else:
                    self._state = "error"
                    self._last_error = f"Pyla exited with code {code}."
        except Exception as exc:
            with self._lock:
                self._state = "error"
                self._last_error = str(exc)
            print(str(exc))
            traceback.print_exc()
        finally:
            with self._lock:
                self._thread = None
                self.rt_control = None
                self._session_started_at = None

    def pause(self) -> dict[str, Any]:
        with self._lock:
            thread_alive = self._thread.is_alive() if self._thread else False
            if not thread_alive or not self.rt_control:
                return {"ok": False, "message": "Pyla is not running."}

            if self._state == "running":
                self.rt_control.request_pause()
                self._state = "pausing"
                return {"ok": True, "message": "Pause requested. Pyla will pause in the lobby."}

            if self._state in {"pausing", "paused"}:
                return {"ok": True, "message": "Pause already requested."}

            return {"ok": False, "message": f"Pyla cannot pause while state is {self._state}."}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            thread_alive = self._thread.is_alive() if self._thread else False
            if not thread_alive or not self.rt_control:
                self._state = "idle"
                self._session_started_at = None
                return {"ok": True, "message": "Pyla is already stopped."}

            thread = self._thread
            control = self.rt_control
            control.request_stop()
            self._state = "stopping"

        # Release movement/device resources immediately from the request
        # thread. This does not depend on the gameplay loop reaching its next
        # cooperative stop checkpoint.
        try:
            control.force_stop()
        except Exception as error:
            print(f"Immediate bot cleanup reported: {error}")

        thread.join(timeout=1.5)
        if thread.is_alive():
            self._terminate_thread(thread)
            thread.join(timeout=1.0)

        if not thread.is_alive():
            with self._lock:
                stopped_state = self._state
                self._thread = None
                self.rt_control = None
                self._session_started_at = None
                if self._state != "error":
                    self._state = "idle"
                    stopped_state = "idle"
            if stopped_state == "error":
                return {"ok": False, "message": self._last_error or "Pyla stopped with an error."}
            return {"ok": True, "message": "Pyla force-stopped.", "forced": True}

        return {
            "ok": False,
            "message": "Pyla could not be force-stopped.",
            "code": "FORCE_STOP_FAILED",
        }

    @staticmethod
    def _terminate_thread(thread):
        """Raise SystemExit in a stuck Python gameplay worker."""
        if not thread or thread.ident is None:
            return False
        result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread.ident), ctypes.py_object(SystemExit)
        )
        if result > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(thread.ident), ctypes.c_void_p(0)
            )
            return False
        return result == 1

    def get_logs(self) -> list[str]:
        with GLOBAL_LOGS_LOCK:
            return list(GLOBAL_LOGS)

    def clear_logs(self):
        with GLOBAL_LOGS_LOCK:
            GLOBAL_LOGS.clear()
