import threading
import time
import unittest

from webui.runtime import RuntimeManager


class RuntimeForceStopTests(unittest.TestCase):
    def test_stop_forces_busy_runtime_and_runs_cleanup(self):
        cleanup_called = threading.Event()

        def busy_runtime(_discord, _queue, runtime_control):
            runtime_control.set_force_stop_callback(cleanup_called.set)
            while True:
                time.sleep(0.01)

        manager = RuntimeManager(busy_runtime)
        self.assertTrue(manager.start([{"brawler": "Shelly"}], None)["ok"])
        time.sleep(0.05)

        result = manager.stop()

        self.assertTrue(result["ok"])
        self.assertTrue(result["forced"])
        self.assertTrue(cleanup_called.wait(0.5))
        self.assertEqual(manager.get_status()["state"], "idle")
