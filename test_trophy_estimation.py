import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trophy_observer import TrophyObserver


class TrophyEstimationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.paths = patch(
            "trophy_observer.resolve_project_path",
            side_effect=lambda *parts: root.joinpath(*parts),
        )
        self.config = patch(
            "trophy_observer.load_toml_as_dict",
            return_value={"trophies_multiplier": 1},
        )
        self.paths.start()
        self.config.start()

    def tearDown(self):
        self.config.stop()
        self.paths.stop()
        self.temp.cleanup()

    def estimate_place(self, one_based_place):
        observer = TrophyObserver()
        observer.current_trophies = 500
        observer.current_wins = 0
        observer.current_brawler = "shelly"
        result = observer.parse_game_result(f"trio_showdown_{one_based_place - 1}")
        observer.add_trophies(result, "shelly", {}, False, calculate_when_missing=True)
        return observer.current_trophies

    def test_every_trio_placement_updates_local_trophy_estimate(self):
        self.assertEqual(self.estimate_place(1), 511)
        self.assertEqual(self.estimate_place(2), 505)
        self.assertEqual(self.estimate_place(3), 501)
        self.assertEqual(self.estimate_place(4), 498)


if __name__ == "__main__":
    unittest.main()
