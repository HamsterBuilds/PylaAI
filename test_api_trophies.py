import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from utils import get_player_info, get_brawler_stats


class ApiTrophyTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows credential protection')
    def test_saved_key_roundtrips_without_plaintext(self):
        import brawl_api_credentials as credentials
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'key.bin'
            with patch.object(credentials, 'credential_path', return_value=path):
                with patch.dict(os.environ, {'BRAWL_STARS_API_TOKEN': ''}):
                    credentials.save_token('test-key-not-a-real-secret')
                    self.assertNotIn(b'test-key-not-a-real-secret', path.read_bytes())
                    self.assertEqual(credentials.load_token(), 'test-key-not-a-real-secret')

    def test_profile_request_and_encoded_tag(self):
        response = Mock()
        response.json.return_value = {'tag': '#ABC', 'brawlers': []}
        with patch.dict(os.environ, {'BRAWL_STARS_API_TOKEN': 'test'}):
            with patch('utils.requests.get', return_value=response) as request:
                self.assertEqual(get_player_info('#abc')['tag'], '#ABC')
                self.assertTrue(request.call_args.args[0].endswith('/%23ABC'))
                self.assertEqual(request.call_args.kwargs['timeout'], (3, 5))

    def test_wrong_account_is_rejected(self):
        response = Mock()
        response.json.return_value = {'tag': '#OTHER'}
        with patch.dict(os.environ, {'BRAWL_STARS_API_TOKEN': 'test'}):
            with patch('utils.requests.get', return_value=response):
                self.assertIsNone(get_player_info('ABC'))

    def test_trophies_are_not_power_or_highest_trophies(self):
        profile = {'brawlers': [{'name': 'MR. P', 'trophies': 503,
                               'highestTrophies': 900, 'power': 11}]}
        self.assertEqual(get_brawler_stats(profile, 'mrp'), (503, None))
        self.assertEqual(get_brawler_stats(profile, 'shelly'), (None, None))

    def test_missing_credentials_do_not_send_request(self):
        with patch.dict(os.environ, {'BRAWL_STARS_API_TOKEN': ''}):
            with patch('utils.requests.get') as request:
                self.assertIsNone(get_player_info('ABC'))
                request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
