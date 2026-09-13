"""Local Windows-protected API credentials; never packaged with the bot."""
import os
from pathlib import Path


def credential_path():
    return Path(os.environ['LOCALAPPDATA']) / 'HamsterBOT' / 'brawl-api.bin'


def load_token():
    token = os.environ.get('BRAWL_STARS_API_TOKEN', '').strip()
    if token:
        return token
    try:
        import win32crypt
        return win32crypt.CryptUnprotectData(
            credential_path().read_bytes(), None, None, None, 0
        )[1].decode('utf-8')
    except (OSError, KeyError, ImportError):
        return ''
    except Exception:
        # DPAPI rejects data belonging to another Windows account.
        return ''


def save_token(token):
    import win32crypt
    encrypted = win32crypt.CryptProtectData(
        token.encode('utf-8'), 'Hamster BOT Brawl Stars API', None, None, None, 0
    )
    path = credential_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encrypted)


def configure():
    import getpass
    import webbrowser
    from utils import get_player_info, load_toml_as_dict, save_dict_as_toml
    config_path = Path(__file__).resolve().parent / 'cfg' / 'general_config.toml'
    config = load_toml_as_dict(config_path)
    print('Brawl Stars API setup: enter your player tag and developer API key.')
    print('Create a key for this connection at https://developer.brawlstars.com/#/account')
    print('Credentials are saved encrypted for your Windows account. No trophy OCR is used.')
    tag = input('Player tag (blank to cancel): ').strip().upper().replace('%23', '').lstrip('#')
    if not tag:
        return False
    webbrowser.open('https://developer.brawlstars.com/#/account')
    token = getpass.getpass('API key (hidden, blank to cancel): ').strip()
    if not token:
        return False
    previous = os.environ.get('BRAWL_STARS_API_TOKEN')
    os.environ['BRAWL_STARS_API_TOKEN'] = token
    try:
        profile = get_player_info(tag)
        if profile is None:
            print('Validation failed. Check the key, its allowed IP address, and player tag. Nothing saved.')
            return False
        save_token(token)
        config['player_tag'] = tag
        save_dict_as_toml(config, config_path)
        print('API connection verified and saved. Future launches load it automatically.')
        return True
    finally:
        if previous is None:
            os.environ.pop('BRAWL_STARS_API_TOKEN', None)
        else:
            os.environ['BRAWL_STARS_API_TOKEN'] = previous


if __name__ == '__main__':
    configure()
