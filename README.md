# PylaAI

[![CC BY-NC 4.0 License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/xUusk3fw4A)
[![Trello](https://img.shields.io/badge/Trello-0079BF?logo=trello&logoColor=white)](https://trello.com/b/SAz9J6AA/public-pyla-trello)

> [!WARNING]
> **Warning**: There are two versions of PylaAI, you are currently browsing the source code for developers. Please visit our [Discord](https://discord.gg/xUusk3fw4A) to use the compiled version, which comes as a ready-to-use `.exe`.

PylaAI is currently the best external Brawl Stars bot.

## Requirements

- **NVIDIA GPUs**
  - `setup.py` installs `onnxruntime-gpu` when an NVIDIA GPU is detected
  - Falls back to DirectML, then CPU, if CUDA does not load

- **AMD / Intel / iGPU**
  - DirectML on Windows (`onnxruntime-directml`)
  - CPU fallback if DirectML does not load

## Installation

### Trophy API connection

Trophy totals come from the official Brawl Stars player API. Trophy OCR is no
longer used. On a console launch, missing API settings start a one-time setup:
enter your player tag and a key from the Brawl Stars developer portal for your
connection's allowed IP. The bot validates the account before saving anything.
The key is encrypted by Windows for your account under LocalAppData/HamsterBOT,
outside the repository and release archive. Future launches load it automatically.

To replace the key or account, run `py -3.11 main.py --configure-api`.
`BRAWL_STARS_API_TOKEN` remains available as an environment override. The bot
cannot create a developer account or issue a key without your account access.
If the connection fails, trophy totals remain unchanged rather than being guessed.

### Windows release

Download the latest `HamsterBOT-<version>.zip` from GitHub Releases and extract it to a folder you own. Open the setup wizard with:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 setup
```

(`python .\setup.py` also opens the same UI.)

```powershell
run hamster_bot
```

The release ZIP is created by `.uild_release.ps1 0.1.0`. It contains the runtime assets and models, so users do not need to clone the repository or install development files.

You will need [Python 3.11.9](https://www.python.org/downloads/release/python-3119/).

### Windows

```sh
python setup.py install
```

### Other Platforms

The official PylaAI does **NOT** support other platforms such as Linux or Mac, but you can visit [Unofficial Ports](https://github.com/4D1-TooFarGone/Pyla-Ports) for cross-platform support.

## Using PylaAI

> [!NOTE]
> **Note**: This open-source version runs in localhost mode. The cloud features have been disabled by default.

Run the bot:

```sh
python main.py
```

### Startup options

| Flag | Effect |
| --- | --- |
| *(none)* | Console visible, UI in the Pyla desktop window |
| `--no-console` | Hides the console window, output goes to `pyla.log` in the current folder. Ignored when Pyla is started from an existing terminal, so your own terminal is never hidden. |
| `--no-webapp` | Opens the UI in the default browser instead of the desktop window |


## License

This project is **not permitted to be sold or monetized** under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Maintainer

### Developers

- **ivanyordanovgt**
- **AngelFireLA**
- **awarzu**

### Contributors

- **Maayan080**
- **simonrejzek**
- **bocchi-the-cat**
- **Ariko842**
- **Nauwk07**
- **aetherwtff**
- **fazelukario**
- **k00shi**
- **mydd7**
