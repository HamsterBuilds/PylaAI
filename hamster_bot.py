"""Single entry point used by installed Hamster Bot shortcuts."""
from __future__ import annotations

import subprocess
import sys
import tomllib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def setup_is_complete():
    try:
        with (ROOT / "cfg" / "general_config.toml").open("rb") as file:
            config = tomllib.load(file)
        return bool(config.get("setup_complete"))
    except (OSError, tomllib.TOMLDecodeError):
        return False


def main():
    destination = ROOT / ("main.py" if setup_is_complete() else "setup.py")
    if destination.name == "main.py":
        launch_dashboard()
    else:
        subprocess.Popen([sys.executable, str(destination)], cwd=str(ROOT))


def launch_dashboard():
    python = Path(sys.executable)
    if python.name.lower() == "pythonw.exe":
        python = python.with_name("python.exe")
    log_dir = Path(os.environ.get("LOCALAPPDATA", ROOT)) / "HamsterBOT"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = (log_dir / "dashboard.log").open("a", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(
        [str(python), str(ROOT / "main.py"), "--no-console"],
        cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
        creationflags=flags,
    )


if __name__ == "__main__":
    main()
