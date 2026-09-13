"""Launch the Hamster BOT setup wizard in a native Windows window."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import webview

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "cfg" / "general_config.toml"
HTML = ROOT / "setup" / "index.html"
ART = ROOT / "images" / "hamster_setup.webp"
ICON = ROOT / "images" / "hamster_setup.ico"


class SetupApi:
    def complete_setup(self, desktop=False, start_menu=False):
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        value = CONFIG.read_text(encoding="utf-8") if CONFIG.exists() else ""
        if "setup_complete =" in value:
            value = re.sub(r'^setup_complete\s*=.*$', "setup_complete = true", value, flags=re.M)
        else:
            value = value.rstrip() + "\nsetup_complete = true\n"
        CONFIG.write_text(value, encoding="utf-8")
        try:
            shortcuts = create_shortcuts(bool(desktop), bool(start_menu))
        except Exception as error:
            shortcuts = {"desktop": False, "start_menu": False, "error": str(error)}
        return {"ok": True, "shortcuts": shortcuts}

    def launch_main(self):
        from hamster_bot import launch_dashboard
        process = launch_dashboard()
        if webview.windows:
            webview.windows[0].destroy()
        return {"ok": True, "pid": process.pid}

    def close(self):
        if webview.windows:
            webview.windows[0].destroy()
        return True


def main():
    window = webview.create_window(
        "Hamster BOT Setup",
        url=HTML.as_uri(),
        js_api=SetupApi(),
        width=1280,
        height=720,
        min_size=(980, 600),
        resizable=True,
        fullscreen=True,
        on_top=True,
        background_color="#f4f7ff",
    )
    webview_data = Path(os.environ.get("LOCALAPPDATA", ROOT)) / "HamsterBOT" / "setup-webview"
    webview.start(
        initialize_window,
        (window, "--e2e-complete" in sys.argv),
        debug=False,
        private_mode=False,
        storage_path=str(webview_data),
    )


def initialize_window(window, run_e2e=False):
    set_native_icon()
    if run_e2e:
        time.sleep(0.8)
        window.evaluate_js(
            "page=2; render(); document.querySelector('#desktop').checked=true; "
            "document.querySelector('#start-menu').checked=true; finishSetup();"
        )


def create_shortcuts(desktop, start_menu):
    """Create launchers that always route incomplete installs back to setup."""
    if not desktop and not start_menu:
        return {"desktop": False, "start_menu": False}
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    shell = win32com.client.Dispatch("WScript.Shell")
    python = Path(sys.executable)
    pythonw = python.with_name("pythonw.exe")
    executable = pythonw if pythonw.exists() else python

    def make_link(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        shortcut = shell.CreateShortcut(str(path))
        shortcut.TargetPath = str(executable)
        shortcut.Arguments = f'"{ROOT / "hamster_bot.py"}"'
        shortcut.WorkingDirectory = str(ROOT)
        shortcut.IconLocation = f"{ICON},0"
        shortcut.Description = "Launch Hamster Bot"
        shortcut.Save()

    desktop_ok = False
    start_ok = False
    errors = []
    try:
        if desktop:
            try:
                make_link(Path(shell.SpecialFolders("Desktop")) / "Hamster Bot.lnk")
                desktop_ok = True
            except Exception as error:
                errors.append(f"Desktop shortcut: {error}")
        if start_menu:
            try:
                start_link = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Hamster Bot.lnk"
                make_link(start_link)
                start_ok = True
                try:
                    explorer = win32com.client.Dispatch("Shell.Application")
                    item = explorer.Namespace(str(start_link.parent)).ParseName(start_link.name)
                    for verb in item.Verbs():
                        if "pin to start" in str(verb.Name).replace("&", "").strip().lower():
                            verb.DoIt()
                            break
                except Exception:
                    # Windows may require the user to confirm tile pinning,
                    # but the app remains installed and searchable in Start.
                    pass
            except Exception as error:
                errors.append(f"Start-menu shortcut: {error}")
        return {"desktop": desktop_ok, "start_menu": start_ok, "errors": errors}
    finally:
        pythoncom.CoUninitialize()


def set_native_icon():
    """Replace Python's default title-bar icon after WebView creates its HWND."""
    try:
        import time
        import win32api
        import win32con
        import win32gui
        handle = 0
        for _ in range(30):
            handle = win32gui.FindWindow(None, "Hamster BOT Setup")
            if handle:
                break
            time.sleep(0.1)
        if not handle:
            return
        icon = win32gui.LoadImage(0, str(ICON), win32con.IMAGE_ICON, 0, 0,
                                  win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE)
        win32api.SendMessage(handle, win32con.WM_SETICON, win32con.ICON_SMALL, icon)
        win32api.SendMessage(handle, win32con.WM_SETICON, win32con.ICON_BIG, icon)
    except Exception:
        pass


if __name__ == "__main__":
    main()
