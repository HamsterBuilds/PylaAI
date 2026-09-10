import os
import re
import subprocess
import tempfile
from pathlib import Path

import cv2

from utils import resolve_project_path


_OCR_SCRIPT = resolve_project_path("scripts", "windows_ocr.ps1")
_TOKEN_PATTERN = re.compile(
    # Keep separate OCR words separate.  The old expression allowed ordinary
    # spaces inside a token, so output such as "584 11" became "58411" and
    # was discarded for being longer than a trophy total.
    r"(?<![A-Za-z0-9])[0-9OoIlSsBbZzGg](?:[0-9OoIlSsBbZzGg,.]*[0-9OoIlSsBbZzGg])?(?![A-Za-z0-9])"
)
_DIGIT_TRANSLATION = str.maketrans({
    "O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "s": "5",
    "B": "8", "b": "8", "Z": "2", "z": "2", "G": "6", "g": "6",
})


def _numeric_candidates(text):
    values = []
    for token in _TOKEN_PATTERN.findall(text or ""):
        normalized = token.translate(_DIGIT_TRANSLATION)
        normalized = normalized.replace(" ", "").replace(",", "").replace(".", "")
        if normalized.isdigit() and len(normalized) <= 4:
            value = int(normalized)
            if 0 <= value <= 9999:
                values.append(value)
    return values


def _run_windows_ocr(image_path):
    return subprocess.run(
        [
            "powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
            "-ExecutionPolicy", "Bypass", "-File", str(_OCR_SCRIPT),
            "-ImagePath", str(image_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=6,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _build_display_glyph_sheet(frame):
    """Create OCR-friendly views of Brawl Stars' outlined display digits."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # WinRT OCR rejects images above its maximum dimension. Two vertically
    # stacked variants plus borders must remain comfortably below that limit.
    scale = min(
        1.8,
        2200.0 / max(gray.shape[1] + 28, 1),
        2200.0 / max(gray.shape[0] * 2 + 56, 1),
    )
    if abs(scale - 1.0) > 0.01:
        gray = cv2.resize(
            gray, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    equalized = cv2.equalizeHist(gray)
    _, otsu = cv2.threshold(
        equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    adaptive = cv2.adaptiveThreshold(
        equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 7,
    )
    # The raw colour pass already covers ordinary glyphs. The fallback sheet
    # carries the two complementary outlined-digit treatments in one process.
    variants = (cv2.bitwise_not(otsu), adaptive)
    bordered = [
        cv2.copyMakeBorder(
            variant, 14, 14, 14, 14,
            cv2.BORDER_CONSTANT, value=255,
        )
        for variant in variants
    ]
    return cv2.vconcat(bordered)


def read_trophies_from_screen(frame, expected_trophies, maximum_delta=50,
                              result=None, normalized_region=None):
    """Read the post-match total using Windows OCR; never estimate a total."""
    try:
        expected = max(0, int(expected_trophies))
    except (TypeError, ValueError):
        return None

    temporary_path = None
    try:
        if normalized_region is not None:
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = normalized_region
            frame = frame[
                int(height * y1):int(height * y2),
                int(width * x1):int(width * x2),
            ]
            if frame.size == 0:
                return None
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temporary_path = Path(handle.name)
        # OCR only needs glyph contrast; RGB/BGR channel order does not affect
        # the white trophy digits and avoiding conversion saves a full copy.
        if not cv2.imwrite(str(temporary_path), frame):
            return None
        completed = _run_windows_ocr(temporary_path)
        if completed.returncode != 0:
            error_text = (completed.stderr or completed.stdout).strip()
            print(f"Windows trophy OCR failed: {error_text or completed.returncode}")
            return None
        ocr_text = completed.stdout

        def plausible_values(text):
            values = [
                value for value in _numeric_candidates(text)
                if abs(value - expected) <= maximum_delta
            ]
            if result == "victory":
                values = [value for value in values if value >= expected]
            elif result == "defeat":
                values = [value for value in values if value <= expected]
            return values

        plausible = plausible_values(ocr_text)
        if not plausible:
            # Brawl Stars uses outlined display glyphs. Windows OCR is much
            # more reliable on a larger, monochrome copy when the normal
            # colour image yields no numeric words.
            glyph_sheet = _build_display_glyph_sheet(frame)
            if cv2.imwrite(str(temporary_path), glyph_sheet):
                fallback = _run_windows_ocr(temporary_path)
                if fallback.returncode == 0:
                    ocr_text = fallback.stdout
                    plausible = plausible_values(ocr_text)
                else:
                    error_text = (fallback.stderr or fallback.stdout).strip()
                    print(
                        f"Windows trophy OCR fallback failed: "
                        f"{error_text or fallback.returncode}"
                    )
        if not plausible:
            print(
                f"Trophy OCR found no plausible total near {expected}; "
                f"recognized: {ocr_text.strip()!r}"
            )
            return None
        return min(plausible, key=lambda value: abs(value - expected))
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
