"""PyInstaller entry point for the windowed GUI build (EpisodeMatch.exe).

Supports ``--smoke``: boots the GUI, closes it automatically and exits 0 —
used to verify the packaged build without a display (CI/offscreen).
"""

from __future__ import annotations

import sys


def main() -> int:
    smoke = "--smoke" in sys.argv[1:]
    from episodematch.core.config import ConfigStore
    from episodematch.core.exceptions import ConfigError
    from episodematch.ui.main_window import run_gui

    try:
        config = ConfigStore().load()
    except ConfigError as exc:
        if smoke:
            print(f"config problem (continuing with defaults): {exc}")
            config = None
        else:
            raise
    return run_gui(config, smoke=smoke)


if __name__ == "__main__":
    raise SystemExit(main())
