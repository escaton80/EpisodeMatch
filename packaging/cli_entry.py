"""PyInstaller entry point for the console CLI build (episodematch.exe)."""

from __future__ import annotations

from episodematch.main import main

if __name__ == "__main__":
    raise SystemExit(main())
