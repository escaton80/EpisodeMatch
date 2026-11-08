"""Rebuild both Windows executables with PyInstaller.

Usage:
    python scripts/build_exe.py            # build CLI + GUI into dist/
    python scripts/build_exe.py --smoke    # smoke-test both after building

Outputs:
    dist/cli/episodematch/episodematch.exe   (console CLI)
    dist/gui/EpisodeMatch.exe                (windowed GUI)

Requires the ``pack`` extra (pip install -e .[pack]) and, for a fully
featured bundle, the optional extras you want frozen in (e.g. [gui], [ocr-winrt]).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_pyinstaller(name: str, *, console: bool, entry: str, distpath: str, workpath: str) -> None:
    command = [
        sys.executable,
        "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name", name,
        "--console" if console else "--windowed",
        "--paths", "src",
        "--distpath", distpath,
        "--workpath", workpath,
        "--specpath", workpath,  # keep the generated .spec out of the project root
        entry,
    ]
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        raise SystemExit(f"PyInstaller failed for {name} (exit {completed.returncode})")


def smoke_test(gui_exe: Path, cli_exe: Path) -> None:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    print(f"+ {gui_exe} --smoke", flush=True)
    gui = subprocess.run([str(gui_exe), "--smoke"], env=env, cwd=str(PROJECT_ROOT), check=False)
    if gui.returncode != 0:
        raise SystemExit(f"GUI smoke test failed (exit {gui.returncode})")
    print(f"+ {cli_exe} version", flush=True)
    cli = subprocess.run([str(cli_exe), "version"], cwd=str(PROJECT_ROOT), check=False)
    if cli.returncode != 0:
        raise SystemExit(f"CLI smoke test failed (exit {cli.returncode})")
    print("smoke tests OK", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the EpisodeMatch executables")
    parser.add_argument("--smoke", action="store_true", help="run smoke tests after building")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    run_pyinstaller("episodematch", console=True, entry="packaging/cli_entry.py", distpath="dist/cli", workpath="build/cli")
    run_pyinstaller("EpisodeMatch", console=False, entry="packaging/gui_entry.py", distpath="dist/gui", workpath="build/gui")

    cli_exe = PROJECT_ROOT / "dist" / "cli" / "episodematch" / "episodematch.exe"
    gui_exe = PROJECT_ROOT / "dist" / "gui" / "EpisodeMatch.exe"
    if not cli_exe.is_file() or not gui_exe.is_file():
        raise SystemExit("Expected executables were not produced")

    if args.smoke:
        smoke_test(gui_exe, cli_exe)

    print("\nBuild complete:")
    print(f"  CLI: {cli_exe}")
    print(f"  GUI: {gui_exe}")
    print("\nNote: run the executables from a folder that contains ./bin/ (external tools)")
    print("and ./data/ (config + cache), e.g. the project root — paths resolve relative to CWD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
