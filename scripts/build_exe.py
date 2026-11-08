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
import zipfile
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


# ---------------------------------------------------------------------------
# Release ZIP with license compliance guard
# ---------------------------------------------------------------------------

#: GPL-licensed external tools must NEVER end up inside the release archive:
#: EpisodeMatch is MIT; shipping third-party GPL binaries would obligate us to
#: provide their complete source. Instead, `fetch-tools` downloads them on the
#: user's machine directly from the original distribution servers (the user is
#: the downloader). This guard is a hard build failure, not a warning.
FORBIDDEN_TOOL_NAMES = frozenset({
    "ffmpeg.exe", "ffprobe.exe", "mkvmerge.exe", "mkvextract.exe",
    "mkvpropedit.exe", "seconv.exe",
})


def find_forbidden_tools(root: Path) -> list[Path]:
    """Return GPL-licensed tool binaries accidentally placed inside a bundle."""
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".exe" and p.name.lower() in FORBIDDEN_TOOL_NAMES
    )


def build_release_zip() -> Path:
    """Assemble the distributable release ZIP from the built bundles.

    Contents: cli/ + gui/ onedir bundles, README-FIRST.txt (beta disclaimer +
    setup steps), THIRD_PARTY_LICENSES.txt (LGPL/PSF/PyInstaller attributions),
    LICENSES/LGPL-3.0-only.txt, and empty bin/ + data/ placeholders. Raises
    SystemExit when GPL tool binaries are found inside the bundles.
    """
    import zipfile

    cli_root = PROJECT_ROOT / "dist" / "cli" / "episodematch"
    gui_root = PROJECT_ROOT / "dist" / "gui" / "EpisodeMatch"
    for root in (cli_root, gui_root):
        if not root.is_dir():
            raise SystemExit(f"Missing bundle: {root}")
        forbidden = find_forbidden_tools(root)
        if forbidden:
            raise SystemExit(
                "Licensing guard: GPL tool binaries found inside the bundle — "
                "they must be fetched by the user via fetch-tools, not shipped: "
                + ", ".join(str(p) for p in forbidden)
            )

    zip_path = PROJECT_ROOT / "dist" / "EpisodeMatch-v0.1.0-win64.zip"
    readme_first = PROJECT_ROOT / "packaging" / "README-FIRST.txt"
    third_party = PROJECT_ROOT / "packaging" / "THIRD_PARTY_LICENSES.txt"
    lgpl_text = PROJECT_ROOT / "packaging" / "LICENSES" / "LGPL-3.0-only.txt"

    bundles = (("cli/episodematch", cli_root), ("gui/EpisodeMatch", gui_root))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zip_root = f"EpisodeMatch-v0.1.0-win64"
        zf.writestr(f"{zip_root}/README-FIRST.txt", readme_first.read_text(encoding="utf-8"))
        zf.writestr(f"{zip_root}/THIRD_PARTY_LICENSES.txt", third_party.read_text(encoding="utf-8"))
        if lgpl_text.is_file():
            zf.write(lgpl_text, f"{zip_root}/LICENSES/LGPL-3.0-only.txt")
        zf.writestr(f"{zip_root}/bin/.gitkeep", "")
        zf.writestr(f"{zip_root}/data/.gitkeep", "")
        for arc_dir, bundle_root in bundles:
            for p in bundle_root.rglob("*"):
                if p.is_file():
                    zf.write(p, f"{zip_root}/{arc_dir}/" + p.relative_to(bundle_root).as_posix())
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the EpisodeMatch executables")
    parser.add_argument("--smoke", action="store_true", help="run smoke tests after building")
    parser.add_argument("--no-zip", action="store_true", help="skip building the release ZIP")
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

    if not args.no_zip:
        zip_path = build_release_zip()
        print(f"  ZIP: {zip_path}")

    print("\nBuild complete:")
    print(f"  CLI: {cli_exe}")
    print(f"  GUI: {gui_exe}")
    print("\nNote: run the executables from a folder that contains ./bin/ (external tools)")
    print("and ./data/ (config + cache), e.g. the project root — paths resolve relative to CWD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
