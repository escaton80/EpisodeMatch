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


def run_pyinstaller(
    name: str,
    *,
    console: bool,
    entry: str,
    distpath: str,
    workpath: str,
    icon: Path | None = None,
    add_data: list[tuple[str, str]] | None = None,
) -> None:
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
    if icon is not None:
        command += ["--icon", str(icon)]
    for source, destination in (add_data or []):
        command += ["--add-data", f"{source}{os.pathsep}{destination}"]
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        raise SystemExit(f"PyInstaller failed for {name} (exit {completed.returncode})")


def _logo_assets(logo_dir: Path) -> tuple[Path | None, Path | None]:
    """Discover the EXE icon (.ico) and window image (.jpg/.png) in logo/."""
    icon = next(iter(sorted(logo_dir.glob("*.ico"))), None) if logo_dir.is_dir() else None
    image = next(
        iter(sorted(logo_dir.glob("*.jpg")) + sorted(logo_dir.glob("*.png"))),
        None,
    ) if logo_dir.is_dir() else None
    return icon, image


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


def find_forbidden_tools(root: Path, *, allowed_top_level: set[str] | None = None) -> list[Path]:
    """Return tool binaries that must not ship inside a bundle.

    Two rules: the six known GPL tool names are forbidden *anywhere*; and
    (audit F3) when ``allowed_top_level`` is given, any OTHER executable at the
    bundle's top level is flagged too — a stray EXE in a PyInstaller onedir
    bundle is almost certainly an accidentally copied third-party tool.
    """
    allowed = {name.lower() for name in allowed_top_level} if allowed_top_level is not None else None
    found: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() != ".exe":
            continue
        if p.name.lower() in FORBIDDEN_TOOL_NAMES:
            found.append(p)
        elif allowed is not None and len(p.relative_to(root).parts) == 1 and p.name.lower() not in allowed:
            found.append(p)
    return sorted(found)


def _app_version() -> str:
    """The released version string (kept in sync with pyproject/__init__)."""
    try:
        from episodematch import __version__
    except ImportError:
        return "0.1.0"
    return __version__


def build_release_zip() -> list[Path]:
    """Assemble the TWO distributable release ZIPs (CLI + GUI) from the built
    bundles — each archive ships its own bundle.

    Per archive: README-FIRST.txt (beta disclaimer + setup steps),
    THIRD_PARTY_LICENSES.txt (LGPL/PSF/PyInstaller attributions),
    LICENSES/LGPL-3.0-only.txt, the project's own MIT LICENSE (audit F3), and
    empty bin/ + data/ placeholders. Raises SystemExit when GPL tool binaries
    (or any other unknown top-level executable) are found inside the bundles.
    """
    import zipfile

    cli_root = PROJECT_ROOT / "dist" / "cli" / "episodematch"
    gui_root = PROJECT_ROOT / "dist" / "gui" / "EpisodeMatch"
    bundle_rules = {
        cli_root: {"episodematch.exe"},
        gui_root: {"EpisodeMatch.exe"},
    }
    for root, allowed in bundle_rules.items():
        if not root.is_dir():
            raise SystemExit(f"Missing bundle: {root}")
        forbidden = find_forbidden_tools(root, allowed_top_level=allowed)
        if forbidden:
            raise SystemExit(
                "Licensing guard: executables that must not ship were found inside the bundle — "
                "GPL tools are fetched by the user via fetch-tools, not shipped: "
                + ", ".join(str(p) for p in forbidden)
            )

    readme_first = PROJECT_ROOT / "packaging" / "README-FIRST.txt"
    third_party = PROJECT_ROOT / "packaging" / "THIRD_PARTY_LICENSES.txt"
    lgpl_text = PROJECT_ROOT / "packaging" / "LICENSES" / "LGPL-3.0-only.txt"
    license_file = PROJECT_ROOT / "LICENSE"

    version = _app_version()
    zips: list[Path] = []
    for label, bundle_root in (("CLI", cli_root), ("GUI", gui_root)):
        zip_path = PROJECT_ROOT / "dist" / f"EpisodeMatch-{label}-v{version}-win64.zip"
        zip_root = f"EpisodeMatch-{label}-v{version}-win64"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.writestr(f"{zip_root}/README-FIRST.txt", readme_first.read_text(encoding="utf-8"))
            zf.writestr(f"{zip_root}/THIRD_PARTY_LICENSES.txt", third_party.read_text(encoding="utf-8"))
            # Audit F3: the archive must carry the project's own MIT license text.
            if license_file.is_file():
                zf.writestr(f"{zip_root}/LICENSE", license_file.read_text(encoding="utf-8"))
            if lgpl_text.is_file():
                zf.write(lgpl_text, f"{zip_root}/LICENSES/LGPL-3.0-only.txt")
            zf.writestr(f"{zip_root}/bin/.gitkeep", "")
            zf.writestr(f"{zip_root}/data/.gitkeep", "")
            # Flat layout: the executable sits directly in the archive root
            # with bin/ and data/ as siblings (no nested cli/ or gui/ folder).
            for p in bundle_root.rglob("*"):
                if p.is_file():
                    zf.write(p, f"{zip_root}/" + p.relative_to(bundle_root).as_posix())
        zips.append(zip_path)
    return zips


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the EpisodeMatch executables")
    parser.add_argument("--smoke", action="store_true", help="run smoke tests after building")
    parser.add_argument("--no-zip", action="store_true", help="skip building the release ZIP")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    logo_dir = PROJECT_ROOT / "logo"
    exe_icon, window_image = _logo_assets(logo_dir)
    # PyInstaller resolves --add-data sources relative to --specpath, so the
    # source must be absolute; "assets" is the bundle-relative destination.
    gui_add_data = [(str(logo_dir), "assets")] if window_image is not None else []
    run_pyinstaller(
        "episodematch",
        console=True,
        entry="packaging/cli_entry.py",
        distpath="dist/cli",
        workpath="build/cli",
        icon=exe_icon,
    )
    run_pyinstaller(
        "EpisodeMatch",
        console=False,
        entry="packaging/gui_entry.py",
        distpath="dist/gui",
        workpath="build/gui",
        icon=exe_icon,
        add_data=gui_add_data,
    )

    cli_exe = PROJECT_ROOT / "dist" / "cli" / "episodematch" / "episodematch.exe"
    gui_exe = PROJECT_ROOT / "dist" / "gui" / "EpisodeMatch" / "EpisodeMatch.exe"
    if not cli_exe.is_file() or not gui_exe.is_file():
        raise SystemExit("Expected executables were not produced")

    if args.smoke:
        smoke_test(gui_exe, cli_exe)

    if not args.no_zip:
        for zip_path in build_release_zip():
            print(f"  ZIP: {zip_path}")

    print("\nBuild complete:")
    print(f"  CLI: {cli_exe}")
    print(f"  GUI: {gui_exe}")
    print("\nNote: run the executables from a folder that contains ./bin/ (external tools)")
    print("and ./data/ (config + cache), e.g. the project root — paths resolve relative to CWD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
