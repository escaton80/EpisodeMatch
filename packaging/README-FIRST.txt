EpisodeMatch - early beta (Windows x64)
================================================

🚨 PLEASE READ BEFORE DOWNLOADING (Early Beta & Manage Your Expectations):
This is an early, experimental beta, and I really don't want to overpromise
here! Please only download this if you enjoy testing raw, early software and
fully expect that things might fail or not work on your setup.
* Your mileage WILL vary: While I had a few successful test runs on my own
  discs (South Park S6 USA Blu-ray, Walking Dead S1 German Blu-ray,
  The Americans S1 UK Blu-ray, Castle Rock S1 German Blu-ray), a lot of the
  matching logic (especially the subtitle comparison) is still heavily based
  on theory. Just because it worked on my test rips does NOT mean you will
  automatically get the same results on yours!
* USE AT YOUR OWN RISK - NO GUARANTEES: Even though the renamer has an undo
  journal, DO NOT run this on video files you don't have a backup of (or can't
  easily re-rip). Always test on a backup copy first! No responsibility is
  taken for mangled filenames or broken MKV headers.

CONTENTS
  This archive ships ONE executable directly at the archive root:
    episodematch.exe   console CLI (match, check-env, fetch-tools, ...)
  The "_internal" folder belongs to the executable - keep it in place.
  bin/ and data/ are created/used NEXT TO the executable.
  (The separate EpisodeMatch-GUI-v*.zip ships the Windows-11 GUI; extract
  the two archives into SEPARATE folders - both contain an "_internal"
  directory and would clash when unpacked into the same one.)

FIRST RUN (console)
1. Extract this archive to a permanent folder (e.g. D:\EpisodeMatch).
2. Run:  episodematch.exe fetch-tools
   (downloads ffmpeg, MKVToolNix and SeConv into .\bin - roughly 400 MB)
3. Create data\config.json with your API keys (see the GitHub README):
     - TMDB key (required, free at themoviedb.org)
     - SubDL or OpenSubtitles key (for the Subtitles pillar)
4. Run:  episodematch.exe match "D:\Rips\Show S01" --series "Show" --season 1
   (episodematch.exe check-env shows the tool/OCR/key status first)

Full documentation: https://github.com/escaton80/EpisodeMatch#readme

Status: early public beta. Disc DB + Subtitles pillars are production-ready;
Audio/Whisper, Title-card OCR, LLM arbiter, Visual Verify and Post-Processing
are marked not ready / beta inside the app.

MIT License - https://github.com/escaton80/EpisodeMatch
