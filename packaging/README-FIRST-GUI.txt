EpisodeMatch - early beta (Windows x64) - GUI
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
  EpisodeMatch.exe   the Windows-11 GUI (this archive)
  _internal/         belongs to the executable - keep it in place
  bin/  data/        created/used NEXT TO the executable
  Extract the GUI and CLI archives into SEPARATE folders (both contain an
  "_internal" directory and would clash when unpacked into the same one).

FIRST RUN - everything is configured inside the app (Settings page)
1. Extract this archive to a permanent folder (e.g. D:\EpisodeMatch).
2. Start EpisodeMatch.exe and open the SETTINGS page:
     - API keys: TMDB key (required, free at themoviedb.org) and a SubDL or
       OpenSubtitles key (for the Subtitles pillar). They are saved to
       data\config.json automatically.
     - Tools: click "Fetch missing tools" - ffmpeg, MKVToolNix and SeConv
       are downloaded into .\bin (roughly 400 MB). The tool list shows what
       is still missing; "Rescan" refreshes it.
3. Go to WORKSPACE: search and select the series, set the season, choose the
   rip folder (optionally paste the TheDiscDB disc URL), press "Scan folder".
4. Review the results and evidence, then "Apply rename" (with undo journal)
   and/or "Write MKV titles".

Full documentation: https://github.com/escaton80/EpisodeMatch#readme

Status: early public beta. Disc DB + Subtitles + Title-card OCR pillars are
ready for testing - the OCR pillar assigns 70-90% of episodes when the show
displays its episode titles on screen (a show that never shows titles cannot
be told apart by it and relies on the DiscDB/Subtitle pillars).
Audio/Whisper, LLM arbiter, Visual Verify and Post-Processing
are marked not ready / beta inside the app.

MIT License - https://github.com/escaton80/EpisodeMatch
