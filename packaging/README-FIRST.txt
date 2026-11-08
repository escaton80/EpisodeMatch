EpisodeMatch v0.1.0 - first beta (Windows x64)
================================================

CONTENTS
  gui/EpisodeMatch.exe      the Windows-11 GUI (recommended)
  cli/episodematch.exe      console CLI (match, check-env, fetch-tools, ...)

FIRST RUN
1. Extract this archive to a permanent folder (e.g. D:\EpisodeMatch).
2. Open a terminal IN THAT FOLDER (the tools look for .\bin and .\data there).
3. Run:  cli\episodematch.exe fetch-tools
   (downloads ffmpeg, MKVToolNix and SeConv into .\bin - roughly 400 MB)
4. Put your API keys into data\config.json (see the GitHub README):
     - TMDB key (required, free at themoviedb.org)
     - SubDL or OpenSubtitles key (for the Subtitles pillar)
5. Run:  gui\EpisodeMatch.exe   (or use the CLI "match" command)

Full documentation: https://github.com/escaton80/EpisodeMatch#readme

Status: first public beta. Disc DB + Subtitles pillars are production-ready;
Audio/Whisper, Title-card OCR, LLM arbiter, Visual Verify and Post-Processing
are marked not ready / beta inside the app.

MIT License - https://github.com/escaton80/EpisodeMatch
