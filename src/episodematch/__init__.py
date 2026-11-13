"""EpisodeMatch — identify ripped Blu-ray/DVD .mkv episode files with near-100% certainty.

EpisodeMatch maps anonymous MakeMKV output (``title_t00.mkv``, ``title_t01.mkv``, ...) to the
exact season/episode (``SxxExx - Episode Title``) using five independent evidence pillars
combined into a weighted confidence score.
"""

from __future__ import annotations

__all__ = ["__version__", "__app_name__"]

__version__ = "0.11.0"
__app_name__ = "EpisodeMatch"
