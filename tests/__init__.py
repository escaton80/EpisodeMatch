"""EpisodeMatch test suite.

Making this a regular package keeps `from tests.conftest import ...` working
in every environment (CI runs `pytest` directly, without `python -m pytest`
placing the repository root on sys.path).
"""
