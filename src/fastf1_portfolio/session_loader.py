from __future__ import annotations
import fastf1
from pathlib import Path


def load_session(year: int, gp: str, session: str, *, cache: str | None = ".fastf1"):
    """Load a FastF1 session with optional local cache directory."""
    if cache:
        cache_path = Path(cache)
        cache_path.mkdir(parents=True, exist_ok=True)  # ensure cache dir exists
        fastf1.Cache.enable_cache(str(cache_path))
    s = fastf1.get_session(year, gp, session)
    s.load()
    return s
