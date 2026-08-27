from __future__ import annotations

from pathlib import Path
from typing import Any

import fastf1


def load_session(
    year: int,
    gp: str,
    session: str,
    *,
    cache: str | None = ".fastf1-cache",
) -> Any:
    """Load a FastF1 session with optional local cache directory.

    Args:
        year: Season year (e.g., 2024).
        gp: Grand Prix name or event string (e.g., 'Monaco' or 'Italian Grand Prix').
        session: Session code: 'R' (race), 'Q' (qualifying), or 'S' (sprint).
        cache: Path to a local FastF1 cache directory. If None or empty, caching is not
            enabled.

    Returns:
        A loaded FastF1 Session object exposing `.event`, `.laps`, `.telemetry`, and
        other FastF1 interfaces.
    """
    if cache:
        cache_path = Path(cache)
        cache_path.mkdir(parents=True, exist_ok=True)  # ensure cache dir exists
        fastf1.Cache.enable_cache(str(cache_path))
    s = fastf1.get_session(year, gp, session)
    s.load()
    return s
