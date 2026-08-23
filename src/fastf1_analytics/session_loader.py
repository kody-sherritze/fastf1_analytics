from __future__ import annotations

from pathlib import Path
from typing import Any

import fastf1


def load_session(
    year: int,
    gp: str,
    session: str,
    *,
    cache: str | None = ".fastf1",
) -> Any:
    """Load a FastF1 session with optional local cache directory.

    Parameters
    ----------
    year : int
        Season year (e.g., 2024).
    gp : str
        Grand Prix name or event string (e.g., 'Monaco' or 'Italian Grand Prix').
    session : str
        Session code: 'R' (race), 'Q' (qualifying), 'S' (sprint).
    cache : str | None, optional
        Path to a local FastF1 cache directory. If None or empty, caching is not enabled.

    Returns
    -------
    Any
        A FastF1 Session object with its data loaded. The returned object behaves like
        the FastF1 Session (see FastF1 docs) and exposes .event, .laps, .telemetry, etc.

    Example
    -------
    >>> from fastf1_analytics.session_loader import load_session
    >>> s = load_session(2024, 'Monaco', 'R', cache='.fastf1-cache')
    >>> print(s.event['EventName'], s.name)
    """
    if cache:
        cache_path = Path(cache)
        cache_path.mkdir(parents=True, exist_ok=True)  # ensure cache dir exists
        fastf1.Cache.enable_cache(str(cache_path))
    s = fastf1.get_session(year, gp, session)
    s.load()
    return s
