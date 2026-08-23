# API: session_loader

Location in repository: src/fastf1_analytics/session_loader.py

Purpose
- Lightweight wrapper around FastF1 data access used by the example plot scripts in tools/plots/. The function load_session(...) centralizes common defaults (cache location, session code handling) so CLI scripts and notebooks can be concise.

Public API
- load_session(year, event, session, *, cache=None, kwargs...)
  - year: integer season (e.g. 2024)
  - event: event name or Grand Prix (e.g. "Monaco", "Italian Grand Prix")
  - session: session code ("R" = Race, "Q" = Qualifying, "S" = Sprint)
  - cache: path to FastF1 cache directory (optional)

Notes
- The returned object is the FastF1 Session object (see fastf1 docs). The loader applies a small layer of convenience but does not replace FastF1's API.
- If you need lower-level control (custom caching, retries), import FastF1 directly in your script.

Example
```python
from fastf1_analytics.session_loader import load_session

session = load_session(2024, "Monaco", "R", cache=".fastf1-cache")
print(session.event["EventName"], session.name)
```