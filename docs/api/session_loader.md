# Session Loader

`load_session()` is the project's small boundary around FastF1 session retrieval. It enables a local cache when requested, loads the session, and returns the underlying FastF1 `Session` object. It does not replace FastF1's data model or lower-level API.

## Usage

```python
from fastf1_analytics.session_loader import load_session

session = load_session(2025, "Italian Grand Prix", "R", cache=".fastf1-cache")
print(session.event["EventName"], session.name)
```

The `session` argument is a FastF1 session code such as `R` for race, `Q` for qualifying, or `S` for sprint. The event name is passed to FastF1.

::: fastf1_analytics.session_loader.load_session