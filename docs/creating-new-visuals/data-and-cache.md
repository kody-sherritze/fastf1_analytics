# Loading Data and Managing Cache

FastF1 owns the underlying event, lap, telemetry, and result data. FastF1 Analytics provides a predictable entry point for obtaining a loaded session.

## The session loader

`fastf1_analytics.session_loader.load_session(year, gp, session, cache=...)` creates the cache directory when needed, enables FastF1's cache, loads the requested session, and returns the FastF1 `Session` object.

```python
from fastf1_analytics.session_loader import load_session

session = load_session(
    2025,
    "Italian Grand Prix",
    "R",
    cache=".fastf1-cache",
)

print(session.event["EventName"])
print(session.laps.columns)
```

The returned object exposes the FastF1 interfaces used by the analyses, including `.event`, `.laps`, `.results`, and telemetry accessors. The `session` argument is a FastF1 session code such as `R` for race, `Q` for qualifying, or `S` for sprint. The event string is passed to FastF1, so use an event name FastF1 recognizes.

## Cache behavior

The plot scripts default to `.fastf1-cache`. The first run can download data; later runs reuse locally cached responses. Passing `None` or an empty string disables this wrapper's cache setup. Cache data is an execution aid, not a published output.

## Loading versus analysis

Keep session retrieval at the boundary of the workflow. Once a script has a session, the analysis should explicitly select columns, laps, drivers, compounds, or telemetry fields and make its assumptions visible. This keeps the data source easy to test and the chart builder easier to reuse.
