# API: session_loader
::: fastf1_analytics.session_loader

**Example**
```python
from fastf1_analytics.session_loader import load_session
session = load_session(2024, "Monaco", "R", cache=".fastf1-cache")
print(session.event["EventName"], session.name)
```