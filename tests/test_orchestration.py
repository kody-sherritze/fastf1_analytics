from types import SimpleNamespace

import fastf1
import pandas as pd
import pytest

from tools.plots import time_in_first


def test_season_lead_time_table_skips_failed_events(monkeypatch) -> None:
    schedule = pd.DataFrame(
        {
            "RoundNumber": [1, 2],
            "EventName": ["Failed Grand Prix", "Working Grand Prix"],
        }
    )
    valid_session = SimpleNamespace(
        laps=pd.DataFrame(
            {
                "Driver": ["VER", "VER"],
                "Position": [1, 2],
                "LapTime": pd.to_timedelta([80, 81], unit="s"),
            }
        ),
        results=pd.DataFrame({"Abbreviation": ["VER"], "TeamName": ["Red Bull Racing"]}),
    )

    monkeypatch.setattr(fastf1, "get_event_schedule", lambda year, include_testing: schedule)

    def fake_load_session(year, event, session, cache):
        if event == "Failed Grand Prix":
            raise ValueError("session unavailable")
        return valid_session

    monkeypatch.setattr(time_in_first, "load_session", fake_load_session)

    result = time_in_first._season_lead_time_table(2024, cache="unused")

    assert result[["Round", "Driver", "TimeLedCum"]].to_dict("records") == [
        {"Round": 2, "Driver": "VER", "TimeLedCum": 1.3333333333333333}
    ]


def test_season_lead_time_table_raises_when_no_data(monkeypatch) -> None:
    schedule = pd.DataFrame({"RoundNumber": [1], "EventName": ["Empty Grand Prix"]})
    monkeypatch.setattr(fastf1, "get_event_schedule", lambda year, include_testing: schedule)
    monkeypatch.setattr(
        time_in_first,
        "load_session",
        lambda year, event, session, cache: SimpleNamespace(
            laps=pd.DataFrame(), results=pd.DataFrame()
        ),
    )

    with pytest.raises(RuntimeError, match="No lead time data"):
        time_in_first._season_lead_time_table(2024, cache="unused")
