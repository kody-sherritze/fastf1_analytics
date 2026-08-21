from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from fastf1_analytics.charts.drs_effectiveness import (
    _drs_open_flags,
    _drs_zone_bounds,
    _segments_from_brake_mask,
    _time_from_resampled,
)
from fastf1_analytics.charts.tyre_performance import (
    _clean_race_laps,
    _per_driver_compound_laptime,
)
from fastf1_analytics.charts.tyre_strategy import _driver_sort_order, _eligible_drivers


def test_clean_race_laps_removes_invalid_conditions_and_normalizes_compounds() -> None:
    session = SimpleNamespace(
        laps=pd.DataFrame(
            {
                "PitInTime": [pd.NaT, pd.NaT, pd.Timestamp("2025-01-01"), pd.NaT, pd.NaT],
                "PitOutTime": [pd.NaT] * 5,
                "InLap": [False, True, False, False, False],
                "OutLap": [False] * 5,
                "TrackStatus": ["1", "4", "1+5", "1", "1"],
                "LapTime": pd.to_timedelta(
                    [
                        "0 days 00:01:20",
                        "0 days 00:01:21",
                        "0 days 00:01:22",
                        None,
                        "0 days 00:01:23",
                    ]
                ),
                "Compound": ["soft", "MEDIUM", "HARD", "SOFT", None],
            }
        )
    )

    cleaned = _clean_race_laps(session)

    assert cleaned["Compound"].tolist() == ["SOFT", ""]
    assert cleaned["LapTime_s"].tolist() == [80.0, 83.0]


def test_per_driver_compound_laptime_aggregates_filters_and_orders() -> None:
    laps = pd.DataFrame(
        {
            "Driver": ["VER"] * 3 + ["HAM"] * 2 + ["VER"] * 2,
            "Compound": ["SOFT"] * 3 + ["MEDIUM"] * 2 + ["WET"] * 2,
            "LapTime_s": [80.0, 82.0, 100.0, 90.0, 94.0, 70.0, 72.0],
        }
    )

    dry = _per_driver_compound_laptime(
        laps, min_laps_per_compound=2, aggregate="median", include_inter_wet=False
    )
    all_compounds = _per_driver_compound_laptime(
        laps, min_laps_per_compound=2, aggregate="mean", include_inter_wet=True
    )

    assert dry[["Driver", "Compound"]].to_dict("records") == [
        {"Driver": "VER", "Compound": "SOFT"},
        {"Driver": "HAM", "Compound": "MEDIUM"},
    ]
    assert dry.loc[dry["Compound"] == "SOFT", "laptime_s"].iat[0] == 82.0
    assert all_compounds["Compound"].tolist() == ["SOFT", "MEDIUM", "WET"]


def test_eligible_and_sorted_drivers_exclude_nonstarters() -> None:
    session = SimpleNamespace(
        laps=pd.DataFrame(
            {
                "Driver": ["VER", "VER", "HAM", "BOT"],
                "LapNumber": [1, 2, 1, 0],
            }
        ),
        results=pd.DataFrame(
            {
                "Abbreviation": ["VER", "HAM", "BOT", "NOR"],
                "ClassifiedPosition": ["1", "2", "F", "3"],
                "Position": [1, 2, 3, 4],
            }
        ),
    )

    assert _eligible_drivers(session) == ["VER", "HAM"]
    assert _driver_sort_order(session, "alpha") == ["HAM", "VER"]
    assert _driver_sort_order(session, ["ham", "VER", "NOR"]) == ["HAM", "VER"]


def test_drs_helpers_classify_zones_and_integrate_time() -> None:
    telemetry = pd.DataFrame({"Distance": [0, 10, 20, 30, 40], "DRS": [0, 12, 14, 0, 12]})

    assert _drs_open_flags(telemetry).tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]
    assert _drs_zone_bounds(telemetry) == (10.0, 20.0)
    assert _drs_zone_bounds(pd.DataFrame({"Distance": [0, 1], "DRS": [0, 0]})) is None
    assert _time_from_resampled(1000.0, np.full(5, 100.0)) == 36.0
    assert np.isnan(_time_from_resampled(0.0, np.array([100.0, 100.0])))


def test_brake_segments_require_minimum_length() -> None:
    mask = np.array([True, False, False, True, False, False, False, False])
    assert _segments_from_brake_mask(mask, min_len=3) == [(4, 7)]
    assert _segments_from_brake_mask(np.array([], dtype=bool)) == []
