from types import SimpleNamespace

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from fastf1_analytics.charts import driver_points, pace_consistency, time_in_first


def test_driver_points_builder_filters_threshold_and_annotations(monkeypatch) -> None:
    monkeypatch.setattr(driver_points, "apply_style", lambda: None)
    monkeypatch.setattr(driver_points, "get_team_color", lambda team: "#123456")
    data = pd.DataFrame(
        {
            "Driver": ["VER", "VER", "HAM", "HAM"],
            "TeamName": ["Red Bull", "Red Bull", "Mercedes", "Mercedes"],
            "Round": [2, 1, 1, 2],
            "PointsCum": [20.0, 10.0, 0.0, 0.0],
        }
    )

    fig, ax = driver_points.build_driver_points_chart(
        data,
        year=2024,
        params=driver_points.DriverPointsParams(min_total_points=0),
    )
    try:
        assert ax.get_title() == "2024 Drivers' Championship - Cumulative points"
        assert ax.get_xlabel() == "Round"
        assert ax.get_ylabel() == "Cumulative points"
        assert len(ax.lines) == 1
        assert ax.lines[0].get_label() == "VER"
        assert [text.get_text() for text in ax.texts] == ["VER (20)"]
    finally:
        plt.close(fig)


def test_time_in_first_builder_supports_custom_title_and_no_annotations(monkeypatch) -> None:
    monkeypatch.setattr(time_in_first, "apply_style", lambda: None)
    monkeypatch.setattr(time_in_first, "get_team_color", lambda team: "#123456")
    data = pd.DataFrame(
        {
            "Driver": ["VER", "VER", "HAM", "HAM"],
            "TeamName": ["Red Bull", "Red Bull", "Mercedes", "Mercedes"],
            "Round": [1, 2, 1, 2],
            "TimeLedCum": [5.0, 15.0, 0.0, 0.0],
        }
    )

    fig, ax = time_in_first.build_time_in_first_chart(
        data,
        year=2024,
        params=time_in_first.TimeInFirstParams(
            annotate_last=False,
            min_total_time=0,
            title="Leading time",
        ),
    )
    try:
        assert ax.get_title() == "Leading time"
        assert ax.get_ylabel() == "Cumulative time in first (min)"
        assert len(ax.lines) == 1
        assert ax.lines[0].get_label() == "VER"
        assert not ax.texts
    finally:
        plt.close(fig)


def test_chart_builders_accept_default_parameter_objects(monkeypatch) -> None:
    monkeypatch.setattr(driver_points, "apply_style", lambda: None)
    monkeypatch.setattr(driver_points, "get_team_color", lambda team: "#123456")
    session = SimpleNamespace()
    assert session is not None
    data = pd.DataFrame(
        {
            "Driver": ["VER"],
            "TeamName": ["Red Bull"],
            "Round": [1],
            "PointsCum": [1.0],
        }
    )
    fig, _ = driver_points.build_driver_points_chart(data, year=2024)
    plt.close(fig)


def test_pace_consistency_builder_uses_shared_baseline_and_stint_ticks(monkeypatch) -> None:
    monkeypatch.setattr(pace_consistency, "apply_style", lambda: None)
    monkeypatch.setattr(pace_consistency, "get_driver_color", lambda driver, session: "#123456")
    monkeypatch.setattr(pace_consistency, "get_compound_color", lambda compound: "#abcdef")
    session = SimpleNamespace(
        event=pd.Series({"EventName": "Test Grand Prix", "year": 2024}),
        laps=pd.DataFrame(
            {
                "Driver": ["VER", "VER", "VER", "HAM", "HAM", "HAM"],
                "LapNumber": [1, 2, 3, 1, 2, 3],
                "LapTime": pd.to_timedelta([80, 82, 84, 90, 92, 94], unit="s"),
                "TrackStatus": ["1"] * 6,
                "PitInTime": [pd.NaT] * 6,
                "PitOutTime": [pd.NaT] * 6,
                "InLap": [False] * 6,
                "OutLap": [False] * 6,
                "Stint": [1, 1, 2, 1, 1, 2],
                "Compound": ["SOFT", "SOFT", "MEDIUM", "HARD", "HARD", "SOFT"],
            }
        ),
        results=pd.DataFrame({"Abbreviation": ["VER", "HAM"]}),
    )

    fig, ax = pace_consistency.build_pace_consistency(
        session, params=pace_consistency.PaceConsistencyParams(drivers=2)
    )
    try:
        assert ax.get_xlabel() == "Race lap"
        assert ax.get_ylabel() == "Lap time delta from selected-driver average median (s)"
        assert len(ax.lines) == 3
        assert list(ax.lines[0].get_xdata()) == [2, 3]
        assert list(ax.lines[1].get_xdata()) == [2, 3]
        assert list(ax.lines[-1].get_ydata()) == [0, 0]
        assert ax.collections
    finally:
        plt.close(fig)


def test_pace_consistency_builder_supports_single_driver(monkeypatch) -> None:
    monkeypatch.setattr(pace_consistency, "apply_style", lambda: None)
    monkeypatch.setattr(pace_consistency, "get_driver_color", lambda driver, session: "#123456")
    monkeypatch.setattr(pace_consistency, "get_compound_color", lambda compound: "#abcdef")
    session = SimpleNamespace(
        event=pd.Series({"EventName": "Test Grand Prix", "year": 2024}),
        laps=pd.DataFrame(
            {
                "Driver": ["VER", "VER", "HAM"],
                "LapNumber": [1, 2, 1],
                "LapTime": pd.to_timedelta([80, 82, 90], unit="s"),
                "TrackStatus": ["1"] * 3,
                "PitInTime": [pd.NaT] * 3,
                "PitOutTime": [pd.NaT] * 3,
                "InLap": [False] * 3,
                "OutLap": [False] * 3,
                "Stint": [1, 1, 1],
                "Compound": ["SOFT"] * 3,
            }
        ),
        results=pd.DataFrame({"Abbreviation": ["VER", "HAM"]}),
    )

    fig, ax = pace_consistency.build_pace_consistency(
        session, params=pace_consistency.PaceConsistencyParams(driver="ver")
    )
    try:
        assert [line.get_label() for line in ax.lines[:1]] == ["VER"]
    finally:
        plt.close(fig)
