from types import SimpleNamespace

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from fastf1_analytics.charts import driver_points, time_in_first


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
