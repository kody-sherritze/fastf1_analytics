from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fastf1_analytics.plotting import (
    apply_style,
    get_compound_color,
    get_driver_color,
    savefig,
)
from fastf1_analytics.utils import clean_race_laps


@dataclass
class PaceConsistencyParams:
    """Options for the multi-driver race pace consistency chart."""

    title: str | None = None
    drivers: int = 10
    driver: str | None = None
    dpi: int = 220


def average_median_lap_time(laps: pd.DataFrame, drivers: list[str] | None = None) -> float:
    """Return the average of each selected driver's median lap time."""
    if drivers is not None:
        laps = laps[laps["Driver"].isin(drivers)]
    medians = laps.groupby("Driver")["LapTime_s"].median()
    if medians.empty:
        raise ValueError("No driver lap times available for the average median.")
    return float(medians.mean())


def stint_boundaries(session: Any, drivers: list[str] | None = None) -> pd.DataFrame:
    """Return each selected driver's stint start/end laps and tyre compounds."""
    laps = session.laps.copy()
    required = {"Driver", "Stint", "LapNumber", "Compound"}
    missing = required.difference(laps.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Session laps are missing required stint columns: {missing_columns}")
    if drivers is not None:
        laps = laps[laps["Driver"].isin(drivers)]
    if laps.empty:
        return pd.DataFrame(columns=["Driver", "Stint", "start_lap", "end_lap", "Compound"])
    boundaries = (
        laps.groupby(["Driver", "Stint"], dropna=False)
        .agg(
            start_lap=("LapNumber", "min"),
            end_lap=("LapNumber", "max"),
            Compound=("Compound", "first"),
        )
        .reset_index()
    )
    boundaries["Compound"] = boundaries["Compound"].fillna("").astype(str).str.upper()
    return boundaries.sort_values(["Driver", "start_lap"]).reset_index(drop=True)


def _driver_order(session: Any, drivers: int, driver: str | None = None) -> list[str]:
    if drivers < 1:
        raise ValueError("drivers must be at least 1")
    available = list(session.laps["Driver"].dropna().drop_duplicates())
    results = getattr(session, "results", None)
    if isinstance(results, pd.DataFrame) and "Abbreviation" in results.columns:
        result_order = results["Abbreviation"].dropna().tolist()
        available_set = set(available)
        available = [driver for driver in result_order if driver in available_set]
        available.extend(
            driver
            for driver in session.laps["Driver"].dropna().drop_duplicates()
            if driver not in available
        )
    if driver is not None:
        normalized_driver = driver.strip().upper()
        if normalized_driver not in available:
            raise ValueError(f"Driver {normalized_driver} was not found in the session.")
        return [normalized_driver]
    return available[:drivers]


def _pace_data(laps: pd.DataFrame, drivers: list[str]) -> pd.DataFrame:
    selected = laps[laps["Driver"].isin(drivers)].copy()
    if selected.empty:
        raise ValueError("No clean race laps found for the selected drivers.")
    baseline = average_median_lap_time(selected, drivers)
    selected["PaceDelta_s"] = selected["LapTime_s"] - baseline
    return selected.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)


def build_pace_consistency(
    session: Any,
    *,
    params: PaceConsistencyParams | None = None,
    out_path: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot clean lap-time deltas from the selected drivers' average median pace."""
    if params is None:
        params = PaceConsistencyParams()
    apply_style()
    laps = clean_race_laps(session)
    laps = laps[laps["LapNumber"] > 1].copy()
    drivers = _driver_order(session, params.drivers, params.driver)
    data = _pace_data(laps, drivers)
    plotted_drivers = [driver for driver in drivers if driver in set(data["Driver"])]
    if not plotted_drivers:
        raise ValueError("No selected drivers have clean race laps.")

    fig, ax = plt.subplots(figsize=(11, 6))
    for driver in plotted_drivers:
        driver_laps = data[data["Driver"] == driver]
        ax.plot(
            driver_laps["LapNumber"],
            driver_laps["PaceDelta_s"],
            label=driver,
            color=get_driver_color(driver, session=session),
            linewidth=1.5,
            alpha=0.9,
        )

    ax.axhline(0, color="#222222", linewidth=1.0, linestyle="--", alpha=0.8)
    boundaries = stint_boundaries(session, plotted_drivers)
    ymin, ymax = ax.get_ylim()
    tick_length = max((ymax - ymin) * 0.025, 0.05)
    for boundary in boundaries.itertuples(index=False):
        if boundary.Driver not in plotted_drivers:
            continue
        color = get_compound_color(boundary.Compound)
        for lap_number in {boundary.start_lap, boundary.end_lap}:
            driver_rows = data[
                (data["Driver"] == boundary.Driver) & (data["LapNumber"] == lap_number)
            ]
            if driver_rows.empty:
                continue
            y = float(driver_rows["PaceDelta_s"].iat[0])
            ax.vlines(
                lap_number, y - tick_length, y + tick_length, color=color, linewidth=2.0, alpha=0.95
            )

    title = (
        params.title or f"{session.event.year} {session.event['EventName']} - Race pace consistency"
    )
    ax.set_title(title)
    ax.set_xlabel("Race lap")
    ax.set_ylabel("Lap time delta from selected-driver average median (s)")
    ax.legend(title="Driver", ncol=2)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.margins(x=0.02)
    if out_path:
        savefig(fig, out_path, dpi=params.dpi)
    return fig, ax


__all__ = [
    "PaceConsistencyParams",
    "average_median_lap_time",
    "build_pace_consistency",
    "stint_boundaries",
]
