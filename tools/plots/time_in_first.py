"""
CLI script for cumulative time-in-first charts.

This module provides a command line interface to generate a season-long
plot showing how much time each driver spent in first place.  The
script computes a long-form DataFrame of cumulative minutes led per
driver per round, delegates rendering to
``fastf1_analytics.charts.time_in_first.build_time_in_first_chart`` and
writes both a PNG image and a YAML sidecar into the gallery assets
folder.  The YAML metadata is used by the MkDocs gallery generator to
build cards for the web site.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import fastf1
import pandas as pd

from fastf1_analytics.charts.time_in_first import (
    TimeInFirstParams,
    build_time_in_first_chart,
)
from fastf1_analytics.session_loader import load_session
from tools.utils import get_output_paths, write_plot_metadata

logger = logging.getLogger(__name__ + ".time_in_first")


def _season_lead_time_table(year: int, cache: str) -> pd.DataFrame:
    """Build a DataFrame of cumulative minutes spent leading per driver per round.

    Parameters
    ----------
    year:
        Season year.

    cache:
        Path to the FastF1 cache directory.  Pass through to
        :func:`fastf1_analytics.session_loader.load_session`.

    Returns
    -------
    pd.DataFrame
        Table with columns ``[Round, EventName, Driver, TeamName, TimeLedCum]``.
        Each row represents the cumulative minutes a driver has spent in
        first place up to and including that round.  Drivers who never
        led a lap will still appear with zero values once they enter the
        cumulative dictionary.
    """
    # Get the official event schedule (exclude testing sessions)
    sched = fastf1.get_event_schedule(year, include_testing=False)
    rounds: list[tuple[int, str]] = [
        (int(row["RoundNumber"]), str(row["EventName"]))
        for _, row in sched.sort_values("RoundNumber").iterrows()
        if str(row.get("EventName", "")).strip()
    ]

    rows: list[dict[str, Any]] = []
    # Running tally of total minutes led per driver
    by_driver_total: dict[str, float] = {}

    for rnd, event in rounds:
        # Load race session; if session fails (e.g. cancelled event), skip
        try:
            race = load_session(year, event, "R", cache=cache)
        except (AttributeError, OSError, KeyError, TypeError, ValueError) as exc:
            logger.debug(
                "Skipping round %s (%s) because race session couldn't be loaded: %s",
                rnd,
                event,
                exc,
                exc_info=True,
            )
            continue

        laps = getattr(race, "laps", None)
        results = getattr(race, "results", None)

        time_by_driver: dict[str, float] = {}
        if laps is not None and not laps.empty:
            # Ensure LapTime is a timedelta; convert to minutes
            # Some versions of FastF1 already provide timedelta; others provide
            # a pandas Timedelta dtype; in both cases to_timedelta is safe.
            # Use errors='coerce' to handle missing or invalid values.
            laps = laps.copy()
            try:
                lap_times = pd.to_timedelta(laps["LapTime"], errors="coerce")
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug(
                    "LapTime conversion failed for round %s (%s): %s",
                    rnd,
                    event,
                    exc,
                    exc_info=True,
                )
                # If LapTime column is missing or conversion fails, fill zeros so chart logic can continue
                lap_times = pd.Series([pd.Timedelta(0) for _ in range(len(laps))])
            laps["LapTime_timedelta"] = lap_times

            # Filter to laps where the driver was in first position
            if "Position" in laps.columns:
                lead_laps = laps[laps["Position"].astype(str) == "1"].copy()
                # numeric fallback if the string comparison found nothing
                if lead_laps.empty:
                    pos_num = pd.to_numeric(laps["Position"], errors="coerce")
                    lead_laps = laps[pos_num == 1].copy()
                for drv, grp in lead_laps.groupby("Driver"):
                    # Sum lap times in minutes
                    t_total = grp["LapTime_timedelta"].dropna().dt.total_seconds().sum() / 60.0
                    time_by_driver[str(drv)] = time_by_driver.get(str(drv), 0.0) + float(t_total)

        # Update cumulative totals
        for drv, t in time_by_driver.items():
            by_driver_total[drv] = by_driver_total.get(drv, 0.0) + t

        # Snapshot totals after this round
        for drv, total in by_driver_total.items():
            # Determine team name for colour mapping: use latest race results
            team = ""
            if results is not None:
                try:
                    rows_matching = results[results["Abbreviation"] == drv]
                    if len(rows_matching) > 0:
                        team = str(rows_matching.iloc[-1].get("TeamName", ""))
                except (KeyError, IndexError, TypeError) as exc:
                    logger.debug(
                        "Couldn't determine TeamName for driver %s in round %s (%s): %s",
                        drv,
                        rnd,
                        event,
                        exc,
                        exc_info=True,
                    )

            rows.append(
                {
                    "Round": rnd,
                    "EventName": event,
                    "Driver": drv,
                    "TeamName": team,
                    "TimeLedCum": total,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No lead time data found for the requested season.")
    return df


def _slug(year: int) -> str:
    return f"{year}-driver-time-in-first"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate cumulative time‑in‑first chart for a given season and write PNG+YAML sidecar."
    )
    ap.add_argument("--year", type=int, required=True, help="Season year")
    ap.add_argument("--cache", default=".fastf1-cache", help="FastF1 cache directory")
    ap.add_argument(
        "--color-variant",
        choices=["primary", "secondary"],
        default="primary",
        help="Use team primary or secondary (lightened) colours",
    )
    ap.add_argument(
        "--min-total-time",
        type=float,
        default=0.0,
        help="Hide drivers whose season total minutes led are ≤ this threshold",
    )
    ap.add_argument("--title", default=None, help="Optional custom title for the plot")
    ap.add_argument("--dpi", type=int, default=220, help="Figure DPI")
    ap.add_argument(
        "--outdir",
        default="docs/assets/gallery",
        help="Directory where PNG and YAML sidecar will be written",
    )
    args = ap.parse_args()

    # Build the long‑form table of cumulative lead times
    time_cum = _season_lead_time_table(args.year, cache=args.cache)

    slug = _slug(args.year)
    png, yml = get_output_paths(args.outdir, slug)

    params = TimeInFirstParams(
        color_variant=args.color_variant,
        annotate_last=True,
        min_total_time=args.min_total_time,
        dpi=args.dpi,
        title=args.title,
    )

    build_time_in_first_chart(time_cum, year=args.year, params=params, out_path=str(png))

    meta = {
        "title": params.title or f"{args.year} Time Spent Leading ‑ Cumulative",
        "subtitle": "Cumulative minutes led by race (lines per driver)",
        "image": f"assets/gallery/{png.name}",
        "code_path": "tools/plots/time_in_first.py",
        "function": "fastf1_analytics.charts.time_in_first.build_time_in_first_chart",
        "params": {
            "year": args.year,
            "color_variant": params.color_variant,
            "min_total_time": params.min_total_time,
            "dpi": params.dpi,
        },
        "tags": ["season", "drivers", "lead", "time"],
    }
    write_plot_metadata(yml, meta)
    logger.info("Wrote %s and %s", png, yml)
    print(f"Wrote {png} and {yml}")


if __name__ == "__main__":
    main()
