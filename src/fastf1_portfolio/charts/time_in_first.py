"""
Chart for Cumulative Time Spent Leading
---------------------------------------

This module provides a simple line chart that visualises how long each
driver spent in first place across a Formula 1 season.  Times are
accumulated per race and plotted cumulatively by round, resulting in a
chart similar to the Drivers' Championship points graph.  Each
driver’s line is coloured according to their team’s primary colour
using :func:`fastf1_portfolio.plotting.get_team_color`.  A secondary
variant is available which lightens the colours for an alternate look.

The expected input is a pandas DataFrame with one row per driver per
round and the columns ``Driver``, ``TeamName``, ``Round`` and
``TimeLedCum``.  See :mod:`tools.plots.time_in_first` for an example of
how to build this table from FastF1 data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fastf1_portfolio.plotting import apply_style, get_team_color, lighten_color, savefig


@dataclass(frozen=True)
class TimeInFirstParams:
    """Customization knobs for the cumulative time‑in‑first chart.

    Attributes
    ----------
    color_variant:
        Choose between the team's primary colour (``"primary"``) and a
        lightened variant (``"secondary"``).  Secondary colours are
        derived via :func:`fastf1_portfolio.plotting.lighten_color`.

    annotate_last:
        Whether to annotate the final cumulative time for each driver.

    min_total_time:
        Filter out drivers whose season total time spent leading (in
        minutes) is less than or equal to this threshold.  Use this to
        declutter the chart when only a few drivers spent significant
        time at the front.  Defaults to zero, so all drivers are shown.

    linewidth:
        Line thickness for the driver traces.

    dpi:
        Resolution for saving figures via :func:`matplotlib.figure.Figure.savefig`.

    title:
        Optional custom title for the plot.  If ``None``, a default
        title is constructed from the year.
    """

    color_variant: Literal["primary", "secondary"] = "primary"
    annotate_last: bool = True
    min_total_time: float = 0.0
    linewidth: float = 1.8
    dpi: int = 220
    title: str | None = None


def build_time_in_first_chart(
    time_cum: pd.DataFrame,
    *,
    year: int,
    params: TimeInFirstParams = TimeInFirstParams(),
    out_path: str | None = None,
) -> Tuple[Figure, Axes]:
    """Plot cumulative time spent leading by round.

    Parameters
    ----------
    time_cum:
        Long‑form DataFrame with columns ``[Driver, TeamName, Round, TimeLedCum]``.
        Each row represents the cumulative minutes a driver has spent
        leading up to and including a given round.

    year:
        Season year for the title.

    params:
        Visual customisation parameters.  See :class:`TimeInFirstParams`.

    out_path:
        If provided, the plot is saved to this path using
        :func:`fastf1_portfolio.plotting.savefig` with the DPI defined in
        ``params``.

    Returns
    -------
    (Figure, Axes)
        The Matplotlib figure and axes objects.
    """
    apply_style()

    # Filter out drivers below the threshold on their final total
    final_totals = time_cum.sort_values("Round").groupby("Driver", sort=False).tail(1)
    keep = final_totals[final_totals["TimeLedCum"] > params.min_total_time]["Driver"]
    df = time_cum[time_cum["Driver"].isin(keep)].copy()

    # Map each driver to a constant colour based on their latest team
    color_by_driver: Dict[str, str] = {}
    for drv, grp in df.groupby("Driver", sort=False):
        drv_str = str(drv)
        team = str(grp.iloc[-1]["TeamName"] or "")
        color = get_team_color(team)
        if params.color_variant == "secondary":
            color = lighten_color(color, amount=0.25)
        color_by_driver[drv_str] = color

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Draw a line for each driver
    for drv, grp in df.groupby("Driver", sort=False):
        drv_str = str(drv)
        grp = grp.sort_values("Round")
        ax.plot(
            grp["Round"],
            grp["TimeLedCum"],
            label=drv_str,
            linewidth=params.linewidth,
            color=color_by_driver.get(drv_str, "#666666"),
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative time in first (min)")
    title = params.title or f"{year} Time Spent Leading – Cumulative"
    ax.set_title(title)

    # Annotate the final point for each driver with their abbreviation and total minutes
    if params.annotate_last:
        tail = df.sort_values("Round").groupby("Driver", sort=False).tail(1)
        for _, row in tail.iterrows():
            drv_str = str(row["Driver"])
            y_val = float(row["TimeLedCum"])
            # Display one decimal place for minutes
            ax.text(
                float(row["Round"]) + 0.15,
                y_val,
                f"{drv_str} ({y_val:.1f})",
                va="center",
                fontsize=8,
                color=color_by_driver.get(drv_str, "#666666"),
            )

    # Aesthetic tweaks
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.margins(x=0.02)

    # Omit legend: annotations make driver identification clear
    # ax.legend(frameon=False)

    if out_path:
        savefig(fig, out_path, dpi=params.dpi)

    return fig, ax
