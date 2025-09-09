from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fastf1_portfolio.plotting import (
    apply_style,
    get_team_color,
    savefig,
)
from fastf1_portfolio.plotting import lighten_color  # secondary variant helper


@dataclass(frozen=True)
class DriverPointsParams:
    """Customization knobs for the Drivers' Championship line chart."""

    color_variant: Literal["primary", "secondary"] = "primary"
    annotate_last: bool = True
    min_total_points: float = 0.0  # filter drivers whose season total <= this value
    linewidth: float = 1.8
    dpi: int = 220
    title: str | None = None


def build_driver_points_chart(
    points_cum: pd.DataFrame,
    *,
    year: int,
    params: DriverPointsParams = DriverPointsParams(),
    out_path: str | None = None,
) -> Tuple[Figure, Axes]:
    """Plot cumulative Drivers' Championship points by race.

    Args:
        points_cum: DataFrame with columns:
            ['Driver', 'TeamName', 'Round', 'PointsCum'].
        year: Season year (for title and context).
        params: Visual customization.
        out_path: If provided, save the figure there with consistent style.

    Returns:
        (Figure, Axes)
    """
    apply_style()

    # Filter low scorers if requested
    final_totals = points_cum.sort_values("Round").groupby("Driver").tail(1)
    keep = final_totals[final_totals["PointsCum"] > params.min_total_points]["Driver"]
    df = points_cum[points_cum["Driver"].isin(keep)].copy()

    # Determine color per driver (constant across season)
    color_by_driver: dict[str, str] = {}
    for drv, grp in df.groupby("Driver", sort=False):
        drv_str = str(drv)
        team = str(grp.iloc[-1]["TeamName"])  # last seen team
        color = get_team_color(team)
        if params.color_variant == "secondary":
            color = lighten_color(color, amount=0.25)
        color_by_driver[drv_str] = color

    fig, ax = plt.subplots(figsize=(12, 7))

    # One line per driver
    for drv, grp in df.groupby("Driver", sort=False):
        drv_str = str(drv)
        grp = grp.sort_values("Round")
        ax.plot(
            grp["Round"],
            grp["PointsCum"],
            label=drv_str,
            linewidth=params.linewidth,
            color=color_by_driver.get(drv_str, "#666666"),
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative points")
    title = params.title or f"{year} Drivers' Championship – Cumulative points"
    ax.set_title(title)

    # Annotations at the last point per driver
    if params.annotate_last:
        tail = df.sort_values("Round").groupby("Driver").tail(1)
        for _, row in tail.iterrows():
            ax.text(
                float(row["Round"]) + 0.15,
                float(row["PointsCum"]),
                f"{row['Driver']} ({int(row['PointsCum'])})",
                va="center",
                fontsize=8,
                color=color_by_driver.get(str(row["Driver"]), "#666666"),
            )

    # Cosmetic cleanup
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.margins(x=0.02)

    # Legend is too busy with many drivers; rely on annotations
    # ax.legend(ncol=3, frameon=False)

    if out_path:
        savefig(fig, out_path, dpi=params.dpi)

    return fig, ax
