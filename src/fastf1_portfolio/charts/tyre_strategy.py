from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from fastf1_portfolio.plotting import apply_style, savefig, get_compound_color


def _eligible_drivers(session: Any) -> list[str]:
    """Drivers who actually started: exclude 'F' in ClassifiedPosition and zero-lap entries."""
    laps = getattr(session, "laps", None)
    results = getattr(session, "results", None)
    if laps is None:
        return []
    # Lap counts per driver (by abbreviation)
    lap_counts = laps.groupby("Driver")["LapNumber"].count()
    if results is None or len(results) == 0:
        return [str(d) for d, n in lap_counts.items() if int(n) > 0]
    keep: list[str] = []
    res = results.copy()
    # Normalize 'ClassifiedPosition' to string for comparisons
    res["ClassifiedPosition"] = res["ClassifiedPosition"].astype(str).str.strip().str.upper()
    for _, row in res.iterrows():
        drv = str(row.get("Abbreviation", "")).strip()
        if not drv:
            continue
        classified = str(row.get("ClassifiedPosition", "")).strip().upper()
        if classified == "F":
            # Failed to start/qualify: omit
            continue
        laps_done = int(lap_counts.get(drv, 0))
        if laps_done == 0:
            # Did not take the start: omit
            continue
        keep.append(drv)
    return keep


@dataclass(frozen=True)
class TyreStrategyParams:
    """Customization knobs for the tyre strategy chart."""

    bar_height: float = 0.6
    bar_gap: float = 0.35
    annotate_compound: bool = True
    dpi: int = 220
    title: str | None = None
    driver_order: str | list[str] = "results"  # "results" | "alpha" | explicit list
    show_legend: bool = False


def _driver_sort_order(session: Any, order: str | list[str]) -> list[str]:
    """Return driver order filtered to eligible starters."""
    eligible = set(_eligible_drivers(session))
    if isinstance(order, list):
        return [d.upper() for d in order if d.upper() in eligible]
    if order == "alpha":
        return sorted(eligible)
    # default: official result order (Position) filtered by eligibility
    try:
        res = session.results.sort_values("Position")
        ordered = [str(x) for x in res["Abbreviation"].tolist()]
        return [d for d in ordered if d in eligible]
    except Exception:
        return sorted(eligible)


def build_tyre_strategy(
    session: Any,
    *,
    params: TyreStrategyParams = TyreStrategyParams(),
    out_path: str | Path | None = None,
) -> Tuple[Figure, Axes]:
    """Plot tyre strategy bars by stint for each driver in a race session.

    Colors reflect Pirelli compound (Soft/Medium/Hard/Inter/Wet). The x-axis is laps.
    """
    apply_style()
    laps = session.laps
    # Compute total race laps for x-axis max
    try:
        total_laps = int(laps["LapNumber"].max())
    except Exception:
        total_laps = int((laps[["Driver", "LapNumber"]].groupby("Driver")["LapNumber"].max()).max())

    # Stints per driver: count consecutive laps with same 'Stint' and carry compound label
    stints = (
        laps[["Driver", "Stint", "Compound", "LapNumber"]]
        .groupby(["Driver", "Stint", "Compound"], as_index=False)
        .count()
        .rename(columns={"LapNumber": "Laps"})
        .sort_values(["Driver", "Stint"])
    )

    # Use the caller's requested order ("results" | "alpha" | explicit list)
    drivers = _driver_sort_order(session, params.driver_order)
    # Restrict to drivers who actually started
    stints = stints[stints["Driver"].isin(drivers)]
    # Max total laps -> xlim

    fig, ax = plt.subplots(figsize=(10, 6))
    y = 0.0
    yticklabels: list[str] = []
    for drv in drivers:
        df = stints[stints["Driver"] == drv]
        if df.empty:
            continue
        start = 0
        for _, row in df.iterrows():
            width = int(row["Laps"])
            compound = str(row["Compound"])
            color = get_compound_color(compound)
            ax.barh(
                y,
                width=width,
                left=start,
                height=params.bar_height,
                color=color,
                edgecolor="none",
            )
            # Annotate compound label with color rule:
            # Medium/Hard -> black text; others -> white
            if width >= 4:
                label_color = (
                    "#000000" if compound.strip().upper() in ("MEDIUM", "HARD") else "#FFFFFF"
                )
                ax.text(
                    start + width / 2.0,
                    y,
                    compound.title(),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=label_color,
                )
            start += width

        # After all stints for this driver, display their total laps completed
        # unless they completed the full race distance (then omit).
        # If there's room, place just to the right of the final bar; otherwise
        # place slightly inside the axis so it's always visible when a driver
        # completes the full distance (driver_laps == total_laps).
        driver_laps = int(start)
        if driver_laps != total_laps:
            outside_x = driver_laps + 0.3
            inside_x = max(total_laps - 0.3, 0)
            if outside_x <= (total_laps - 0.3):
                label_x = outside_x
                ha = "left"
            else:
                label_x = inside_x
                ha = "right"
            ax.text(
                label_x,
                y,
                f"{driver_laps}",
                ha=ha,
                va="center",
                fontsize=8,
                color="#FFFFFF",  # high-contrast for FastF1 dark style
                clip_on=False,
            )

        yticklabels.append(drv)
        y += params.bar_height + params.bar_gap

    ax.set_yticks([i * (params.bar_height + params.bar_gap) for i in range(len(yticklabels))])
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(-params.bar_gap, y - params.bar_gap)
    # X-axis: match the total number of laps exactly
    ax.set_xlim(0, total_laps)
    ax.set_xlabel("Laps →")
    title = params.title or f"{session.event.year} {session.event['EventName']} – Tyre Strategy"
    ax.set_title(title)

    # Clean look
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    # P1 at the top (invert Y)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    if params.show_legend:
        # Simple legend with compounds present
        uniques = [str(c) for c in stints["Compound"].unique().tolist()]
        handles = [
            Rectangle((0, 0), 1, 1, color=get_compound_color(c), label=c.title()) for c in uniques
        ]
        ax.legend(handles=handles, title="Compound", ncols=5, frameon=False, loc="lower right")

    if out_path is not None:
        savefig(fig, out_path, dpi=params.dpi)

    return fig, ax
