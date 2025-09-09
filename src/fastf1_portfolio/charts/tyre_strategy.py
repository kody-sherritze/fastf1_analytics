from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fastf1_portfolio.plotting import apply_style, get_compound_color, savefig


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
    if isinstance(order, list):
        return [d.upper() for d in order]

    if order == "alpha":
        return sorted({str(d) for d in session.laps["Driver"].unique()})

    # default: official results order if available, else alpha
    try:
        res = session.results.sort_values("Position")
        return [str(x) for x in res["Abbreviation"].tolist()]
    except Exception:
        return sorted({str(d) for d in session.laps["Driver"].unique()})


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
    drivers = _driver_sort_order(session, params.driver_order)

    # Stints per driver: count consecutive laps with same 'Stint' and carry compound label
    stints = (
        laps[["Driver", "Stint", "Compound", "LapNumber"]]
        .groupby(["Driver", "Stint", "Compound"], as_index=False)
        .count()
        .rename(columns={"LapNumber": "Laps"})
        .sort_values(["Driver", "Stint"])
    )

    # Max total laps -> xlim
    total_laps = int(stints.groupby("Driver")["Laps"].sum().max())

    fig, ax = plt.subplots(figsize=(11, 8))
    yticklabels: list[str] = []

    y = 0.0
    for drv in drivers:
        df = stints[stints["Driver"] == drv]
        if df.empty:
            continue
        left = 0
        for _, row in df.iterrows():
            width = int(row["Laps"])
            compound = str(row["Compound"])
            color = get_compound_color(compound)
            ax.barh(
                y,
                width=width,
                left=left,
                height=params.bar_height,
                color=color,
                edgecolor="none",
            )
            if params.annotate_compound and width >= 4:
                ax.text(
                    left + width / 2.0, y, compound.title(), ha="center", va="center", fontsize=8
                )
            left += width

        yticklabels.append(drv)
        y += params.bar_height + params.bar_gap

    ax.set_yticks([i * (params.bar_height + params.bar_gap) for i in range(len(yticklabels))])
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(-params.bar_gap, y - params.bar_gap)
    ax.set_xlim(0, total_laps + 1)
    ax.set_xlabel("Laps →")
    title = params.title or f"{session.event.year} {session.event['EventName']} – Tyre Strategy"
    ax.set_title(title)

    # Clean look
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    if params.show_legend:
        # Simple legend with compounds present
        uniques = [str(c) for c in stints["Compound"].unique().tolist()]
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=get_compound_color(c), label=c.title())
            for c in uniques
        ]
        ax.legend(handles=handles, title="Compound", ncols=5, frameon=False, loc="lower right")

    if out_path is not None:
        savefig(fig, out_path, dpi=params.dpi)

    return fig, ax
