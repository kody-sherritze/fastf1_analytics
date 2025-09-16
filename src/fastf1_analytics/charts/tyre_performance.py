from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


from fastf1_analytics.plotting import (
    apply_style,
    savefig,
    get_compound_color,
    seconds_formatter,
    get_driver_color,  # FastF1-first; team color via driver+session
)

# Track status codes for SC/VSC/yellows to exclude
_DANGER_CODES = {"4", "5", "6", "7", "8"}

# Dry compounds in a stable order
COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD"]


@dataclass
class TyrePerformanceParams:
    title: str | None = None
    min_laps_per_compound: int = 5
    aggregate: Literal["median", "mean"] = "median"
    include_inter_wet: bool = False
    dpi: int = 220


def _clean_race_laps(session: Any) -> pd.DataFrame:
    """Return clean race laps with LapTime in seconds (no in/out laps, no SC/VSC/yellows)."""
    laps_any = session.laps.copy()
    laps: pd.DataFrame = cast(pd.DataFrame, laps_any)

    # drop in/out laps
    for col in ("PitInTime", "PitOutTime"):
        if col in laps.columns:
            laps = laps[laps[col].isna()]
    for col in ("InLap", "OutLap"):
        if col in laps.columns:
            laps = laps[~laps[col].fillna(False)]

    # drop unsafe track status
    if "TrackStatus" in laps.columns:

        def _ok(ts: str | float | int) -> bool:
            s = str(ts) if pd.notna(ts) else ""
            parts = {p.strip() for p in s.split("+") if p.strip()}
            return parts.isdisjoint(_DANGER_CODES)

        laps = laps[laps["TrackStatus"].apply(_ok)]

    # valid times
    laps = laps[laps["LapTime"].notna()].copy()
    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

    # compound labels
    if "Compound" in laps.columns:
        laps["Compound"] = laps["Compound"].fillna("").astype(str).str.upper()
    else:
        laps["Compound"] = ""

    return laps.reset_index(drop=True)


def _per_driver_compound_laptime(
    laps: pd.DataFrame,
    *,
    min_laps_per_compound: int,
    aggregate: Literal["median", "mean"],
    include_inter_wet: bool,
) -> pd.DataFrame:
    """Per-driver-per-compound lap time (seconds), aggregated by driver."""
    valid = (
        set(COMPOUND_ORDER)
        if not include_inter_wet
        else set(COMPOUND_ORDER) | {"INTERMEDIATE", "WET"}
    )
    laps = laps[laps["Compound"].isin(valid)]
    if laps.empty:
        return pd.DataFrame(columns=["Driver", "Compound", "laptime_s"])

    grp = laps.groupby(["Driver", "Compound"])
    counts = grp["LapTime_s"].count().rename("n")
    agg = grp["LapTime_s"].median() if aggregate == "median" else grp["LapTime_s"].mean()
    out = pd.concat([agg.rename("laptime_s"), counts], axis=1).reset_index()
    out = out[out["n"] >= min_laps_per_compound].drop(columns="n")

    # stable order (extend with inter/wet only when requested)
    order = COMPOUND_ORDER + (["INTERMEDIATE", "WET"] if include_inter_wet else [])
    out["Compound"] = pd.Categorical(out["Compound"], categories=order, ordered=True)
    return out.sort_values(["Compound", "Driver"]).reset_index(drop=True)


def build_tyre_performance(
    session: Any,
    *,
    params: TyrePerformanceParams = TyrePerformanceParams(),
    out_path: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot actual lap times per compound (bars = median across drivers; dots = each driver)."""
    apply_style()

    laps = _clean_race_laps(session)
    if laps.empty:
        raise ValueError("No clean race laps found.")

    per_driver = _per_driver_compound_laptime(
        laps,
        min_laps_per_compound=params.min_laps_per_compound,
        aggregate=params.aggregate,
        include_inter_wet=params.include_inter_wet,
    )
    if per_driver.empty:
        raise ValueError("No drivers met the min-laps-per-compound threshold.")

    # Attach a stable team per driver and map to team colors via your helper
    teams = (
        laps.groupby("Driver")["Team"]
        .agg(lambda s: s.dropna().mode().iat[0] if not s.dropna().empty else "")
        .rename("Team")
        .reset_index()
    )
    per_driver = per_driver.merge(teams, on="Driver", how="left")
    per_driver["dot_color"] = per_driver["Driver"].apply(
        lambda t: get_driver_color(t, session=session)
    )

    # Median lap time across drivers per compound (for the bars)
    comp_stats = (
        per_driver.groupby("Compound", observed=True)["laptime_s"]
        .median()
        .rename("median_s")
        .to_frame()
        .reset_index()
        .sort_values("Compound")
    )

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    xcats = comp_stats["Compound"].tolist()
    xs = np.arange(len(xcats))
    medians = comp_stats["median_s"].to_numpy()
    bar_top_by_compound = {c: m for c, m in zip(xcats, medians)}

    # Bars = median lap time across drivers
    bar_colors = [get_compound_color(c) for c in xcats]
    ax.bar(xs, medians, width=0.65, edgecolor="#222", linewidth=0.8, color=bar_colors, alpha=0.9)

    SWARM_WIDTH = 0.18  # max horizontal spread (in x-axis category units)
    EPS_SECONDS = 0.25  # points within this lap-time window are considered overlapping

    x_map = {c: i for i, c in enumerate(xcats)}
    per_driver = per_driver[per_driver["Compound"].isin(xcats)].copy()

    # Build x/y arrays and also keep row indices for annotations
    comp_arr = per_driver["Compound"].map(x_map).to_numpy(dtype=float)
    y_arr = per_driver["laptime_s"].to_numpy()
    colors = per_driver["dot_color"].to_numpy()
    drivers = per_driver["Driver"].to_numpy()

    # Compute horizontal offsets per compound so close-in-y points are spread
    x_offsets = np.zeros_like(y_arr, dtype=float)

    for c_idx in range(len(xcats)):
        mask = comp_arr == c_idx
        if not mask.any():
            continue
        y_grp = y_arr[mask]

        # cluster by lap-time proximity (round onto EPS-sized bins)
        bins = np.round(y_grp / EPS_SECONDS).astype(int)
        x_off_grp = np.zeros_like(y_grp, dtype=float)

        # for each bin (cluster of near-overlapping points), spread them across [-W, +W]
        for b in np.unique(bins):
            idx = np.where(mask)[0][bins == b]  # indices into the full arrays
            n = len(idx)
            if n == 1:
                continue
            x_off_grp[bins == b] = np.linspace(-SWARM_WIDTH, SWARM_WIDTH, n)

        x_offsets[mask] = x_off_grp

    # Final coordinates
    xpts = comp_arr + x_offsets
    ypts = y_arr

    # Plot points (team-colored)
    ax.scatter(xpts, ypts, s=28, alpha=0.95, c=colors, edgecolor="#111", linewidth=0.5)

    compounds = per_driver["Compound"].to_numpy()

    # Annotate each dot with the driver code; black text on Medium/Hard for contrast
    for xi, yi, drv, xo, comp in zip(xpts, ypts, drivers, x_offsets, compounds):
        # inside the bar if the point's y is <= the bar top for its compound
        bar_top = bar_top_by_compound.get(comp, float("inf"))
        inside_bar = yi <= (bar_top - 0.01)
        label_color = "#000000" if (comp in ("MEDIUM", "HARD") and inside_bar) else "#EEEEEE"
        ax.annotate(
            drv,
            (xi, yi),
            xytext=(6 if xo >= 0 else -6, 0),  # push label away from the dot
            textcoords="offset points",
            fontsize=8,
            ha="left" if xo >= 0 else "right",
            va="center",
            color=label_color,
            clip_on=True,
        )
    ax.set_xticks(xs, [c.title() for c in xcats])
    ax.set_ylabel("Lap Time")
    ax.yaxis.set_major_formatter(seconds_formatter())

    title = (
        params.title
        or f"{session.event.year} {session.event['EventName']} – Tyre lap times (clean race laps)"
    )
    ax.set_title(title)

    # Flip y so faster (smaller) is at the top; bottom = worst clean lap - 2s
    fastest = float(min(per_driver["laptime_s"].min(), comp_stats["median_s"].min()))
    worst = float(max(per_driver["laptime_s"].max(), comp_stats["median_s"].max()))
    bottom_bound = worst + 0.5
    top_bound = fastest - 0.5
    ax.set_ylim(bottom_bound, top_bound)

    ax.grid(axis="y", linestyle=":", alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.margins(x=0.03)

    if out_path:
        savefig(fig, out_path, dpi=params.dpi)
    return fig, ax


__all__ = ["TyrePerformanceParams", "build_tyre_performance"]
