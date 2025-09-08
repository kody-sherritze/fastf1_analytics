from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import fastf1.plotting as f1plot


def apply_style(color_scheme: str | None = "fastf1", timedelta_support: bool = True) -> None:
    """Apply a consistent, portfolio-ready Matplotlib style.

    Uses FastF1's built-in styling (dark theme by default) and a few readability tweaks.
    """
    f1plot.setup_mpl(mpl_timedelta_support=timedelta_support, color_scheme=color_scheme)
    plt.rcParams.update(
        {
            # readability
            "figure.dpi": 160,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.25,
            # savefig defaults
            "savefig.bbox": "tight",
        }
    )


def get_team_color(team: str, *, session=None) -> str:
    """Return the FastF1 team color hex for a given team name.
    Wrapper to keep imports local to this module."""
    return f1plot.get_team_color(team, session=session)


def get_driver_color(driver: str, *, session=None) -> str:
    """Return the (team) color for a driver abbreviation in a given session."""
    return f1plot.get_driver_color(driver, session=session)


def fmt_laptime_seconds(sec: float) -> str:
    """Format seconds as M:SS.sss (e.g., 73.456 -> '1:13.456')."""
    m, s = divmod(float(sec), 60.0)
    return f"{int(m)}:{s:06.3f}"


def seconds_formatter() -> FuncFormatter:
    """Matplotlib formatter that renders seconds as M:SS.sss."""
    return FuncFormatter(lambda x, pos: fmt_laptime_seconds(x))
