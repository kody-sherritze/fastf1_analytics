from __future__ import annotations
from typing import cast
from pathlib import Path
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import fastf1.plotting as f1plot  # type: ignore[import-untyped]


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


_COMPOUND_FALLBACK = {
    "SOFT": "#EA3223",  # red
    "MEDIUM": "#F7E200",  # yellow
    "HARD": "#F0F0F0",  # white
    "INTERMEDIATE": "#43AC2D",  # green
    "INTER": "#43AC2D",
    "WET": "#00A3E0",  # blue
    "FULL WET": "#00A3E0",
}


def get_compound_color(compound: str) -> str:
    """Return hex color for a tyre compound (FastF1 or fallback)."""
    c = compound.strip().upper()
    # Prefer FastF1 if available
    try:
        # FastF1 exposes a mapping in plotting; normalize keys when present
        mapping = getattr(f1plot, "COMPOUND_COLORS", None)  # type: ignore[attr-defined]
        if isinstance(mapping, dict):
            # keys in FastF1 are typically 'SOFT','MEDIUM','HARD','INTERMEDIATE','WET'
            return cast(str, mapping.get(c, _COMPOUND_FALLBACK.get(c, "#888888")))
    except Exception:
        pass
    return _COMPOUND_FALLBACK.get(c, "#888888")


def savefig(fig: Figure, path: str | Path, *, dpi: int = 220) -> Path:
    """Save figure to *path* with consistent DPI and tight layout; ensure parent dir exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(p, dpi=dpi)
    return p
