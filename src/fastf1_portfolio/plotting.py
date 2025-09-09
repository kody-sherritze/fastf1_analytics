from __future__ import annotations

from typing import Any, cast
import re
from pathlib import Path
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import to_hex, to_rgb
import fastf1.plotting as f1plot


def apply_style(
    color_scheme: str | None = "fastf1",
    timedelta_support: bool = True,
) -> None:
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


def get_team_color(team: str, *, session: Any | None = None) -> str:
    """Return a hex color for a team.
    Tries FastF1's mapping; falls back to a local mapping so charts work even
    without a session object or for alternate team names.
    """
    if not team:
        return "#888888"
    try:
        # Prefer FastF1 when available; it knows historic liveries/aliases.
        return cast(str, f1plot.get_team_color(team, session=session))
    except Exception:
        # Public mapping if present in this FastF1 version
        mapping = getattr(f1plot, "TEAM_COLORS", None)
        if isinstance(mapping, dict) and team in mapping:
            return cast(str, mapping[team])
        # Normalize and try our local fallback
        return _TEAM_COLOR_FALLBACK.get(_norm_key(team), "#888888")


_TEAM_COLOR_FALLBACK = {
    # Common 2024/2025 team names and aliases (normalized keys below)
    "redbull": "#0600EF",
    "oracleredbullracing": "#0600EF",
    "ferrari": "#DC0000",
    "scuderiaferrari": "#DC0000",
    "mercedes": "#00D2BE",
    "mercedesamgpetronas": "#00D2BE",
    "mclaren": "#FF8700",
    "astonmartin": "#006F62",
    "alpine": "#0090FF",
    "williams": "#005AFF",
    "rb": "#143CFF",
    "visacashapprb": "#143CFF",
    "haas": "#B6BABD",
    "sauber": "#00E701",
    "kicksauber": "#00E701",
    "stake": "#00E701",
}


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def get_driver_color(driver: str, *, session: Any | None = None) -> str:
    """Return the (team) color for a driver abbreviation in a given session."""
    return cast(str, f1plot.get_driver_color(driver, session=session))


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
        mapping = getattr(f1plot, "COMPOUND_COLORS", None)
        if isinstance(mapping, dict):
            # keys in FastF1 are typically 'SOFT','MEDIUM','HARD','INTERMEDIATE','WET'
            return cast(str, mapping.get(c, _COMPOUND_FALLBACK.get(c, "#888888")))
    except Exception:
        pass
    return _COMPOUND_FALLBACK.get(c, "#888888")


def lighten_color(color: str, amount: float = 0.25) -> str:
    """Return a lightened variant of a hex color (used as a 'secondary' line color).

    amount=0 keeps the same color; amount=1 goes to white.
    """
    r, g, b = to_rgb(color)
    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount
    return to_hex((r, g, b))


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
