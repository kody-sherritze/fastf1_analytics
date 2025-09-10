"""
DRS Effectiveness Chart
-----------------------

This module implements a distance‑aligned view of the effect of DRS (Drag
Reduction System) along the main straight of a race track.  It aggregates
telemetry from multiple laps for a single driver, separates laps into those
where DRS was active during the straight and those where it was not, and
visualizes the median speed traces with interquartile range (IQR) bands.
A thin band underneath the main traces shows the pointwise speed delta (ON
minus OFF).  The design reflects user preferences:

* Only race laps are considered by default.
* In‑ and out‑laps as well as safety car/virtual safety car/yellow flag laps
  are excluded.
* A sustained deceleration threshold (rolling dV/dt) identifies the end of
  the main straight.  The first segment where dV/dt is below a threshold
  for at least ``sustain_sec`` seconds marks the braking point.
* Laps are labelled "DRS ON" if any telemetry sample in the window has
  ``DRS > 0``, else "DRS OFF".
* Each lap’s window is resampled to a fixed number of points on a 0–1
  normalized distance axis; median and IQR are computed across laps for
  ON and OFF classes separately.
* If a race yields no DRS‑ON laps (e.g. driver never had DRS on the straight),
  the fallback reference is the driver’s fastest valid qualifying lap for
  the same event and year.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Optional, List, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_timedelta64_dtype
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fastf1_portfolio.plotting import apply_style, savefig
from fastf1_portfolio.session_loader import load_session


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DRSEffectivenessParams:
    """Parameters controlling the DRS effectiveness plot.

    Attributes:
        n_points: Number of samples along the normalized 0–1 distance axis.
        accel_threshold_kmh_s: Threshold (in km/h/s) for sustained deceleration
            used to detect the onset of braking.
        sustain_sec: Minimum duration (in seconds) that the deceleration must
            remain below ``accel_threshold_kmh_s`` to count as braking.
        dpi: Resolution (dots per inch) for saving figures.
        title: Optional custom title for the plot.  If None, a title is
            constructed from the session and driver.
        show_annotations: Whether to annotate activation points.  Defaults to
            False per user preference.
    """

    n_points: int = 200
    accel_threshold_kmh_s: float = -8.0
    sustain_sec: float = 0.30
    dpi: int = 220
    title: Optional[str] = None
    show_annotations: bool = False


# -----------------------------------------------------------------------------
# Lap filtering helpers
# -----------------------------------------------------------------------------

_DANGER_CODES = {"4", "5", "6", "7", "8"}  # SC, VSC, VSC End, Yellow, Double Yellow


def _clean_laps(session: Any, driver: str) -> pd.DataFrame:
    """Return race laps for ``driver`` excluding in/out and SC/VSC/Yellow laps.

    If called on a non‑race session, the same filters are applied.
    """
    laps_any = session.laps.pick_drivers(driver).copy()
    laps: pd.DataFrame = cast(pd.DataFrame, laps_any)
    # Remove in/out laps based on pit timing or flags
    for col in ("PitInTime", "PitOutTime"):
        if col in laps.columns:
            laps = laps[laps[col].isna()]
    for col in ("InLap", "OutLap"):
        if col in laps.columns:
            laps = laps[~laps[col].fillna(False)]
    # Remove laps with dangerous track status codes (SC/VSC/Yellow)
    if "TrackStatus" in laps.columns:

        def _ok(ts: str | float | int) -> bool:
            s = str(ts) if pd.notna(ts) else ""
            parts = {p.strip() for p in s.split("+") if p.strip()}
            return parts.isdisjoint(_DANGER_CODES)

        laps = laps[laps["TrackStatus"].apply(_ok)]
    return laps.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Telemetry processing helpers
# -----------------------------------------------------------------------------


def _ensure_timedelta_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a Timedelta index for time‑based rolling.

    If a ``Time`` column exists and is a timedelta dtype, set it as the index.
    Otherwise, construct a synthetic monotonic time index using the sample
    order (1 ms apart) as a fallback.
    """
    df = df.copy()
    if not isinstance(df.index, pd.TimedeltaIndex):
        # If a 'Time' column is Timedelta, set it as the index
        if "Time" in df.columns and is_timedelta64_dtype(df["Time"]):
            df = df.set_index("Time")
        else:
            # fallback: monotonic synthetic time axis
            df = df.set_index(pd.to_timedelta(np.arange(len(df)), unit="ms"))
    # fallback: monotonic synthetic time axis
    return df


def _first_sustained_brake_idx(
    tel: pd.DataFrame,
    accel_threshold_kmh_s: float,
    sustain_sec: float,
) -> Optional[int]:
    """Return the row index where sustained braking starts, else None.

    A sustained braking event is detected when the rolling mean of d(Speed)/dt
    (in km/h/s) over a time window of ``sustain_sec`` seconds falls below
    ``accel_threshold_kmh_s``.
    """
    if tel is None or len(tel) == 0:
        return None
    df = tel[["Time", "Speed"]].dropna().copy()
    if df.empty:
        return None
    df = _ensure_timedelta_index(df)
    # Convert to floats
    v = df["Speed"].astype(float)
    # Time values for derivative calculation
    idx_ns = df.index.to_numpy(dtype="timedelta64[ns]").astype("int64")
    t_seconds = idx_ns.astype(float) / 1e9
    dv = v.diff().to_numpy()
    dt = np.diff(t_seconds, prepend=t_seconds[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        a_vals = np.divide(dv, dt, out=np.zeros_like(dv), where=dt != 0)
    # Rolling mean over the sustain window
    a = pd.Series(a_vals, index=df.index, dtype=float)

    # Time-based rolling mean of acceleration (uses TimedeltaIndex)
    a_roll = a.rolling(f"{sustain_sec}s").mean()
    a_roll_np = a_roll.to_numpy(dtype=float)

    hits = np.where(a_roll_np < float(accel_threshold_kmh_s))[0]
    return int(hits[0]) if len(hits) else None


def _slice_main_straight_window(tel: pd.DataFrame, brake_idx: Optional[int]) -> pd.DataFrame:
    """Slice telemetry to the main straight window.

    Returns data from the start/finish (Distance near 0) up to ``brake_idx``.
    If ``brake_idx`` is None or out of bounds, returns the full telemetry.
    """
    if tel is None or len(tel) == 0:
        return tel
    cols_needed = [c for c in ("Distance", "Speed", "DRS", "Time") if c in tel.columns]
    df = tel[cols_needed].dropna(subset=[c for c in cols_needed if c != "DRS"]).copy()
    if brake_idx is not None and 0 <= brake_idx < len(df):
        df = df.iloc[: brake_idx + 1]
    # Sort by distance and deduplicate on Distance for safety
    df = df.sort_values("Distance").drop_duplicates(subset=["Distance"], keep="first")
    return df.reset_index(drop=True)


def _resample_normalized(
    df: pd.DataFrame,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample telemetry onto a normalized 0–1 distance axis.

    Returns (x_norm, speed_interp, drs_binary_interp).
    """

    # empty/None guard
    if df is None or len(df) == 0:
        x = np.linspace(0.0, 1.0, n_points)
        return x, np.full_like(x, np.nan, dtype=float), np.zeros_like(x, dtype=float)

    # pull arrays; coerce to float and drop bad samples
    d = pd.to_numeric(df["Distance"], errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(df["Speed"], errors="coerce").to_numpy(dtype=float)

    if "DRS" in df.columns:
        drs_raw = pd.to_numeric(df["DRS"], errors="coerce").fillna(0).astype(int).to_numpy()
        drs_open = np.isin(drs_raw, (12, 14)).astype(float)
    else:
        drs_open = np.zeros_like(s, dtype=float)

    d0, d1 = float(np.nanmin(d)), float(np.nanmax(d))
    span = max(d1 - d0, 1e-6)
    x = np.linspace(0.0, 1.0, n_points)
    d_target = d0 + x * span

    speed_i = np.interp(d_target, d, s)
    drs_i = np.interp(d_target, d, drs_open)  # still 0..1 after interp
    drs_i = (drs_i > 0.5).astype(float)  # binarize on the grid
    return x, speed_i, drs_i


def _aggregate_class(samples: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (median, IQR) across a list of aligned 1D arrays."""
    if not samples:
        return np.array([]), np.array([])
    arr = np.vstack(samples)
    med = np.nanmedian(arr, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    iqr = q75 - q25
    return med, iqr


def _median_activation_point(drs_traces: List[np.ndarray], x: np.ndarray) -> Optional[float]:
    """Return the median normalized activation distance across DRS‑ON laps."""
    if not drs_traces:
        return None
    starts = []
    for b in drs_traces:
        idx = np.where(b > 0.5)[0]
        if len(idx):
            starts.append(float(x[idx[0]]))
    return float(np.median(starts)) if starts else None


# -----------------------------------------------------------------------------
# Main chart builder
# -----------------------------------------------------------------------------


def build_drs_effectiveness_distance(
    session: Any,
    *,
    driver: str,
    params: DRSEffectivenessParams = DRSEffectivenessParams(),
    out_path: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Build a distance‑aligned DRS effectiveness chart for a given driver.

    The returned figure contains two curves (median speed with DRS ON and OFF)
    with IQR bands and an optional delta trace plotted on a secondary axis.
    """
    apply_style()
    drv = str(driver).upper()
    # Filter laps according to user criteria
    laps = _clean_laps(session, drv)
    # Prepare containers
    x_grid = np.linspace(0.0, 1.0, params.n_points)
    on_speeds: List[np.ndarray] = []
    on_drsbin: List[np.ndarray] = []
    off_speeds: List[np.ndarray] = []
    # Process each lap
    for _, lap in laps.iterrows():
        try:
            tel = lap.get_car_data().add_distance()
        except Exception:
            continue
        brake_idx = _first_sustained_brake_idx(
            tel, params.accel_threshold_kmh_s, params.sustain_sec
        )
        win = _slice_main_straight_window(tel, brake_idx)
        if win is None or len(win) < 5:
            continue
        _, speed_i, drs_i = _resample_normalized(win, params.n_points)
        if np.all(np.isnan(speed_i)):
            continue
        open_ratio = float(drs_i.mean())  # fraction of straight with flap open
        if open_ratio >= 0.15:  # 10–20% works; 0.15 is a good default
            on_speeds.append(speed_i)
            on_drsbin.append(drs_i)
        else:
            off_speeds.append(speed_i)
    # Fallback: if no ON laps in race, use fastest valid qualifying lap
    if not on_speeds:
        try:
            qual = load_session(
                session.event.year, session.event["EventName"], "Q", cache=str(session._api.path)
            )
        except Exception:
            qual = None
        if qual is not None:
            qlaps = qual.laps.pick_drivers(drv).copy()
            qlap = None
            try:
                qlap = qlaps.pick_quicklaps().sort_values("LapTime").iloc[0]
            except Exception:
                if len(qlaps):
                    qlap = qlaps.sort_values("LapTime").iloc[0]
            if qlap is not None:
                try:
                    qtel = qlap.get_car_data().add_distance()
                    qbrk = _first_sustained_brake_idx(
                        qtel, params.accel_threshold_kmh_s, params.sustain_sec
                    )
                    qwin = _slice_main_straight_window(qtel, qbrk)
                    _, qs_i, qdrs_i = _resample_normalized(qwin, params.n_points)
                    if not np.all(np.isnan(qs_i)):
                        on_speeds.append(qs_i)
                        on_drsbin.append((qdrs_i > 0.5).astype(float))
                except Exception:
                    pass
    # Aggregate medians and IQRs
    on_med, on_iqr = _aggregate_class(on_speeds)
    off_med, off_iqr = _aggregate_class(off_speeds)
    fig, ax = plt.subplots(figsize=(10, 6))
    # Handle empty
    if on_med.size == 0 and off_med.size == 0:
        ax.text(0.5, 0.5, "No usable telemetry after filtering", ha="center", va="center")
        ax.set_axis_off()
        if out_path:
            savefig(fig, out_path, dpi=params.dpi)
        return fig, ax
    # Plot OFF first
    if off_med.size:
        ax.plot(x_grid, off_med, label="DRS OFF", linewidth=2.0)
        if off_iqr.size:
            q25 = off_med - 0.5 * off_iqr
            q75 = off_med + 0.5 * off_iqr
            ax.fill_between(x_grid, q25, q75, alpha=0.20, linewidth=0)
    # Plot ON
    if on_med.size:
        ax.plot(x_grid, on_med, label="DRS ON", linewidth=2.2)
        if on_iqr.size:
            q25 = on_med - 0.5 * on_iqr
            q75 = on_med + 0.5 * on_iqr
            ax.fill_between(x_grid, q25, q75, alpha=0.20, linewidth=0)
    # Delta trace on secondary axis
    if on_med.size and off_med.size:
        delta = on_med - off_med
        ax2 = ax.twinx()
        ax2.plot(x_grid, delta, linewidth=1.2, alpha=0.8, linestyle=":", label="Δ speed (ON−OFF)")
        ax2.set_ylabel("Δ speed (km/h)")
        ax2.grid(False)
    else:
        ax2 = None

    # Small caption: what the shaded bands mean (top-left of main axes)
    ax.text(
        0.015,
        0.97,
        "Shaded bands = interquartile range (25–75%)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#CCCCCC",
    )
    # If we drew the delta line on ax2, include it in the legend
    if ax2 is not None:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower right", frameon=False)
    else:
        ax.legend(loc="lower right", frameon=False)

    # Title and labels
    if params.title:
        title = params.title
    else:
        try:
            title = f"{session.event.year} {session.event['EventName']} – DRS Effectiveness (Main Straight) – {drv}"
        except Exception:
            title = f"DRS Effectiveness – {drv}"
    ax.set_title(title)
    ax.set_xlabel("Normalized distance along main straight (0 → 1)")
    ax.set_ylabel("Speed (km/h)")
    ax.margins(x=0.01)
    # Activation annotation
    if params.show_annotations and on_drsbin:
        act = _median_activation_point(on_drsbin, x_grid)
        if act is not None:
            ax.axvline(act, color="#888888", linewidth=1.0, linestyle="--", alpha=0.6)
    if out_path:
        savefig(fig, out_path, dpi=params.dpi)
    return fig, ax


__all__ = [
    "DRSEffectivenessParams",
    "build_drs_effectiveness_distance",
]
