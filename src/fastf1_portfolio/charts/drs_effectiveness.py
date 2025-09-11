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
from typing import Any, Tuple, Optional, List, cast, NamedTuple

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
    df: pd.DataFrame, n_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample (Distance, Speed, DRS) onto [0..1] grid; return x, speed, drs_open (0/1)."""
    if df is None or len(df) == 0:
        x = np.linspace(0.0, 1.0, n_points)
        return x, np.full_like(x, np.nan, dtype=float), np.zeros_like(x, dtype=float)

    d = pd.to_numeric(df["Distance"], errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(df["Speed"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(d) & np.isfinite(s)
    if not np.any(m):
        x = np.linspace(0.0, 1.0, n_points)
        return x, np.full_like(x, np.nan, dtype=float), np.zeros_like(x, dtype=float)
    d = d[m]
    s = s[m]

    drs_open = _drs_open_flags(df)
    drs_open = drs_open[m]

    d0, d1 = float(np.nanmin(d)), float(np.nanmax(d))
    span = max(d1 - d0, 1e-6)
    x = np.linspace(0.0, 1.0, n_points)
    d_target = d0 + x * span

    speed_i = np.interp(d_target, d, s)
    drs_i = np.interp(d_target, d, drs_open)  # in [0,1]
    drs_i = (drs_i > 0.5).astype(float)  # binarize grid
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


def _drs_open_flags(df: pd.DataFrame) -> np.ndarray:
    """Return 1 when the DRS flap is actually open (codes 12 or 14), else 0."""
    if "DRS" not in df.columns:
        return np.zeros(len(df), dtype=float)
    drs_raw = pd.to_numeric(df["DRS"], errors="coerce").fillna(0).astype(int).to_numpy()
    return np.isin(drs_raw, (12, 14)).astype(float)


def _ensure_timedelta_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a TimedeltaIndex for time-based rolling windows."""
    df = df.copy()
    if not isinstance(df.index, pd.TimedeltaIndex):
        if "Time" in df.columns and is_timedelta64_dtype(df["Time"]):
            df = df.set_index("Time")
        else:
            df = df.set_index(pd.to_timedelta(np.arange(len(df)), unit="ms"))
    return df


def _accel_series(tel: pd.DataFrame) -> pd.Series:
    """Compute a float acceleration Series (km/h/s) with a TimedeltaIndex."""
    df = tel[["Time", "Speed"]].dropna().copy()
    if df.empty:
        return pd.Series(dtype=float)
    df = _ensure_timedelta_index(df)
    v = df["Speed"].astype(float)
    # seconds from TimedeltaIndex
    idx_ns = df.index.to_numpy(dtype="timedelta64[ns]").astype("int64")
    t_seconds = idx_ns.astype(float) / 1e9
    dv = v.diff().to_numpy()
    dt = np.diff(t_seconds, prepend=t_seconds[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        a_vals = np.divide(dv, dt, out=np.zeros_like(dv), where=dt != 0)
    return pd.Series(a_vals, index=df.index, dtype=float)


def _brake_mask(a: pd.Series, sustain_sec: float, threshold: float) -> np.ndarray:
    """Return bool array where sustained braking is happening."""
    if a.empty:
        return np.zeros(0, dtype=bool)
    a_roll = a.rolling(f"{sustain_sec}s").mean()
    # Convert to ndarray for typed comparisons
    return a_roll.to_numpy(dtype=float) < float(threshold)


def _segments_from_brake_mask(mask: np.ndarray, min_len: int = 8) -> list[tuple[int, int]]:
    """Return contiguous (start, end) index pairs for non-braking segments."""
    if mask.size == 0:
        return []
    non_brake = ~mask
    segs: list[tuple[int, int]] = []
    i = 0
    n = non_brake.size
    while i < n:
        if non_brake[i]:
            j = i
            while j + 1 < n and non_brake[j + 1]:
                j += 1
            if (j - i + 1) >= min_len:
                segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs


def _turn_pair_for_segment(
    tel: pd.DataFrame,
    s_idx: int,
    e_idx: int,
    accel_threshold_kmh_s: float,
    sustain_sec: float,
) -> tuple[int, int] | None:
    """Return (turn_start, turn_end) for the non-braking segment [s_idx, e_idx].

    We (re)build the braking mask from this lap's telemetry using the same
    time-based dV/dt rule, find all non-braking segments (straights), then
    locate which straight contains (or best overlaps) [s_idx, e_idx].

    Mapping: straight #k (0-based) is between Turn (k+1) and Turn (k+2),
    wrapping so the last straight is between T{n} and T1.
    """
    # Build braking mask
    a = _accel_series(tel)
    if a.empty or len(a) != len(tel):
        return None
    brake = _brake_mask(a, sustain_sec=sustain_sec, threshold=accel_threshold_kmh_s)
    segs = _segments_from_brake_mask(brake, min_len=8)
    if not segs:
        return None

    # Find straight with max overlap with [s_idx, e_idx]
    best_k = -1
    best_overlap = -1
    for k, (s, e) in enumerate(segs):
        overlap = min(e, e_idx) - max(s, s_idx) + 1
        if overlap > best_overlap:
            best_overlap = overlap
            best_k = k

    if best_k < 0:
        return None

    n = len(segs)
    t_start = best_k + 1
    t_end = 1 if (t_start == n) else (t_start + 1)
    return (t_start, t_end)


def _select_drs_straight_indices(
    tel: pd.DataFrame, accel_threshold_kmh_s: float, sustain_sec: float
) -> tuple[int, int] | None:
    """Pick the lap's straight with strongest DRS-open presence (length-weighted).
    Falls back to the longest straight if DRS coverage is minimal.
    """
    # Ensure required columns
    cols = [c for c in ("Time", "Speed", "Distance", "DRS") if c in tel.columns]
    if not cols:
        return None
    t = tel[cols].dropna(subset=[c for c in cols if c != "DRS"]).copy()
    if t.empty:
        return None

    # Acceleration and braking mask
    a = _accel_series(t)
    if a.empty or len(a) != len(t):
        return None
    brake = _brake_mask(a, sustain_sec=sustain_sec, threshold=accel_threshold_kmh_s)

    segs = _segments_from_brake_mask(brake, min_len=8)
    if not segs:
        return None

    dists = pd.to_numeric(t["Distance"], errors="coerce").to_numpy(dtype=float)
    drs_open = _drs_open_flags(t)

    # Score segments: (mean DRS open) * (distance length)
    best = None
    best_score = -1.0
    best_len = -1.0
    for s_idx, e_idx in segs:
        seg_len = float(dists[e_idx] - dists[s_idx])
        if seg_len <= 1.0:
            continue
        coverage = float(np.nanmean(drs_open[s_idx : e_idx + 1]))
        score = coverage * seg_len
        if (score > best_score) or (np.isclose(score, best_score) and seg_len > best_len):
            best_score = score
            best_len = seg_len
            best = (s_idx, e_idx)

    # If coverage is tiny everywhere, use the longest straight instead
    if best is None or best_score < 1e-6:
        # Longest by distance
        best = max(segs, key=lambda se: dists[se[1]] - dists[se[0]])

    return best


class _BestLap(NamedTuple):
    x: np.ndarray
    speed: np.ndarray
    drs_bin: np.ndarray
    time_sec: float
    d_start: float
    d_end: float
    turn_start: Optional[int]
    turn_end: Optional[int]


def _window_time_seconds(win: pd.DataFrame) -> float:
    """Duration across the selected straight window."""
    # Prefer Time column if it’s Timedelta
    if "Time" in win.columns and is_timedelta64_dtype(win["Time"]):
        t = win["Time"].dt.total_seconds().to_numpy(dtype=float)
        if t.size >= 2:
            return float(np.nanmax(t) - np.nanmin(t))

    # Fallback: integrate Δd / v (robust if Time missing)
    d = pd.to_numeric(win["Distance"], errors="coerce").to_numpy(dtype=float)
    s_kmh = pd.to_numeric(win["Speed"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(d) & np.isfinite(s_kmh)
    d = d[m]
    s_kmh = s_kmh[m]
    if d.size < 2:
        return float("nan")
    v_ms = np.maximum(s_kmh, 1e-3) * (1000.0 / 3600.0)  # clamp to avoid div/0
    dd = np.diff(d)
    v_mid = 0.5 * (v_ms[1:] + v_ms[:-1])
    return float(np.sum(dd / v_mid))


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
    """Best-case DRS effectiveness on the auto-selected DRS straight.

    Compares the fastest DRS-ON straight vs the fastest DRS-OFF straight
    (per-lap windows selected automatically where DRS is actually used).
    """
    apply_style()
    drv = str(driver).upper()

    # Filter laps (race-only, no in/out, no SC/VSC/Yellow)
    laps = _clean_laps(session, drv)

    # Track best ON and OFF windows (by minimal time across the straight)
    best_on: _BestLap | None = None
    best_off: _BestLap | None = None
    off_windows: list[pd.DataFrame] = []

    # thresholds for classification (keep tight to avoid glitches)
    MIN_OPEN_RATIO_ON = 0.15  # ≥15% of window open → ON
    MAX_OPEN_RATIO_OFF = 0.02  # ≤2% open → OFF

    for _, lap in laps.iterrows():
        try:
            tel = lap.get_car_data().add_distance()
        except Exception:
            continue

        sel = _select_drs_straight_indices(tel, params.accel_threshold_kmh_s, params.sustain_sec)
        if not sel:
            continue
        s_idx, e_idx = sel

        # build the straight window for the lap
        win = tel.iloc[s_idx : e_idx + 1][
            [c for c in ("Distance", "Speed", "DRS", "Time") if c in tel.columns]
        ].copy()
        if win is None or len(win) < 5:
            continue

        win = win.sort_values("Distance")

        # resample to normalized grid
        x_i, s_i, drs_i = _resample_normalized(win, params.n_points)
        if np.all(np.isnan(s_i)):
            continue

        # classify and time the window
        open_ratio = float(drs_i.mean())
        t_sec = _window_time_seconds(win)
        if not np.isfinite(t_sec):
            continue

        if open_ratio >= MIN_OPEN_RATIO_ON:
            # distance bounds of THIS ON window
            d0 = float(pd.to_numeric(win["Distance"], errors="coerce").iloc[0])
            d1 = float(pd.to_numeric(win["Distance"], errors="coerce").iloc[-1])

            # which turns bound this straight in THIS lap
            tp = _turn_pair_for_segment(
                tel, s_idx, e_idx, params.accel_threshold_kmh_s, params.sustain_sec
            )
            t_s, t_e = tp if tp is not None else (None, None)

            if (best_on is None) or (t_sec < best_on.time_sec):
                best_on = _BestLap(x_i, s_i, drs_i, t_sec, d0, d1, t_s, t_e)
        elif open_ratio <= MAX_OPEN_RATIO_OFF:
            off_windows.append(win)

    # Fallback: if no ON in race, try fastest valid qualifying lap (same auto-straight)
    if best_on is None:
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
                    sel_q = _select_drs_straight_indices(
                        qtel, params.accel_threshold_kmh_s, params.sustain_sec
                    )
                    if sel_q:
                        qs, qe = sel_q
                        qwin = (
                            qtel.iloc[qs : qe + 1][
                                [
                                    c
                                    for c in ("Distance", "Speed", "DRS", "Time")
                                    if c in qtel.columns
                                ]
                            ]
                            .copy()
                            .sort_values("Distance")
                        )
                        xq, sq, dq = _resample_normalized(qwin, params.n_points)
                        open_ratio_q = float(dq.mean())
                        tq = _window_time_seconds(qwin)
                        if open_ratio_q >= MIN_OPEN_RATIO_ON and np.isfinite(tq):
                            d0 = float(pd.to_numeric(qwin["Distance"], errors="coerce").iloc[0])
                            d1 = float(pd.to_numeric(qwin["Distance"], errors="coerce").iloc[-1])
                            tp = _turn_pair_for_segment(
                                qtel, qs, qe, params.accel_threshold_kmh_s, params.sustain_sec
                            )
                            t_s, t_e = tp if tp is not None else (None, None)
                            best_on = _BestLap(xq, sq, dq, tq, d0, d1, t_s, t_e)
                except Exception:
                    pass

    # Build the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Handle empty state
    if best_on is None and best_off is None:
        ax.text(0.5, 0.5, "No usable telemetry after filtering", ha="center", va="center")
        ax.set_axis_off()
        if out_path:
            savefig(fig, out_path, dpi=params.dpi)
        return fig, ax

    if best_on is not None and not off_windows:
        pass  # nothing to do
    elif best_on is not None:
        d0_on, d1_on = best_on.d_start, best_on.d_end
        for win_off in off_windows:
            # slice the OFF window to ON's distance span
            w = win_off[
                (pd.to_numeric(win_off["Distance"], errors="coerce") >= d0_on)
                & (pd.to_numeric(win_off["Distance"], errors="coerce") <= d1_on)
            ].copy()
            if len(w) < 5:
                # If the original OFF window is slightly wider, try re-slicing from the full tel segment
                # (optional) but generally skip if too sparse
                continue
            w = w.sort_values("Distance")
            xo, so, do = _resample_normalized(w, params.n_points)
            # ensure it's truly OFF within this span
            if float(do.mean()) > MAX_OPEN_RATIO_OFF:
                continue
            to = _window_time_seconds(w)
            if not np.isfinite(to):
                continue
            if (best_off is None) or (to < best_off.time_sec):
                best_off = _BestLap(
                    xo, so, do, to, d0_on, d1_on, best_on.turn_start, best_on.turn_end
                )

    if best_off is not None:
        ax.plot(
            best_off.x,
            best_off.speed,
            label="Fastest DRS OFF",
            linewidth=2.0,
            color="#ff6b81",  # pink
        )

    if best_on is not None:
        ax.plot(
            best_on.x,
            best_on.speed,
            label="Fastest DRS ON",
            linewidth=2.2,
            color="#2ecc71",  # green
        )

    if best_on is not None and best_on.turn_start is not None and best_on.turn_end is not None:
        ax.text(
            0.015,
            0.90,
            f"Selected straight: Turn {best_on.turn_start} \u2192 Turn {best_on.turn_end}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#CCCCCC",
        )

    ax.set_xlabel("Normalized distance along selected straight (0 → 1)")
    ax.set_ylabel("Speed (km/h)")

    # Dotted Δ line (right axis) if both present
    ax2 = None
    if (best_on is not None) and (best_off is not None):
        delta = best_on.speed - best_off.speed
        ax2 = ax.twinx()
        ax2.plot(
            best_on.x,
            delta,
            linewidth=1.2,
            alpha=0.85,
            linestyle=":",
            label="Δ speed (ON−OFF) (right axis)",
        )
        ax2.set_ylabel("Δ speed (km/h)")
        ax2.grid(False)

        # Time saved annotation
        time_saved = best_off.time_sec - best_on.time_sec  # >0 → ON faster
        ax.text(
            0.015,
            0.97,
            f"Best-lap time gain across straight: {time_saved:+.3f} s",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#CCCCCC",
        )

    # Legend (merge both axes so dotted line appears)
    if ax2 is not None:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower right", frameon=False)
    else:
        ax.legend(loc="lower right", frameon=False)

    # Title
    if params.title:
        title = params.title
    else:
        try:
            title = f"{session.event.year} {session.event['EventName']} – DRS Effectiveness (Best-Lap) – {drv}"
        except Exception:
            title = f"DRS Effectiveness (Best-Lap) – {drv}"
    ax.set_title(title)

    ax.margins(x=0.01)

    # Optional activation marker (if you want to keep it for the ON trace)
    if params.show_annotations and (best_on is not None):
        idx_on = np.where(best_on.drs_bin > 0.5)[0]
        if idx_on.size:
            act_x = float(best_on.x[idx_on[0]])
            ax.axvline(act_x, color="#888888", linewidth=1.0, linestyle="--", alpha=0.6)

    if out_path:
        savefig(fig, out_path, dpi=params.dpi)
    return fig, ax


__all__ = [
    "DRSEffectivenessParams",
    "build_drs_effectiveness_distance",
]
