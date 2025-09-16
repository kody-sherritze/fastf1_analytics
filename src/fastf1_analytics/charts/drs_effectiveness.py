"""
DRS Effectiveness Chart
-----------------------

This module provides a “best‑lap” view of the effect of DRS (Drag
Reduction System) on a race track.  It locates the straight on each lap where
the DRS flap is actually open and selects the fastest such lap along with
the fastest lap without DRS.  Both laps are resampled onto a common
distance grid and plotted together to visualise the pointwise speed
difference and total time gain.  Only race laps are considered by default
and laps under pit, safety car or yellow flag conditions are excluded.  If
no DRS‑enabled lap is found in the race, the driver’s fastest valid
qualifying lap is used as the DRS reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Optional, cast, NamedTuple

import numpy as np
import pandas as pd
from pandas.api.types import is_timedelta64_dtype
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fastf1_analytics.plotting import apply_style, savefig
from fastf1_analytics.session_loader import load_session


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

    #: Minimum fraction of the window that must have the DRS flap open to
    #: classify a lap as "DRS ON".  Values should be in the range [0, 1].
    min_open_ratio_on: float = 0.15
    #: Maximum fraction of the window that may have the DRS flap open to
    #: classify a lap as "DRS OFF".  Values should be in the range [0, 1].
    max_open_ratio_off: float = 0.02


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


def _drs_open_flags(df: pd.DataFrame) -> np.ndarray:
    """Return 1 when the DRS flap is actually open (codes 12 or 14), else 0."""
    if "DRS" not in df.columns:
        return np.zeros(len(df), dtype=float)
    drs_raw = pd.to_numeric(df["DRS"], errors="coerce").fillna(0).astype(int).to_numpy()
    return np.isin(drs_raw, (12, 14)).astype(float)


def _drs_zone_bounds(win: pd.DataFrame) -> Optional[tuple[float, float]]:
    """Return (d_start, d_end) for the longest contiguous region where DRS is open (codes 12/14).
    If none found, return None.
    """
    if "Distance" not in win.columns:
        return None
    d = pd.to_numeric(win["Distance"], errors="coerce").to_numpy(dtype=float)
    drs = _drs_open_flags(win)  # 1 for open, 0 otherwise
    if d.size == 0 or drs.size == 0 or np.all(drs == 0):
        return None

    # find contiguous runs where drs == 1
    runs = []
    i = 0
    n = drs.size
    while i < n:
        if drs[i] >= 0.5:
            j = i
            while j + 1 < n and drs[j + 1] >= 0.5:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    if not runs:
        return None

    # choose the run with the largest distance span
    best = max(runs, key=lambda ij: d[ij[1]] - d[ij[0]])
    i0, i1 = best
    d0, d1 = float(d[i0]), float(d[i1])
    if d1 <= d0:
        return None
    return d0, d1


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


class _LapMeta(NamedTuple):
    lap_no: int
    lap_time_s: Optional[float]
    stint: Optional[int]
    compound: Optional[str]
    session: str  # "R" or "Q"


def _time_from_resampled(distance_m: float, speed_kmh: np.ndarray) -> float:
    """Compute time (s) by integrating over the resampled speed trace.
    Assumes the x-grid is uniform in distance from d_start to d_end.
    """
    if not np.isfinite(distance_m) or distance_m <= 0 or speed_kmh.size < 2:
        return float("nan")
    v_ms = np.maximum(speed_kmh, 1e-3) * (1000.0 / 3600.0)  # km/h -> m/s, clamp
    seg_len = distance_m / (speed_kmh.size - 1)  # equal Δd per segment
    v_mid = 0.5 * (v_ms[1:] + v_ms[:-1])
    return float(np.sum(seg_len / v_mid))


def _fmt_meta(m: _LapMeta | None) -> str:
    if m is None:
        return "n/a"
    lt = f"{m.lap_time_s:.3f}s" if (m.lap_time_s is not None) else "n/a"
    st = f"{m.stint}" if (m.stint is not None) else "n/a"
    cp = m.compound or "n/a"
    return f"Lap {m.lap_no} (session={m.session}, lap_time={lt}, stint={st}, compound={cp})"


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
    off_windows: list[dict[str, Any]] = []  # will store {"win": DataFrame, "meta": _LapMeta}

    best_on_meta: _LapMeta | None = None
    best_off_meta: _LapMeta | None = None

    # thresholds for classification come from the params dataclass
    MIN_OPEN_RATIO_ON = params.min_open_ratio_on
    MAX_OPEN_RATIO_OFF = params.max_open_ratio_off

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

        # determine the physical bounds of this straight window
        # (we compute these before resampling so we can derive a time from the full length)
        d0 = float(pd.to_numeric(win["Distance"], errors="coerce").iloc[0])
        d1 = float(pd.to_numeric(win["Distance"], errors="coerce").iloc[-1])
        # resample to normalized grid
        x_i, s_i, drs_i = _resample_normalized(win, params.n_points)
        if np.all(np.isnan(s_i)):
            continue

        # classify and time the window
        open_ratio = float(drs_i.mean())
        # compute the time across the full window using the resampled speeds
        t_sec = _time_from_resampled(d1 - d0, s_i)
        if not np.isfinite(t_sec):
            continue

        # --- collect lap metadata ---
        lap_no = int(lap["LapNumber"]) if "LapNumber" in lap else -1
        lap_time_s = None
        if "LapTime" in lap and pd.notna(lap["LapTime"]):
            try:
                lap_time_s = float(lap["LapTime"].total_seconds())
            except Exception:
                lap_time_s = None
        stint = int(lap["Stint"]) if "Stint" in lap and pd.notna(lap["Stint"]) else None
        compound = str(lap["Compound"]) if "Compound" in lap and pd.notna(lap["Compound"]) else None
        meta = _LapMeta(
            lap_no=lap_no, lap_time_s=lap_time_s, stint=stint, compound=compound, session="R"
        )

        if open_ratio >= MIN_OPEN_RATIO_ON:
            # distance bounds of THIS ON window
            d0 = float(pd.to_numeric(win["Distance"], errors="coerce").iloc[0])
            d1 = float(pd.to_numeric(win["Distance"], errors="coerce").iloc[-1])

            # which turns bound this straight in THIS lap
            tp = _turn_pair_for_segment(
                tel, s_idx, e_idx, params.accel_threshold_kmh_s, params.sustain_sec
            )
            t_s, t_e = tp if tp is not None else (None, None)

            # tighten straight to the actual DRS zone
            zone = _drs_zone_bounds(win)
            if zone is not None:
                zd0, zd1 = zone
                win_zone = win[
                    (pd.to_numeric(win["Distance"], errors="coerce") >= zd0)
                    & (pd.to_numeric(win["Distance"], errors="coerce") <= zd1)
                ].copy()
                if len(win_zone) >= 5:
                    win_zone = win_zone.sort_values("Distance")
                    xz, sz, dz = _resample_normalized(win_zone, params.n_points)
                    tz = _time_from_resampled(zd1 - zd0, sz)
                    # Note: dz.mean() should be close to 1.0 by construction
                    if (best_on is None) or (tz < best_on.time_sec):
                        best_on = _BestLap(xz, sz, dz, tz, zd0, zd1, t_s, t_e)
                        best_on_meta = meta  # (or Q meta in the fallback)
            else:
                # Fallback: no DRS-adjacent zone found; keep the whole straight window (rare)
                if (best_on is None) or (t_sec < best_on.time_sec):
                    best_on = _BestLap(x_i, s_i, drs_i, t_sec, d0, d1, t_s, t_e)
                    best_on_meta = meta

        elif open_ratio <= MAX_OPEN_RATIO_OFF:
            # Skip Lap 1 for DRS OFF
            if meta.lap_no == 1:
                pass
            else:
                off_windows.append({"tel": tel, "meta": meta})

    # Fallback: if no ON in race, try fastest valid qualifying lap (same auto-straight)
    if best_on is None:
        try:
            qual = load_session(
                session.event.year, session.event["EventName"], "Q", cache=str(session._api.path)
            )
        except Exception:
            qual = None
        if qual is not None:
            qlaps = qual.laps.pick_drivers([drv]).copy()
            qlap = None
            try:
                qlap = qlaps.pick_quicklaps().sort_values("LapTime").iloc[0]
            except Exception:
                qlap = qlaps.sort_values("LapTime").iloc[0] if len(qlaps) else None
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
                        # compute distance bounds for the qualifying window
                        d0 = float(pd.to_numeric(qwin["Distance"], errors="coerce").iloc[0])
                        d1 = float(pd.to_numeric(qwin["Distance"], errors="coerce").iloc[-1])
                        tq = _time_from_resampled(d1 - d0, sq)
                        if open_ratio_q >= MIN_OPEN_RATIO_ON and np.isfinite(tq):
                            tp = _turn_pair_for_segment(
                                qtel, qs, qe, params.accel_threshold_kmh_s, params.sustain_sec
                            )
                            t_s, t_e = tp if tp is not None else (None, None)
                            best_on = _BestLap(xq, sq, dq, tq, d0, d1, t_s, t_e)
                            best_on_meta = _LapMeta(
                                lap_no=int(qlap["LapNumber"]) if "LapNumber" in qlap else -1,
                                lap_time_s=(
                                    float(qlap["LapTime"].total_seconds())
                                    if "LapTime" in qlap and pd.notna(qlap["LapTime"])
                                    else None
                                ),
                                stint=(
                                    int(qlap["Stint"])
                                    if "Stint" in qlap and pd.notna(qlap["Stint"])
                                    else None
                                ),
                                compound=(
                                    str(qlap["Compound"])
                                    if "Compound" in qlap and pd.notna(qlap["Compound"])
                                    else None
                                ),
                                session="Q",
                            )
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
    elif best_on is not None and off_windows:
        d0_on, d1_on = best_on.d_start, best_on.d_end
        for cand in off_windows:
            tel_off = cand["tel"]
            meta_off = cand["meta"]

            # never use Lap 1 for OFF
            if meta_off.lap_no == 1:
                continue

            if "Distance" not in tel_off.columns:
                continue  # safety guard

            # require the same compound for both laps
            if best_on_meta is not None and best_on_meta.compound:
                comp_on = str(best_on_meta.compound).strip().upper()
                comp_off = str(meta_off.compound).strip().upper() if meta_off.compound else ""
                if comp_off != comp_on:
                    continue

            w = tel_off[
                (pd.to_numeric(tel_off["Distance"], errors="coerce") >= d0_on)
                & (pd.to_numeric(tel_off["Distance"], errors="coerce") <= d1_on)
            ].copy()

            if len(w) < 5:
                continue
            w = w.sort_values("Distance")

            xo, so, do = _resample_normalized(w, params.n_points)

            # ensure it's truly OFF within this span
            if float(do.mean()) > MAX_OPEN_RATIO_OFF:
                print(
                    f"Skip OFF candidate (Lap {meta_off.lap_no}): DRS open {float(do.mean()):.0%} in ON zone"
                )
                continue

            to = _time_from_resampled(d1_on - d0_on, so)
            if not np.isfinite(to):
                continue

            if (best_off is None) or (to < best_off.time_sec):
                best_off = _BestLap(
                    xo, so, do, to, d0_on, d1_on, best_on.turn_start, best_on.turn_end
                )
                best_off_meta = meta_off

    if best_off is not None:
        off_avg_speed = float(np.nanmean(best_off.speed))
        off_open_ratio = float(np.nanmean(best_off.drs_bin))
        turns = (
            f"T{best_on.turn_start}→T{best_on.turn_end}"
            if (
                best_on is not None
                and best_on.turn_start is not None
                and best_on.turn_end is not None
            )
            else "n/a"
        )
        print(
            f"[DRS OFF] {_fmt_meta(best_off_meta)} | "
            f"window_time={best_off.time_sec:.3f}s | "
            f"avg_speed={off_avg_speed:.1f} km/h | "
            f"open_ratio={off_open_ratio:.1%} | "
            f"straight={turns}"
        )
        ax.plot(
            best_off.x,
            best_off.speed,
            label="Fastest DRS OFF",
            linewidth=2.0,
            color="#ff6b81",  # pink
        )

    if best_on is not None:
        on_avg_speed = float(np.nanmean(best_on.speed))
        on_open_ratio = float(np.nanmean(best_on.drs_bin))
        turns = (
            f"T{best_on.turn_start}→T{best_on.turn_end}"
            if (best_on.turn_start is not None and best_on.turn_end is not None)
            else "n/a"
        )
        print(
            f"[DRS ON]  {_fmt_meta(best_on_meta)} | "
            f"window_time={best_on.time_sec:.3f}s | "
            f"avg_speed={on_avg_speed:.1f} km/h | "
            f"open_ratio={on_open_ratio:.1%} | "
            f"straight={turns}"
        )
        span_m = best_on.d_end - best_on.d_start
        print(f"DRS-zone distance used: {span_m:.1f} m (T{best_on.turn_start}→T{best_on.turn_end})")
        ax.plot(
            best_on.x,
            best_on.speed,
            label="Fastest DRS ON",
            linewidth=2.2,
            color="#2ecc71",  # green
        )

    if best_on is not None and best_off is None:
        msg = "No valid DRS-OFF candidate found within the ON window"
        if best_on_meta is not None and best_on_meta.compound:
            msg += f" (compound={best_on_meta.compound} enforced)"
        msg += "."
        print(msg)

    if best_on is not None and best_on.turn_start is not None and best_on.turn_end is not None:
        ax.text(
            0.015,
            0.90,
            (
                f"Selected DRS zone: Turn {best_on.turn_start} \u2192 Turn {best_on.turn_end}"
                if (
                    best_on is not None
                    and best_on.turn_start is not None
                    and best_on.turn_end is not None
                )
                else "Selected DRS zone"
            ),
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
        # cumulative time gain along the DRS zone (ON faster -> positive)
        v_on_ms = np.maximum(best_on.speed, 1e-3) * (1000.0 / 3600.0)
        v_off_ms = np.maximum(best_off.speed, 1e-3) * (1000.0 / 3600.0)
        seg_len = (best_on.d_end - best_on.d_start) / (best_on.speed.size - 1)
        v_on_mid = 0.5 * (v_on_ms[1:] + v_on_ms[:-1])
        v_off_mid = 0.5 * (v_off_ms[1:] + v_off_ms[:-1])
        dt_diff = seg_len * (1.0 / v_off_mid - 1.0 / v_on_mid)  # seconds per segment
        cum_dt = np.concatenate([[0.0], np.cumsum(dt_diff)])

        ax2 = ax.twinx()
        ax2.plot(
            best_on.x,
            cum_dt,
            linewidth=1.2,
            alpha=0.9,
            linestyle=":",
            label="Cumulative time gain (s) (right axis)",
        )
        ax2.set_ylabel("Cumulative time gain (s)")
        ax2.grid(False)

        # Time saved annotation from the same cumulative curve
        time_saved = float(cum_dt[-1])
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
