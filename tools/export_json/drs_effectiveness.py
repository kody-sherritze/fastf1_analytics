from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from fastf1_analytics.session_loader import load_session


# ------------------------- helpers ------------------------- #


def _slugify_event(name: str) -> str:
    s = (name or "").strip().lower().replace(" ", "_")
    return "".join(ch for ch in s if ch.isalnum() or ch == "_")


def _clean_laps(laps: pd.DataFrame, driver: str) -> pd.DataFrame:
    """
    Filter laps to a single driver and remove pit in/out and SC/VSC laps.
    Works across FastF1 versions where columns differ in dtype.
    """
    # FastF1's pick_drivers returns a DataFrame-like; cast to satisfy mypy.
    picked = getattr(laps, "pick_drivers")(driver)
    df: pd.DataFrame = cast(pd.DataFrame, picked).copy()

    # 1) Drop pit in/out laps (handle datetime-like or boolean flags)
    for col in ("PitInTime", "PitOutTime"):
        if col in df.columns:
            s: pd.Series = df[col]
            if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
                df = df[s.isna()]
            else:
                # Treat truthy as pit lap; keep rows where value is falsy or NA
                df = df[~s.astype(bool).fillna(False)]

    # Optional flags if present
    for col in ("InLap", "OutLap"):
        if col in df.columns and pd.api.types.is_bool_dtype(df[col]):
            df = df[~df[col].fillna(False)]

    # 2) Drop laps under SC/VSC using TrackStatus (tokens like "1+2+4")
    if "TrackStatus" in df.columns:
        ts_str: pd.Series = df["TrackStatus"].astype("string").fillna("")
        # Vectorized contains for tokens 4 or 5 delimited by start/end or '+'
        # Matches: 4 or 5 as whole tokens in strings like "1+2+4"
        mask_sc_vsc: pd.Series = ts_str.str.contains(
            r"(?:(?:^|\+)(?:4|5)(?=$|\+))", regex=True, na=False
        )
        df = df[~mask_sc_vsc]

    return df.reset_index(drop=True)


def _lap_meta(lap: pd.Series) -> Dict[str, Any]:
    lap_number_val = lap.get("LapNumber", None)
    lap_no: int | None = int(lap_number_val) if pd.notna(lap_number_val) else None

    stint_val = lap.get("Stint", None)
    stint: int | None = int(stint_val) if pd.notna(stint_val) else None

    comp_val = lap.get("Compound", None)
    compound: str | None = str(comp_val) if pd.notna(comp_val) else None

    lap_time_s: float | None = None
    lt = lap.get("LapTime", None)
    if pd.notna(lt):
        try:
            # FastF1 LapTime is typically pandas.Timedelta
            lap_time_s = float(getattr(lt, "total_seconds")())
        except Exception:
            pass

    return {
        "lap": lap_no,
        "lap_time_s": lap_time_s,
        "stint": stint,
        "compound": compound,
    }


def _quantize(arr: npt.ArrayLike, decimals: int) -> List[float]:
    if arr is None:
        return []
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return []
    q = np.round(a, decimals=decimals).astype(float)
    return [float(x) for x in q.tolist()]


def _event_fields(session: Any) -> Tuple[int, str, str]:
    # year
    year: int
    try:
        year = int(getattr(session.event, "year"))
    except Exception:
        try:
            year = int(session.event.get("Year"))
        except Exception:
            year = int(session.event["Year"])
    # name
    try:
        event_name = str(session.event["EventName"])
    except Exception:
        event_name = str(getattr(session.event, "EventName", ""))
    slug = _slugify_event(event_name)
    return year, event_name, slug


# ------------------------- exporter ------------------------- #


def export(
    year: int,
    event: str,
    session_code: str,
    driver: str,
    out_dir: str = "docs/assets/data/drs",
    index_path: str = "docs/assets/data/drs/index.json",
) -> Path:
    """
    Export raw telemetry (t_s, dist_m, speed_kmh, drs) per lap for a given driver
    into JSON for client-side interactive DRS plotting.

    Returns: Path to the JSON file written.
    """
    session = load_session(year, event, session_code)

    # Laps filter
    laps = _clean_laps(session.laps, driver)
    files_dir = Path(out_dir)
    files_dir.mkdir(parents=True, exist_ok=True)

    ev_year, ev_name, ev_slug = _event_fields(session)

    obj: Dict[str, Any] = {
        "schema_version": 1,
        "meta": {
            "year": ev_year,
            "event_name": ev_name,
            "event_slug": ev_slug,
            "session": str(session_code),
            "driver": str(driver).upper(),
            "unit_speed": "km/h",
            "unit_dist": "m",
        },
        "laps": [],
    }

    # Build per-lap series
    for _, lap in laps.iterrows():
        # lap.get_car_data() is a FastF1 API; type is dynamic at type-check time
        try:
            car_data = getattr(lap, "get_car_data")()
            tel = getattr(car_data, "add_distance")()
        except Exception:
            continue

        cols = [c for c in ("Time", "Distance", "Speed", "DRS") if c in tel.columns]
        if not cols:
            continue

        t = tel[cols].dropna(subset=[c for c in cols if c != "DRS"]).copy()
        if t.empty:
            continue

        # Relative seconds (0 at start of this lap telemetry)
        if "Time" in t.columns:
            try:
                t0 = t["Time"].iloc[0]
                # Support both datetime64 and timedelta64-like
                t_s = (t["Time"] - t0).dt.total_seconds().to_numpy()
            except Exception:
                t_s = pd.to_timedelta(t["Time"]).dt.total_seconds().to_numpy()
        else:
            t_s = np.arange(len(t), dtype=float) * 0.001

        dist = (
            t["Distance"].to_numpy() if "Distance" in t.columns else np.arange(len(t), dtype=float)
        )
        speed = t["Speed"].to_numpy() if "Speed" in t.columns else np.zeros(len(t), dtype=float)

        if "DRS" in t.columns:
            drs_series: pd.Series = t["DRS"].fillna(0)
            drs_np = drs_series.to_numpy()
            drs_list: List[int] = [int(x) if pd.notna(x) else 0 for x in drs_np]
        else:
            drs_list = [0] * len(t)

        obj["laps"].append(
            {
                **_lap_meta(lap),
                "t_s": _quantize(t_s, 3),
                "dist_m": _quantize(dist, 2),
                "speed_kmh": _quantize(speed, 2),
                "drs": drs_list,
            }
        )

    # Write main JSON
    slug = f"{ev_slug}_{ev_year}_drs_effect_{obj['meta']['driver']}.json"
    out_path = files_dir / slug
    out_path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")

    # Update/Write index.json
    idx_path = Path(index_path)
    try:
        index: Dict[str, Any] = json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception:
        index = {"sessions": []}

    sess_key = (obj["meta"]["year"], obj["meta"]["event_name"], obj["meta"]["session"])
    found: Dict[str, Any] | None = None
    for s in index.get("sessions", []):
        if (s.get("year"), s.get("event_name"), s.get("session")) == sess_key:
            found = s
            break
    if not found:
        found = {
            "year": obj["meta"]["year"],
            "event_name": obj["meta"]["event_name"],
            "event_slug": obj["meta"]["event_slug"],
            "session": obj["meta"]["session"],
            "drivers": [],
            "files": {},
        }
        index["sessions"].append(found)

    drv = obj["meta"]["driver"]
    if drv not in found["drivers"]:
        found["drivers"].append(drv)
    found["files"][drv] = f"assets/data/drs/{slug}"

    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Export DRS telemetry JSON for client-side rendering")
    p.add_argument("--year", type=int, required=True)
    p.add_argument(
        "--event", type=str, required=True, help="Event name (e.g., 'Italian Grand Prix')"
    )
    p.add_argument("--session", type=str, default="R", help="Session code (R/Q/S/SP)")
    p.add_argument("--driver", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="docs/assets/data/drs")
    p.add_argument("--index", type=str, default="docs/assets/data/drs/index.json")
    args = p.parse_args()

    out = export(args.year, args.event, args.session, args.driver, args.out_dir, args.index)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
