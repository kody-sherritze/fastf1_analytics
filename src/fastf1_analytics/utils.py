from typing import Any, cast

import pandas as pd


def ensure_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


def clean_race_laps(session: Any, driver: str | None = None) -> pd.DataFrame:
    """Return race laps excluding pit, caution, and invalid-time laps.

    If ``driver`` is provided, only that driver's laps are returned.
    """
    laps_any = session.laps
    if driver is not None:
        if hasattr(laps_any, "pick_drivers"):
            laps_any = laps_any.pick_drivers(driver)
        else:
            laps_any = laps_any[laps_any["Driver"].astype(str).str.upper() == driver.upper()]
    laps_any = laps_any.copy()
    laps: pd.DataFrame = cast(pd.DataFrame, laps_any)

    for column in ("PitInTime", "PitOutTime"):
        if column in laps.columns:
            laps = laps[laps[column].isna()]
    for column in ("InLap", "OutLap"):
        if column in laps.columns:
            laps = laps[~laps[column].fillna(False)]

    if "TrackStatus" in laps.columns:

        def is_safe(track_status: Any) -> bool:
            status = str(track_status) if pd.notna(track_status) else ""
            codes = {part.strip() for part in status.split("+") if part.strip()}
            return codes == {"1"}

        laps = laps[laps["TrackStatus"].apply(is_safe)]

    laps = laps[laps["LapTime"].notna()].copy()
    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

    if "Compound" in laps.columns:
        laps["Compound"] = laps["Compound"].fillna("").astype(str).str.upper()
    else:
        laps["Compound"] = ""

    return laps.reset_index(drop=True)
