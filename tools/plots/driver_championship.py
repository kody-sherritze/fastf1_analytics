from __future__ import annotations

import argparse
from typing import Any
from pathlib import Path

import pandas as pd
import yaml
import fastf1

from fastf1_portfolio.session_loader import load_session
from fastf1_portfolio.charts.driver_points import (
    DriverPointsParams,
    build_driver_points_chart,
)


def _season_points_table(year: int, include_sprints: bool, cache: str) -> pd.DataFrame:
    """Build long-form table of cumulative points per driver per round."""
    sched = fastf1.get_event_schedule(year, include_testing=False)  # DataFrame
    rounds = [
        (int(row["RoundNumber"]), str(row["EventName"]))
        for _, row in sched.sort_values("RoundNumber").iterrows()
        if str(row.get("EventName", "")).strip()
    ]

    rows: list[dict[str, Any]] = []
    by_driver_total: dict[str, float] = {}

    for rnd, event in rounds:
        # Race points
        race = load_session(year, event, "R", cache=cache)
        race_res = getattr(race, "results", None)
        if race_res is not None and len(race_res) > 0:
            for _, r in race_res.iterrows():
                drv = str(r["Abbreviation"])
                pts = float(r.get("Points", 0.0))
                team = str(r.get("TeamName", ""))
                by_driver_total[drv] = by_driver_total.get(drv, 0.0) + pts

        # Optional sprint points
        if include_sprints:
            try:
                sprint = load_session(year, event, "S", cache=cache)
                spr_res = getattr(sprint, "results", None)
                if spr_res is not None and len(spr_res) > 0:
                    for _, r in spr_res.iterrows():
                        drv = str(r["Abbreviation"])
                        pts = float(r.get("Points", 0.0))
                        team = str(r.get("TeamName", ""))
                        by_driver_total[drv] = by_driver_total.get(drv, 0.0) + pts
            except Exception:
                # no sprint for this event
                pass

        # Snapshot totals after this round
        for drv, total in by_driver_total.items():
            # Use team from last race result if available, else blank
            # (Color selection later will still work with team="")
            team = ""
            if race_res is not None:
                try:
                    team_rows = race_res[race_res["Abbreviation"] == drv]
                    if len(team_rows) > 0:
                        team = str(team_rows.iloc[-1]["TeamName"])
                except Exception:
                    pass

            rows.append(
                {
                    "Round": rnd,
                    "EventName": event,
                    "Driver": drv,
                    "TeamName": team,
                    "PointsCum": total,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No points data found for the requested season.")
    return df


def _slug(year: int) -> str:
    return f"{year}_drivers_championship_points"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Drivers' Championship points chart + YAML.")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--cache", default=".fastf1-cache")
    ap.add_argument("--include-sprints", action="store_true")
    ap.add_argument("--color-variant", choices=["primary", "secondary"], default="primary")
    ap.add_argument("--min-total-points", type=float, default=0.0)
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--outdir", default="docs/assets/gallery")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    points_cum = _season_points_table(args.year, args.include_sprints, cache=args.cache)

    png = outdir / f"{_slug(args.year)}.png"
    yml = outdir / f"{_slug(args.year)}.yaml"

    params = DriverPointsParams(
        color_variant=args.color_variant,
        annotate_last=True,
        min_total_points=args.min_total_points,
        dpi=args.dpi,
        title=args.title,
    )

    build_driver_points_chart(points_cum, year=args.year, params=params, out_path=str(png))

    meta = {
        "title": params.title or f"{args.year} Drivers' Championship – Cumulative points",
        "subtitle": "Total points by race (lines per driver)",
        "image": f"assets/gallery/{png.name}",
        "code_path": "tools/plots/driver_championship.py",
        "function": "fastf1_portfolio.charts.driver_points.build_driver_points_chart",
        "params": {
            "year": args.year,
            "include_sprints": bool(args.include_sprints),
            "color_variant": params.color_variant,
            "min_total_points": params.min_total_points,
            "dpi": params.dpi,
        },
        "tags": ["season", "drivers", "points"],
    }
    yml.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    print(f"Wrote {png} and {yml}")


if __name__ == "__main__":
    main()
