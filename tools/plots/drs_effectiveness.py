from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from fastf1_portfolio.session_loader import load_session
from fastf1_portfolio.charts.drs_effectiveness import (
    DRSEffectivenessParams,
    build_drs_effectiveness_distance,
)


def _slug(year: int, gp: str, driver: str) -> str:
    gp_slug = gp.lower().replace(" ", "_")
    return f"{gp_slug}_{year}_drs_effect_{driver.upper()}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate distance-aligned DRS effectiveness chart + YAML for the gallery."
    )
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--event", required=True, help="Grand Prix name as FastF1 expects it")
    ap.add_argument("--session", default="R", help="R (race) by default")
    ap.add_argument("--driver", required=True, help="Driver abbreviation, e.g. VER")
    ap.add_argument("--cache", default=".fastf1-cache")
    # chart params
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--accel-threshold-kmh-s", type=float, default=-8.0)
    ap.add_argument("--sustain-sec", type=float, default=0.30)
    ap.add_argument("--title", default=None)
    ap.add_argument("--outdir", default="docs/assets/gallery")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    session = load_session(args.year, args.event, args.session, cache=args.cache)

    params = DRSEffectivenessParams(
        n_points=args.n_points,
        accel_threshold_kmh_s=args.accel_threshold_kmh_s,
        sustain_sec=args.sustain_sec,
        show_annotations=False,  # per your preference
        title=args.title,
        dpi=220,
    )

    slug = _slug(args.year, args.event, args.driver)
    png = outdir / f"{slug}.png"
    yml = outdir / f"{slug}.yml"

    build_drs_effectiveness_distance(
        session,
        driver=args.driver,
        params=params,
        out_path=str(png),
    )

    meta = {
        "title": params.title
        or f"{session.event.year} {session.event['EventName']} – DRS effect on main straight ({args.driver.upper()})",
        "subtitle": "Median speed traces along main straight (DRS ON/OFF)",
        "image": f"assets/gallery/{png.name}",
        "code_path": "tools/plots/drs_effectiveness.py",
        "function": "fastf1_portfolio.charts.drs_effectiveness.build_drs_effectiveness_distance",
        "params": {
            "year": args.year,
            "event": args.event,
            "session": args.session,
            "driver": args.driver.upper(),
            "n_points": params.n_points,
            "accel_threshold_kmh_s": params.accel_threshold_kmh_s,
            "sustain_sec": params.sustain_sec,
        },
        "tags": ["drs", "speed", "straight"],
    }
    yml.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    print(f"Wrote {png} and {yml}")


if __name__ == "__main__":
    main()
