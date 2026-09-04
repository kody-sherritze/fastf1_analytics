from __future__ import annotations

import argparse
import logging

from fastf1_analytics.charts.pace_consistency import (
    PaceConsistencyParams,
    build_pace_consistency,
)
from fastf1_analytics.session_loader import load_session
from tools.utils import (
    configure_logging,
    event_slug,
    get_metadata_image_path,
    get_output_paths,
    write_plot_metadata,
)

logger = logging.getLogger(__name__ + ".pace_consistency")


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser(description="Generate a race pace consistency plot (+ YAML).")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--event", required=True, help="e.g. 'Monaco' or 'Italian Grand Prix'")
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument("--drivers", type=int, default=10, help="Maximum drivers to plot")
    selection.add_argument("--driver", help="Plot one driver by abbreviation, e.g. VER")
    ap.add_argument("--cache", default=".fastf1-cache")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--outdir", default="docs/assets/gallery")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    session = load_session(args.year, args.event, "R", cache=args.cache)
    params = PaceConsistencyParams(
        title=args.title,
        drivers=1 if args.driver else args.drivers,
        driver=args.driver,
        dpi=args.dpi,
    )
    base = f"{args.year}-{event_slug(session.event['EventName'])}-pace-consistency"
    png, yml = get_output_paths(args.outdir, base)
    build_pace_consistency(session, params=params, out_path=png)

    meta = {
        "title": params.title
        or f"{session.event.year} {session.event['EventName']} - Race pace consistency",
        "subtitle": "Clean lap-time deltas from the selected drivers' average median pace",
        "image": get_metadata_image_path(png),
        "code_path": "tools/plots/pace_consistency.py",
        "function": "fastf1_analytics.charts.pace_consistency.build_pace_consistency",
        "params": {
            "year": args.year,
            "event": args.event,
            "drivers": params.drivers,
            "driver": params.driver,
            "dpi": params.dpi,
        },
        "tags": ["race", "pace", "consistency", "lap times"],
    }
    write_plot_metadata(yml, meta)
    logger.info("Wrote %s and %s", png, yml)


if __name__ == "__main__":
    main()
