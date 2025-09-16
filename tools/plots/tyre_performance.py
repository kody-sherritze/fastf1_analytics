from __future__ import annotations
import argparse
from pathlib import Path
import yaml

from fastf1_analytics.session_loader import load_session
from fastf1_analytics.charts.tyre_performance import (
    TyrePerformanceParams,
    build_tyre_performance,
)


def slug(year: int, event: str) -> str:
    return f"{event.strip().lower().replace(' ', '_')}_{year}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a Tyre Lap Times plot (+ YAML).")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--event", type=str, required=True, help='e.g., "Italian Grand Prix"')
    ap.add_argument("--cache", type=str, default=".fastf1-cache")
    ap.add_argument("--min-laps-per-compound", type=int, default=5)
    ap.add_argument("--aggregate", choices=["median", "mean"], default="median")
    ap.add_argument("--include-inter-wet", action="store_true")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--outdir", type=str, default="docs/assets/gallery")
    args = ap.parse_args()

    session = load_session(args.year, args.event, "R", cache=args.cache)

    params = TyrePerformanceParams(
        min_laps_per_compound=args.min_laps_per_compound,
        aggregate=args.aggregate,
        include_inter_wet=args.include_inter_wet,
        dpi=args.dpi,
    )

    png_dir = Path(args.outdir)
    png_dir.mkdir(parents=True, exist_ok=True)
    base = f"{slug(args.year, args.event)}_tyre_performance"
    png = png_dir / f"{base}.png"

    build_tyre_performance(session, params=params, out_path=str(png))

    # YAML for docs gallery
    yml = Path("docs") / "assets" / "gallery" / f"{base}.yaml"
    yml.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "title": params.title
        or f"{session.event.year} {session.event['EventName']} – Tyre lap times (clean race laps)",
        "subtitle": "Bars = median across drivers; dots = each driver (team-colored), annotated by driver code",
        "image": f"assets/gallery/{png.name}",
        "code_path": "tools/plots/tyre_performance.py",
        "function": "fastf1_analytics.charts.tyre_performance.build_tyre_performance",
        "params": {
            "year": args.year,
            "event": args.event,
            "min_laps_per_compound": params.min_laps_per_compound,
            "aggregate": params.aggregate,
            "include_inter_wet": params.include_inter_wet,
            "dpi": params.dpi,
        },
        "tags": ["race", "tyres", "performance", "lap times"],
    }
    with yml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    print(f"Wrote {png} and {yml}")


if __name__ == "__main__":
    main()
