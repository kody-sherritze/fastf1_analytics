from __future__ import annotations
import argparse
from pathlib import Path
import yaml

from fastf1_portfolio.session_loader import load_session
from fastf1_portfolio.charts.tyre_strategy import TyreStrategyParams, build_tyre_strategy


def slug(year: int, event: str) -> str:
    return f"{event.strip().lower().replace(' ', '_')}_{year}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a Tyre Strategy plot + YAML.")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--event", required=True, help="e.g. 'Monaco' or 'Italian Grand Prix'")
    ap.add_argument("--cache", default=".fastf1-cache")
    ap.add_argument("--outdir", default="docs/assets/gallery")
    ap.add_argument(
        "--driver-order", default="results", help="'results'|'alpha'|comma-list e.g. LEC,VER'"
    )
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    # Resolve driver order param
    order = (
        [x.strip().upper() for x in args.driver_order.split(",")]
        if "," in args.driver_order
        else args.driver_order
    )

    session = load_session(args.year, args.event, "R", cache=args.cache)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    s = slug(args.year, session.event["EventName"])
    png = outdir / f"{s}_tyre_strategy.png"
    yml = outdir / f"{s}_tyre_strategy.yaml"

    params = TyreStrategyParams(
        driver_order=order, title=args.title, dpi=args.dpi, annotate_compound=True
    )
    build_tyre_strategy(session, params=params, out_path=png)

    meta = {
        "title": params.title or f"{args.year} {session.event['EventName']} – Tyre Strategy",
        "subtitle": "Stints and compounds by driver",
        "image": f"assets/gallery/{png.name}",
        "code_path": "tools/plots/tyre_strategy.py",
        "function": "fastf1_portfolio.charts.tyre_strategy.build_tyre_strategy",
        "params": {
            "driver_order": order if isinstance(order, list) else str(order),
            "bar_height": params.bar_height,
            "bar_gap": params.bar_gap,
            "annotate_compound": params.annotate_compound,
            "dpi": params.dpi,
        },
        "tags": ["race", "strategy", "tyres"],
    }
    with yml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    print(f"Wrote {png} and {yml}")


if __name__ == "__main__":
    main()
