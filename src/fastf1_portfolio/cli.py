from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from . import load_session, apply_style


def monaco_main() -> None:
    parser = argparse.ArgumentParser(description="Generate Monaco qualifying best-lap chart.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--session", default="Q")
    parser.add_argument("--cache", default=".fastf1")
    args = parser.parse_args()

    apply_style()
    s = load_session(args.year, "Monaco", args.session, cache=args.cache)
    laps = s.laps.pick_quicklaps()
    ax = laps.groupby("Driver")["LapTime"].min().sort_values().plot(kind="bar")
    ax.set_title(f"Best Qualifying Lap per Driver - Monaco {args.year}")
    ax.set_ylabel("Lap Time")
    out = Path("docs") / "assets" / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    fig_path = out / f"monaco_{args.year}_{args.session}_bestlaps.png"
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"Saved figure to {fig_path}")
