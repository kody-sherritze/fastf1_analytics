from pathlib import Path
import matplotlib.pyplot as plt

from fastf1_portfolio.session_loader import load_session
from fastf1_portfolio.plotting import apply_style, seconds_formatter


def pace_evolution(year: int, gp: str, session: str, drivers: list[str]) -> None:
    apply_style()
    s = load_session(year, gp, session)
    laps = s.laps

    fig, ax = plt.subplots(figsize=(10, 6))
    for drv in drivers:
        dl = laps.pick_drivers(drv).pick_quicklaps()
        if dl.empty:
            continue
        ax.plot(
            dl["LapNumber"], dl["LapTime"].dt.total_seconds(), marker="o", linewidth=1, label=drv
        )
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap Time (s)")
    ax.yaxis.set_major_formatter(seconds_formatter())
    ax.legend(title="Driver")
    ax.set_title(f"Pace Evolution – {s.event['EventName']} {s.event.year} {session}")

    out = Path("docs") / "assets" / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{s.event['EventName'].replace(' ', '_').lower()}_{s.event.year}_pace.png"
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    print(f"Saved figure → {path}")


if __name__ == "__main__":
    pace_evolution(2024, "Monaco", "R", ["VER", "LEC", "HAM", "NOR"])
