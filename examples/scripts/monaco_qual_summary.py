from pathlib import Path
import matplotlib.pyplot as plt

from fastf1_portfolio.session_loader import load_session
from fastf1_portfolio.plotting import apply_style, get_team_color, seconds_formatter


def main(year: int = 2024, gp: str = "Monaco", session: str = "Q") -> None:
    apply_style()
    s = load_session(year, gp, session)
    laps = s.laps.pick_quicklaps()
    fastest = laps.groupby("Driver")["LapTime"].min().sort_values()
    # Convert LapTime (timedelta) to seconds for easy labeling
    fastest_sec = fastest.dt.total_seconds()

    teams = (
        s.laps[["Driver", "Team"]].drop_duplicates().set_index("Driver").loc[fastest.index]["Team"]
    )
    colors = [get_team_color(team, session=s) for team in teams]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(fastest.index, fastest_sec.values, color=colors, edgecolor="black", alpha=0.9)
    ax.invert_yaxis()  # pole at top
    ax.set_xlabel("Best Lap (s)")
    ax.set_title(f"{s.event['EventName']} {s.event.year} – Best Qualifying Lap per Driver")

    # nice seconds -> M:SS.sss ticks
    ax.xaxis.set_major_formatter(seconds_formatter())

    # data labels
    for y, v in enumerate(fastest_sec.values):
        ax.text(v + 0.05, y, seconds_formatter()(v, None), va="center")

    out = Path("docs") / "assets" / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    path = (
        out / f"{s.event['EventName'].replace(' ', '_').lower()}_{s.event.year}_qual_bestlaps.png"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    print(f"Saved figure → {path}")


if __name__ == "__main__":
    main()
