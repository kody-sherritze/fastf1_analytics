from fastf1_portfolio import load_session, apply_style
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    apply_style()
    s = load_session(2024, "Monaco", "Q")
    laps = s.laps.pick_quicklaps()
    fastest = laps.pick_fastest()
    print(f"Fastest lap: {fastest['Driver']} in {fastest['LapTime']}")
    ax = laps.groupby("Driver")["LapTime"].min().sort_values().plot(kind="bar")
    ax.set_title("Best Qualifying Lap per Driver - Monaco 2024")
    ax.set_ylabel("Lap Time")
    plt.tight_layout()

    out = Path("docs") / "assets" / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    fig_path = out / "monaco_2024_q_bestlaps.png"
    plt.savefig(fig_path)
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
