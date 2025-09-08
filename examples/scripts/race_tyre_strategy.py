from pathlib import Path
import matplotlib.pyplot as plt

from fastf1_portfolio.session_loader import load_session
from fastf1_portfolio.plotting import apply_style
import fastf1.plotting as f1plot


def plot_tyre_strategy(year: int, gp: str, session: str = "R") -> None:
    apply_style()
    s = load_session(year, gp, session)
    laps = s.laps

    # Group to compute stint lengths per compound (FastF1 example approach)
    stints = (
        laps[["Driver", "Stint", "Compound", "LapNumber"]]
        .groupby(["Driver", "Stint", "Compound"])
        .count()
        .reset_index()
    )
    stints = stints.rename(columns={"LapNumber": "StintLength"})

    drivers = [s.get_driver(d)["Abbreviation"] for d in s.drivers]
    # Order by finishing position (session results)
    try:
        order = list(s.results.sort_values("Position")["Abbreviation"])
        drivers = [d for d in order if d in drivers]
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, 10))
    for drv in drivers:
        d_stints = stints.loc[stints["Driver"] == drv]
        left = 0
        for _, row in d_stints.iterrows():
            color = f1plot.get_compound_color(row["Compound"], session=s)
            ax.barh(drv, row["StintLength"], left=left, color=color, edgecolor="black")
            left += int(row["StintLength"])

    ax.set_title(f"{s.event['EventName']} {s.event.year} – Tyre Strategies by Driver")
    ax.set_xlabel("Lap Number")
    ax.grid(False)
    try:
        ax.invert_yaxis()
    except Exception:
        pass
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    out = Path("docs") / "assets" / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    fig_path = (
        out / f"{s.event['EventName'].replace(' ', '_').lower()}_{s.event.year}_tyre_strategy.png"
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    print(f"Saved figure → {fig_path}")


if __name__ == "__main__":
    plot_tyre_strategy(2024, "Monaco", "R")
