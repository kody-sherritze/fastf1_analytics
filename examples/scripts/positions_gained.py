from pathlib import Path
import matplotlib.pyplot as plt

from fastf1_portfolio.session_loader import load_session
from fastf1_portfolio.plotting import apply_style


def positions_gained(year: int, gp: str, session: str = "R") -> None:
    apply_style()
    s = load_session(year, gp, session)
    res = s.results.copy()

    res["Net"] = res["GridPosition"] - res["Position"]
    gainers = res.loc[res["Net"] > 0].sort_values("Net", ascending=False)
    losers = res.loc[res["Net"] < 0].sort_values("Net")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    ax1.barh(gainers["Abbreviation"], gainers["Net"], color="tab:green")
    ax1.set_title("Positions Gained")
    ax1.set_xlabel("+")

    ax2.barh(losers["Abbreviation"], losers["Net"], color="tab:red")
    ax2.set_title("Positions Lost")
    ax2.set_xlabel("-")

    fig.suptitle(f"Positions Change – {s.event['EventName']} {s.event.year}")
    for ax in (ax1, ax2):
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    out = Path("docs") / "assets" / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{s.event['EventName'].replace(' ', '_').lower()}_{s.event.year}_positions.png"
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    print(f"Saved figure → {path}")


if __name__ == "__main__":
    positions_gained(2024, "Monaco", "R")
