from pathlib import Path
import matplotlib.pyplot as plt
from fastf1_portfolio.session_loader import load_session
from fastf1_portfolio.plotting import apply_style
from fastf1 import plotting as f1plot
from fastf1 import utils as f1utils


def compare_fastest_laps(year: int, gp: str, session: str, drv1: str, drv2: str) -> None:
    apply_style()
    s = load_session(year, gp, session)

    lap1 = s.laps.pick_drivers(drv1).pick_fastest()
    lap2 = s.laps.pick_drivers(drv2).pick_fastest()

    tel1 = lap1.get_car_data().add_distance()
    tel2 = lap2.get_car_data().add_distance()

    color1 = f1plot.get_team_color(lap1["Team"], session=s)
    color2 = f1plot.get_team_color(lap2["Team"], session=s)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tel1["Distance"], tel1["Speed"], color=color1, label=drv1)
    ax.plot(tel2["Distance"], tel2["Speed"], color=color2, label=drv2)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.legend(loc="upper left")
    plt.suptitle(f"Fastest Lap Speed Trace – {s.event['EventName']} {s.event.year} {session}")

    # Optional delta time (approximate; FastF1 notes it is not perfectly accurate)
    try:
        delta_t, ref_tel, comp_tel = f1utils.delta_time(lap1, lap2)
        twin = ax.twinx()
        twin.plot(ref_tel["Distance"], delta_t, linestyle="--", color="white")
        twin.set_ylabel("Δt (s)  (drv2 ahead  ◀  |  ▶  drv1 ahead)")
    except Exception as e:
        print(f"delta_time unavailable: {e}")

    out = Path("docs") / "assets" / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    path = (
        out
        / f"{s.event['EventName'].replace(' ', '_').lower()}_{s.event.year}_tel_{drv1}_{drv2}.png"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    print(f"Saved figure → {path}")


if __name__ == "__main__":
    compare_fastest_laps(2024, "Monaco", "Q", "VER", "LEC")
