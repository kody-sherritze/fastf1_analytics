from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

from fastf1_analytics.plotting import (
    _norm_key,
    fmt_laptime_seconds,
    get_compound_color,
    get_team_color,
    lighten_color,
    savefig,
)
from fastf1_analytics.utils import ensure_list
from tools.utils import (
    event_slug,
    get_metadata_image_path,
    get_output_paths,
    write_plot_metadata,
)


def test_ensure_list_normalizes_common_inputs() -> None:
    assert ensure_list(None) == []
    assert ensure_list("VER") == ["VER"]
    assert ensure_list(("VER", "HAM")) == ["VER", "HAM"]
    assert ensure_list({"VER"}) == ["VER"]


def test_get_output_paths_uses_same_directory_for_png_and_yaml(tmp_path: Path) -> None:
    png, yml = get_output_paths(tmp_path / "nested" / "out", "2024-italian-gp")
    assert png == tmp_path / "nested" / "out" / "2024-italian-gp.png"
    assert yml == tmp_path / "nested" / "out" / "2024-italian-gp.yaml"
    assert png.parent.is_dir()


def test_event_slug_uses_gp_suffix() -> None:
    assert event_slug("Austrian Grand Prix") == "austrian-gp"
    assert event_slug("Austrian") == "austrian-gp"


def test_write_plot_metadata_writes_utf8_yaml(tmp_path: Path) -> None:
    output = write_plot_metadata(
        tmp_path / "nested" / "plot.yaml",
        {"title": "Résumé", "tags": ["race", "tyres"]},
    )

    assert output.read_text(encoding="utf-8") == (
        "title: Résumé\n" "tags:\n" "- race\n" "- tyres\n"
    )


def test_get_metadata_image_path_uses_output_location() -> None:
    assert get_metadata_image_path("docs/assets/gallery/plot.png") == "assets/gallery/plot.png"
    assert get_metadata_image_path("docs/assets/custom/plot.png") == "assets/custom/plot.png"


def test_norm_key_removes_case_and_separators() -> None:
    assert _norm_key("Visa Cash App RB") == "visacashapprb"


def test_fallback_colors_cover_known_and_unknown_values() -> None:
    assert get_team_color("Ferrari") == "#DC0000"
    assert get_team_color("unknown team") == "#888888"
    assert get_team_color("") == "#888888"
    assert get_compound_color("soft") == "#EA3223"
    assert get_compound_color("unknown") == "#888888"


def test_lap_time_and_lighten_color_are_deterministic() -> None:
    assert fmt_laptime_seconds(73.456) == "1:13.456"
    assert lighten_color("#000000", amount=0) == "#000000"
    assert lighten_color("#000000", amount=1) == "#ffffff"


def test_savefig_creates_parent_and_writes_figure(tmp_path: Path) -> None:
    fig = plt.figure()
    try:
        output = savefig(fig, tmp_path / "nested" / "plot.png", dpi=40)
        assert output.exists()
        assert output.stat().st_size > 0
    finally:
        plt.close(fig)


def test_timedelta_values_are_compatible_with_expected_plot_inputs() -> None:
    values = pd.to_timedelta(["0 days 00:01:13.456", "0 days 00:01:14.000"])
    assert values.total_seconds().tolist() == [73.456, 74.0]
