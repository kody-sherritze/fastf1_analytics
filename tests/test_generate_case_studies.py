from tools import generate_case_studies


def test_build_repro_command_includes_script_and_params() -> None:
    item = {
        "title": "Tyre Strategy",
        "image": "assets/gallery/strategy.png",
        "code_path": "tools/plots/tyre_strategy.py",
        "params": {"year": 2025, "event": "Italian Grand Prix", "driver_order": "results"},
    }

    result = generate_case_studies.build_repro_command(item)

    assert "python tools/plots/tyre_strategy.py" in result
    assert "--year 2025" in result
    assert '--event "Italian Grand Prix"' in result
    assert "--driver-order results" in result
    assert "--cache .fastf1-cache" in result


def test_render_case_study_replaces_template_placeholders() -> None:
    item = {
        "title": "2025 Italian Grand Prix - Tyre Strategy",
        "subtitle": "Stints and compounds by driver",
        "summary": "Compared how tyre compounds and stint timing shaped the race.",
        "params": {"year": 2025, "event": "Italian Grand Prix", "session": "R"},
        "tags": ["strategy", "tyres"],
    }

    template = "# {title}\n\n## Why this matters\n\n{summary}\n"
    result = generate_case_studies.render_case_study(item, template)

    assert "2025 Italian Grand Prix - Tyre Strategy" in result
    assert "Compared how tyre compounds and stint timing shaped the race." in result


def test_slugify_makes_url_safe_name() -> None:
    result = generate_case_studies.slugify("2025 Italian Grand Prix - Tyre Strategy")

    assert result == "2025-italian-grand-prix-tyre-strategy"
