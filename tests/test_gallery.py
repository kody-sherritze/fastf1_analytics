from tools import generate_gallery


def test_render_gallery_builds_source_urls_and_optional_metadata() -> None:
    rendered = generate_gallery.render_gallery(
        [
            {
                "title": "Tyre strategy",
                "image": "assets/gallery/strategy.png",
                "code_path": r"tools\plots\tyre_strategy.py",
                "params": {"year": 2025, "driver_order": "results"},
            }
        ]
    )

    assert "Tyre strategy" in rendered
    assert "tools\\plots\\tyre_strategy.py" in rendered
    assert "tools/plots/tyre_strategy.py" in rendered
    assert "year=2025, driver_order=results" in rendered
    assert '<a id="tyre-strategy"></a>' in rendered


def test_render_gallery_handles_empty_items() -> None:
    rendered = generate_gallery.render_gallery([])
    assert rendered.startswith('<div class="grid cards" markdown>')
    assert rendered.endswith("</div>\n")


def test_replace_block_replaces_existing_markers() -> None:
    text = "before\n<!-- AUTO-GALLERY:BEGIN -->\nold\n<!-- AUTO-GALLERY:END -->\nafter"

    result = generate_gallery.replace_block(text, "new")

    assert result == "before\n<!-- AUTO-GALLERY:BEGIN -->\nnew\n<!-- AUTO-GALLERY:END -->\nafter"


def test_replace_block_appends_when_markers_are_missing() -> None:
    result = generate_gallery.replace_block("existing\n", "new")

    assert result.endswith("<!-- AUTO-GALLERY:BEGIN -->\nnew\n<!-- AUTO-GALLERY:END -->\n")
