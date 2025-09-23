from __future__ import annotations

from pathlib import Path
from typing import Any, List
import yaml

GALLERY_MD = Path("docs/gallery.md")
ASSETS_DIR = Path("docs/assets/gallery")
BEGIN = "<!-- AUTO-GALLERY:BEGIN -->"
END = "<!-- AUTO-GALLERY:END -->"
REPO = "kody-sherritze/fastf1_analytics"


def load_items() -> List[dict[str, Any]]:
    items: List[dict[str, Any]] = []
    if not ASSETS_DIR.exists():
        return items
    for yf in sorted(ASSETS_DIR.glob("*.yaml")):
        data = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
        items.append(
            {
                "title": data.get("title") or yf.stem.replace("_", " ").title(),
                "subtitle": data.get("subtitle", ""),
                "image": data.get("image") or f"assets/gallery/{yf.stem}.png",
                "function": data.get("function", ""),
                "code_path": data.get("code_path", ""),
                "code_url": data.get("code_url", ""),
                "params": data.get("params", {}) or {},
            }
        )
    return items


def render_gallery(items: List[dict[str, Any]]) -> str:
    # IMPORTANT: This exact structure is what Material expects for card grids.
    lines: List[str] = ['<div class="grid cards" markdown>', ""]

    for it in items:
        title = it["title"]
        subtitle = it.get("subtitle", "")
        img = it.get("image", "")
        code = it.get("code_path", "")
        code_url = it.get("code_url", "")
        if not code_url and code:
            code_url = f"https://github.com/{REPO}/blob/main/{str(code).replace('\\','/')}"
        params = it.get("params", {})
        ppreview = ", ".join(f"{k}={v}" for k, v in params.items())

        # DRS live widget for the drs_effectiveness function; static image for others.
        is_drs = (
            it.get("function")
            == "fastf1_analytics.charts.drs_effectiveness.build_drs_effectiveness_distance"
        )

        lines.append(f"- :material-chart-bar: **{title}**")
        lines.append("  ---")

        if is_drs:
            default_year = params.get("year", "")
            default_event = str(params.get("event", "")).lower().replace(" ", "_")
            default_driver = params.get("driver", "")
            lines.append(
                f'  <div class="drs-widget" '
                f'data-index="assets/data/drs/index.json" '
                f'data-default-year="{default_year}" '
                f'data-default-event="{default_event}" '
                f'data-default-driver="{default_driver}"></div>'
            )
        else:
            # Static image card w/ lightbox
            lines.append(f"  [![{title}]({img}){{ loading=lazy }}]({img}){{ .glightbox }}")

        # Meta lines (kept minimal and on separate lines for readability)
        if subtitle:
            lines.append(f"  _{subtitle}_")
        lines.append("")
        if code_url:
            lines.append(f"  `Source:` [{code}]({code_url})")
        elif code:
            lines.append(f"  `Source:` `{code}`")
        if ppreview:
            lines.append(f"  `Params:` `{ppreview}`")
        lines.append("")

    lines.append("</div>")
    lines.append("")
    return "\n".join(lines)


def replace_block(current: str, new_block: str) -> str:
    # Replace the region between BEGIN and END; if markers are missing, append the block + markers.
    if BEGIN in current and END in current:
        start = current.index(BEGIN) + len(BEGIN)
        end = current.index(END, start)
        return current[:start] + "\n\n" + new_block + "\n\n" + current[end:]
    if current and not current.endswith("\n"):
        current += "\n"
    return current + "\n".join([BEGIN, "", new_block, "", END, ""])


def main() -> None:
    items = load_items()
    gallery_block = render_gallery(items)

    GALLERY_MD.parent.mkdir(parents=True, exist_ok=True)
    if not GALLERY_MD.exists():
        # Create a simple scaffold if the page doesn't exist yet
        scaffold = f"# Gallery\n\n{BEGIN}\n\n{gallery_block}\n\n{END}\n"
        GALLERY_MD.write_text(scaffold, encoding="utf-8")
        print(f"Created {GALLERY_MD} with {len(items)} items.")
        return

    current = GALLERY_MD.read_text(encoding="utf-8")
    updated = replace_block(current, gallery_block)
    GALLERY_MD.write_text(updated, encoding="utf-8")
    print(f"Updated {GALLERY_MD} with {len(items)} items.")


if __name__ == "__main__":
    main()
