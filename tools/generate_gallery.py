from __future__ import annotations

from pathlib import Path
import yaml
from typing import Any

GALLERY_MD = Path("docs/gallery.md")
ASSETS_DIR = Path("docs/assets/gallery")
BEGIN = "<!-- AUTO-GALLERY:BEGIN -->"
END = "<!-- AUTO-GALLERY:END -->"
REPO = "kody-sherritze/fastf1_analytics"


def load_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for yml in sorted(ASSETS_DIR.glob("*.yaml")):
        with yml.open("r", encoding="utf-8") as f:
            items.append(yaml.safe_load(f))
    return items


def render_gallery(items: list[dict[str, Any]]) -> str:
    lines = ['<div class="grid cards" markdown>', ""]
    for it in items:
        title = it["title"]
        subtitle = it.get("subtitle", "")
        img = it["image"]
        code = it.get("code_path", "")
        code_url = it.get("code_url", "")
        # Build a GitHub URL if only code_path is present
        if not code_url and code:
            code_url = f"https://github.com/{REPO}/blob/main/{str(code).replace('\\\\','/')}"
        params = it.get("params", {})
        # Compact param preview
        ppreview = ", ".join(f"{k}={v}" for k, v in params.items())
        # Make the title itself a hyperlink to source, when available
        title_line = f"- :material-chart-bar: **{title}**"
        lines += [
            title_line,
            "  ---",
            f"  [![{title}]({img}){{ loading=lazy }}]({img}){{ .glightbox }}",
            f"  _{subtitle}_",
            "",
            # Keep a source line for quick scanning; prefer a clickable link
            (f"  `Source:` [{code}]({code_url})  " if code_url else f"  `Source:` `{code}`  "),
            f"  `Params:` `{ppreview}`",
            "",
        ]
    lines += ["</div>", ""]
    return "\n".join(lines)


def replace_block(text: str, block: str) -> str:
    if BEGIN in text and END in text:
        pre = text.split(BEGIN, 1)[0]
        post = text.split(END, 1)[1]
        return f"{pre}{BEGIN}\n{block}\n{END}{post}"
    # If markers are missing, append after any existing content
    return f"{text.rstrip()}\n\n{BEGIN}\n{block}\n{END}\n"


def main() -> None:
    items = load_items()
    gallery_block = render_gallery(items)

    GALLERY_MD.parent.mkdir(parents=True, exist_ok=True)
    current = GALLERY_MD.read_text(encoding="utf-8") if GALLERY_MD.exists() else ""
    updated = replace_block(current, gallery_block)
    GALLERY_MD.write_text(updated, encoding="utf-8")
    print(f"Updated {GALLERY_MD} with {len(items)} items.")


if __name__ == "__main__":
    main()
