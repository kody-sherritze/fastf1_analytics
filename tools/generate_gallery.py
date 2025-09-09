from __future__ import annotations

from pathlib import Path
import yaml

GALLERY_MD = Path("docs/gallery.md")
ASSETS_DIR = Path("docs/assets/gallery")
BEGIN = "<!-- AUTO-GALLERY:BEGIN -->"
END = "<!-- AUTO-GALLERY:END -->"


def load_items() -> list[dict]:
    items: list[dict] = []
    for yml in sorted(ASSETS_DIR.glob("*.yml")):
        with yml.open("r", encoding="utf-8") as f:
            items.append(yaml.safe_load(f))
    return items


def render_gallery(items: list[dict]) -> str:
    lines = ['<div class="grid cards" markdown>', ""]
    for it in items:
        title = it["title"]
        subtitle = it.get("subtitle", "")
        img = it["image"]
        code = it.get("code_path", "")
        params = it.get("params", {})
        # Compact param preview
        ppreview = ", ".join(f"{k}={v}" for k, v in params.items())
        lines += [
            f"- :material-chart-bar: **{title}**",
            "  ---",
            f"  [![{title}]({img}){{ loading=lazy }}]({img}){{ .glightbox }}",
            f"  _{subtitle}_",
            "",
            f"  `Source:` `{code}`  ",
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
