from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

CASE_STUDIES_DIR = Path("docs/case-studies")
GALLERY_DIR = Path("docs/assets/gallery")
TEMPLATE_PATH = CASE_STUDIES_DIR / "_template.md"
INDEX_PATH = CASE_STUDIES_DIR / "index.md"


def load_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for yml in sorted(GALLERY_DIR.glob("*.yaml")):
        with yml.open("r", encoding="utf-8") as fh:
            item = yaml.safe_load(fh) or {}
        if item.get("case_study") is False:
            continue
        items.append(item)
    return items


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "case-study"


def format_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return f'"{value}"' if " " in value else value
    return str(value)


def build_repro_command(item: dict[str, Any]) -> str:
    code_path = str(item.get("code_path", "")).strip()
    params = item.get("params", {})
    flags: list[str] = []
    for key, value in params.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                flags.append(flag)
        elif value is None:
            continue
        else:
            flags.append(f"{flag} {format_cli_value(value)}")

    command = ["python", code_path, *flags]
    if code_path and "--cache" not in " ".join(command):
        command.append("--cache")
        command.append(".fastf1-cache")
    return " ".join(command)


def build_key_insights(item: dict[str, Any]) -> list[str]:
    subtitle = item.get("subtitle") or "This visual summarizes the key race pattern."
    tags = item.get("tags", [])
    tag_text = ", ".join(str(tag) for tag in tags[:2]) if tags else "race data"
    return [
        f"{subtitle} highlights the main pattern across the {tag_text} comparison.",
        "The chart helps explain how the selected metric changes over the event or across the field.",
        "The result is useful for identifying performance gaps, strategy trade-offs, or timing differences.",
    ]


def build_index(items: list[dict[str, Any]]) -> str:
    lines = [
        "# Case studies",
        "",
        "These pages are generated from the same gallery metadata used to build the main gallery grid. They provide a more narrative portfolio-style view of the project's strongest visuals.",
        "",
        "## Featured analyses",
        "",
    ]
    for item in items:
        slug = slugify(item.get("title", "case-study"))
        title = item.get("title", "Case study")
        lines.append(f"- [{title}](./{slug}.md)")
    return "\n".join(lines) + "\n"


def render_case_study(item: dict[str, Any], template: str) -> str:
    title = item.get("title", "Case study")
    subtitle = item.get("subtitle", "")
    summary = item.get("summary") or subtitle or "TODO: add a concise summary for this visual."
    analytical_question = (
        item.get("analytical_question")
        or "what stands out when comparing the selected metric across the field?"
    )
    key_insights = build_key_insights(item)
    params = item.get("params", {})
    event = params.get("event") or item.get("event") or "selected event"
    year = params.get("year") or item.get("year") or "the selected season"
    session = params.get("session") or item.get("session") or "R"
    plot_type = item.get("plot_type") or "race analysis chart"
    filtering_notes = (
        item.get("filtering_notes") or "selected session filters and relevant race laps"
    )
    calculation_notes = (
        item.get("calculation_notes")
        or "derived from lap-level and telemetry data for the selected comparison"
    )
    findings = item.get("key_findings") or (
        "The main takeaway is that the chart makes the most important comparison immediately visible, "
        "which helps explain the analytical story behind the result."
    )
    command = build_repro_command(item)
    skills = item.get("skills") or [
        "telemetry analysis",
        "Python plotting",
        "reproducible reporting",
        "visual storytelling",
    ]

    image = item.get("image", "")
    if image:
        image = image.replace("assets/gallery/", "../assets/gallery/")

    values = {
        "{title}": title,
        "{subtitle}": subtitle,
        "{summary}": summary,
        "{image}": image,
        "{analytical_question}": analytical_question,
        "{key_insight_1}": key_insights[0],
        "{key_insight_2}": key_insights[1],
        "{key_insight_3}": key_insights[2],
        "{year}": str(year),
        "{event}": str(event),
        "{session}": str(session),
        "{plot_type}": str(plot_type),
        "{filtering_notes}": filtering_notes,
        "{calculation_notes}": calculation_notes,
        "{key_findings_narrative}": findings,
        "{repro_command}": command,
        "{skill_1}": str(skills[0]),
        "{skill_2}": str(skills[1]),
        "{skill_3}": str(skills[2]),
        "{skill_4}": str(skills[3]) if len(skills) > 3 else str(skills[2]),
        "{related_visual_title}": "How it works",
        "{related_visual_path}": "/how-it-works/",
        "{related_visual_title_2}": "Gallery",
        "{related_visual_path_2}": "/gallery/",
    }

    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(key, value)
    return rendered


def ensure_directories() -> None:
    CASE_STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_directories()
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Missing template: {TEMPLATE_PATH}")
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    items = load_items()

    for item in items:
        title = item.get("title", "Case study")
        slug = slugify(title)
        target = CASE_STUDIES_DIR / f"{slug}.md"
        target.write_text(render_case_study(item, template), encoding="utf-8")

    INDEX_PATH.write_text(build_index(items), encoding="utf-8")
    print(f"Generated {len(items)} case-study stubs in {CASE_STUDIES_DIR}.")


if __name__ == "__main__":
    main()
