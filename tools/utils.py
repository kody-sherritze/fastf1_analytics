import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def configure_logging() -> None:
    """Configure visible INFO-level logging for standalone command-line tools."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO)
        return

    for handler in root_logger.handlers:
        if handler.level == logging.NOTSET or handler.level > logging.INFO:
            handler.setLevel(logging.INFO)


def event_slug(event: str) -> str:
    """Normalize a Grand Prix name for use in generated filenames."""
    normalized = "-".join(event.strip().lower().split())
    if normalized.endswith("-grand-prix"):
        normalized = normalized[: -len("-grand-prix")]
    elif normalized.endswith("-grandprix"):
        normalized = normalized[: -len("-grandprix")]
    return f"{normalized}-gp"


def get_output_paths(
    outdir: str | Path, slug: str, *, create_dir: bool = True
) -> tuple[Path, Path]:
    """Return the paired PNG/YAML output paths for a generated plot.

    The PNG and YAML sidecars are always written into the same directory, and the
    directory is created automatically unless disabled.
    """
    out_path = Path(outdir)
    if create_dir:
        out_path.mkdir(parents=True, exist_ok=True)

    stem = str(slug).strip()
    if not stem:
        raise ValueError("slug must not be empty")

    return out_path / f"{stem}.png", out_path / f"{stem}.yaml"


def get_metadata_image_path(path: str | Path) -> str:
    """Return a POSIX image path relative to the documentation root when possible."""
    image_path = Path(path)
    parts = image_path.parts
    if parts and parts[0].lower() == "docs":
        image_path = Path(*parts[1:])
    return image_path.as_posix()


REQUIRED_PLOT_METADATA_FIELDS = (
    "title",
    "subtitle",
    "image",
    "code_path",
    "function",
    "params",
    "tags",
)


def validate_plot_metadata(metadata: Any, source: str | Path) -> dict[str, Any]:
    """Validate and return the required metadata contract for a plot sidecar."""
    source_path = Path(source)
    if not isinstance(metadata, Mapping):
        raise TypeError(f"Plot metadata in {source_path} must be a YAML mapping")

    missing = [
        field
        for field in REQUIRED_PLOT_METADATA_FIELDS
        if field not in metadata or metadata[field] is None
    ]
    if missing:
        fields = ", ".join(missing)
        raise ValueError(f"Plot metadata in {source_path} is missing required fields: {fields}")

    for field in REQUIRED_PLOT_METADATA_FIELDS:
        if field in {"params", "tags"}:
            continue
        if not isinstance(metadata[field], str) or not metadata[field].strip():
            raise ValueError(f"Plot metadata field '{field}' in {source_path} must not be empty")
    if not isinstance(metadata["params"], Mapping):
        raise TypeError(f"Plot metadata field 'params' in {source_path} must be a mapping")
    if not isinstance(metadata["tags"], list):
        raise TypeError(f"Plot metadata field 'tags' in {source_path} must be a list")

    return dict(metadata)


def write_plot_metadata(path: str | Path, metadata: Mapping[str, Any]) -> Path:
    """Write plot metadata as consistently formatted UTF-8 YAML."""
    metadata_path = Path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        yaml.safe_dump(dict(metadata), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return metadata_path
