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


def write_plot_metadata(path: str | Path, metadata: Mapping[str, Any]) -> Path:
    """Write plot metadata as consistently formatted UTF-8 YAML."""
    metadata_path = Path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        yaml.safe_dump(dict(metadata), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return metadata_path
