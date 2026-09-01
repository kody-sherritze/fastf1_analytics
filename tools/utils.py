from pathlib import Path


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
