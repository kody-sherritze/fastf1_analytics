from pathlib import Path


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
