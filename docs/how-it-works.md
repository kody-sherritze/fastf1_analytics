# How it works

## Pipeline
1. **Plot script** in `tools/plots/*.py` loads data (via `session_loader.load_session`), calls a chart
   function from `src/fastf1_analytics/charts/*`, and writes:
   - a **PNG** to `docs/assets/gallery/…`
   - a **YAML** sidecar with title, code path, and parameters.
2. **Gallery generator** (`tools/generate_gallery.py`) reads all sidecars and rewrites the card grid
   between markers in `docs/gallery.md`.
3. **MkDocs** builds a static site — no live API calls on GitHub Pages.

## Caching
- FastF1 cache (default `.fastf1-cache/`) speeds up runs; feel free to delete to refresh data.
- Scripts create the cache directory if missing.

## Consistent look & colors
- `fastf1_analytics.plotting` centralizes:
  - `apply_style`, `savefig`
  - `get_team_color`, `get_compound_color`, `lighten_color`

## Reproduce a tile
```bash
python tools/plots/tyre_strategy.py --year 2025 --event "Italian Grand Prix" --cache .fastf1-cache
python tools/generate_gallery.py
mkdocs serve
```