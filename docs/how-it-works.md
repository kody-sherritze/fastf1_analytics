# How it works

## Pipeline
1. **Plot script** in `tools/plots/*.py` loads data (via `session_loader.load_session`), calls a chart
   function from `src/fastf1_analytics/charts/*`, and writes:
   - a **PNG** to `docs/assets/gallery/…`
   - a **YAML** sidecar with title, code path, and parameters.
2. **Gallery generator** (`tools/generate_gallery.py`) reads all sidecars and rewrites the card grid
   between markers in `docs/gallery.md`.
3. **MkDocs** builds a static site — no live API calls on GitHub Pages.

## Generate and preview visuals

The plot scripts under `tools/plots/` generate the gallery assets. They produce:

- a **PNG** in `docs/assets/gallery/`
- a **YAML** sidecar with metadata such as title, parameters, and script path

Common examples:

```bash
python tools/plots/tyre_strategy.py \
  --year 2025 \
  --event "Italian Grand Prix" \
  --cache .fastf1-cache

python tools/plots/drs_effectiveness.py \
  --year 2025 \
  --event "Italian Grand Prix" \
  --session R \
  --driver VER \
  --cache .fastf1-cache

python tools/generate_gallery.py
mkdocs serve
```

> The first run may download data; subsequent runs are much faster when you keep a local FastF1 cache.

## Requirements

- **Python:** 3.13+
- **FastF1:** used for race/session data access
- **Matplotlib:** chart rendering
- **PyYAML:** metadata sidecars for gallery generation
- **MkDocs + Material + mkdocstrings:** for docs and API pages

## Caching
- FastF1 cache (default `.fastf1-cache/`) speeds up runs; feel free to delete to refresh data.
- Scripts create the cache directory if missing.

## Consistent look & colors
- `fastf1_analytics.plotting` centralizes:
  - `apply_style`, `savefig`
  - `get_team_color`, `get_compound_color`, `lighten_color`

## Troubleshooting

- **Cache issues:** delete `.fastf1-cache` and rerun the plot script
- **Gallery not updating:** rerun `python tools/generate_gallery.py`
- **Docs preview not showing changes:** run `mkdocs build` or `mkdocs serve`
- **Plot script errors:** verify the event name, session flag, and cache path

## Quality checks

Run locally to match the repo's checks:

```bash
ruff check .
black --check .
pytest -q
```

## Reproduce a tile
```bash
python tools/plots/tyre_strategy.py --year 2025 --event "Italian Grand Prix" --cache .fastf1-cache
python tools/generate_gallery.py
mkdocs serve
```