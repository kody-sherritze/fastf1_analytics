# FastF1 Portfolio

High-quality analyses and reusable helpers built on [FastF1](https://docs.fastf1.dev/).

## Quickstart

```bash
pip install -e .[dev]
# Generate gallery tiles (PNG + YAML sidecar)
python tools/plots/tyre_strategy.py --year 2025 --event "Italian Grand Prix" --cache .fastf1-cache
python tools/plots/driver_championship.py --year 2024 --include-sprints --cache .fastf1-cache

# Rebuild the Gallery page from sidecars
python tools/generate_gallery.py

# Preview the site
mkdocs serve
```

See the **[Gallery](gallery.md)** for the latest visuals.