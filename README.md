# FastF1 Analytics

[![CI](https://github.com/kody-sherritze/fastf1_analytics/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/kody-sherritze/fastf1_analytics/actions/workflows/ci.yaml)
[![Docs](https://github.com/kody-sherritze/fastf1_analytics/actions/workflows/docs.yaml/badge.svg?branch=main)](https://kody-sherritze.github.io/fastf1_analytics/)

This project combines reusable FastF1 analysis helpers with publication-quality visual outputs for race strategy, telemetry, and season-long comparisons. It is built to be reproducible, easy to extend, and useful as both a technical project and a portfolio of analysis work.

## Overview

The project is organized around a simple workflow:

1. Load session data with the project helper layer
2. Build chart logic in the reusable library under `src/fastf1_analytics/`
3. Render PNG outputs into `docs/assets/gallery/`
4. Use YAML metadata to drive the gallery page and narrative case studies
5. Publish the result via MkDocs

This keeps the analytical work reproducible and makes the final visuals easy to showcase in reports, documentation, or portfolio reviews.

## Quickstart

```bash
# Clone the project
git clone https://github.com/kody-sherritze/fastf1_analytics.git
cd fastf1_analytics

# Create a virtual environment (recommended)
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies and project in editable mode
pip install -e .[dev]

# Generate a chart into the gallery assets folder
python tools/plots/tyre_strategy.py --year 2025 --event "Italian Grand Prix" --cache .fastf1-cache

# Rebuild the gallery page from YAML metadata
python tools/generate_gallery.py

# Preview the site locally
mkdocs serve
```

## Documentation

This project has a few different documentation layers, each with a clear purpose:

- **Gallery:** a grid of finished visual outputs
- https://kody-sherritze.github.io/fastf1_analytics/gallery/

- **Case Studies:** walkthroughs of the most compelling findings and race narratives
- https://kody-sherritze.github.io/fastf1_analytics/case-studies/

- **How it works:** the pipeline, caching strategy, and project architecture
- https://kody-sherritze.github.io/fastf1_analytics/how-it-works/

- **API Reference:** generated docs for the reusable Python modules
- https://kody-sherritze.github.io/fastf1_analytics/reference/fastf1_analytics/

## Project Structure

```text
fastf1_analytics/
├─ src/fastf1_analytics/           # reusable library code
│  ├─ charts/                      # chart builders (Matplotlib)
│  ├─ plotting.py                  # theme and color helpers
│  └─ session_loader.py            # FastF1 session loader
├─ tools/
│  ├─ plots/                       # CLI plot scripts
│  ├─ generate_gallery.py          # rebuilds docs/gallery.md from YAML sidecars
│  └─ generate_case_studies.py     # generates narrative case-study stubs
├─ docs/
│  ├─ assets/gallery/              # PNG outputs and YAML metadata
│  ├─ case-studies/                # portfolio-style narrative pages
│  ├─ api/                         # API docs pages
│  ├─ gallery.md                   # generated gallery page
│  ├─ how-it-works.md              # project pipeline and notes
│  └─ index.md                     # docs landing page
├─ tests/                          # project checks used by CI
├─ mkdocs.yaml                     # MkDocs config and navigation
├─ pyproject.toml                  # packaging and dependency configuration
├─ README.md                       # project landing page
└─ .github/                        # CI and automation config
```

## Generate and Preview Visuals

The plot scripts under `tools/plots/` generate the gallery assets. They write:

- a **PNG** file into `docs/assets/gallery/`
- a **YAML** sidecar with metadata such as title, params, and script path

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
- **Matplotlib:** final chart rendering
- **PyYAML:** metadata sidecars for gallery generation
- **MkDocs + Material + mkdocstrings:** for documentation and API pages

## Troubleshooting

- **Cache issues:** delete `.fastf1-cache` and rerun the plot script
- **Gallery not updating:** rerun `python tools/generate_gallery.py`
- **Docs preview not showing changes:** run `mkdocs build` or `mkdocs serve`
- **Plot script errors:** verify the event name, session flag, and cache path

## CI and Quality Checks

Run locally to match the repo’s quality gates:

```bash
ruff check .
black --check .
pytest -q
```

This project is designed to keep the outputs reproducible, the docs easy to navigate, and the visuals suitable for technical discussion and portfolio presentation.

> Built on [FastF1](https://docs.fastf1.dev/)