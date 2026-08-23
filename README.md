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

This project is organized across a few complementary layers:

- **Gallery:** a portfolio-style grid of the finished visual outputs
  - https://kody-sherritze.github.io/fastf1_analytics/gallery/
- **Featured analyses:** deeper walkthroughs of the strongest race narratives and findings
  - https://kody-sherritze.github.io/fastf1_analytics/case-studies/
- **How it works:** the project pipeline, caching strategy, and reproduction workflow
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
├─ README.md                       # repo landing page
└─ .github/                        # CI and automation config
```

This project is designed to keep the outputs reproducible, the docs easy to navigate, and the visuals suitable for technical discussion and portfolio presentation.

> Built on [FastF1](https://docs.fastf1.dev/)