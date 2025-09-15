# FastF1 Portfolio

[![CI](https://github.com/ksherr0/fastf1_portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/ksherr0/fastf1_portfolio/actions/workflows/ci.yml)
[![Docs](https://github.com/ksherr0/fastf1_portfolio/actions/workflows/docs.yml/badge.svg)](https://ksherr0.github.io/fastf1_portfolio/)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fksherr0.github.io%2Ffastf1_portfolio%2F)](https://ksherr0.github.io/fastf1_portfolio/)
[![Made with: Material for MkDocs](https://img.shields.io/badge/docs-Material%20for%20MkDocs-000?logo=materialformkdocs)](https://squidfunk.github.io/mkdocs-material/)

![Python](https://img.shields.io/badge/python-3.13%2B-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-0055A4?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)


Analyses and reusable helpers built on [FastF1](https://docs.fastf1.dev/) to showcase race strategy, telemetry, and season-long insights.

- [CI](https://github.com/ksherr0/fastf1_portfolio/actions/workflows/ci.yml)
- [Docs (GitHub Pages)](https://ksherr0.github.io/fastf1_portfolio/)

## Table of Contents:
- [Quickstart](#quickstart)
- [Repository Directory](#repository-directory)
- [Docs Overview (GitHub Pages)](#docs-overview-github-pages)
- [Requirements](#requirements)
- [Install](#install)
- [Pre-checks (what CI runs)](#pre-checks-what-ci-runs)
- [Generate the Plots (CLI)](#generate-the-plots-cli)
    - [Tyre Strategy](#tyre-strategy)
    - [DRS Effectiveness](#drs-effectiveness)
    - [Drivers' Phampionship Points](#drivers-championship-points)
    - [Common Flags](#common-flags)
    - [Update the Gallery Page](#update-the-gallery-page)
    - [Where to find Outputs](#where-to-find-outputs)
- [Programmatic Usage (quick examples)](#programmatic-usage-quick-examples)
- [Troubleshooting & Tips](#troubleshooting--tips)

## Quickstart
```bash
# 1) Clone & install (Python 3.13+)
git clone https://github.com/ksherr0/fastf1_portfolio.git
cd fastf1_portfolio

# 2) Create Virual Environment (optional, but recommended)
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate 
# macOS/Linux
source .venv/bin/activate

# 3) Install project dependencies
pip install -e . # run pip install -e .[dev] for optional dependencies

# 4) Generate a plot (writes PNG+YAML to docs/assets/gallery/)
python tools/plots/tyre_strategy.py --year 2025 --event "Italian Grand Prix" --cache .fastf1-cache

# 5) Update gallery and preview docs
python tools/generate_gallery.py
mkdocs serve  # open http://127.0.0.1:8000
```

## Repository Directory
```text
fastf1_portfolio/
├─ src/fastf1_portfolio/           # library code
│  ├─ charts/                      # chart builders (Matplotlib)
│  ├─ plotting.py                  # style/colors/helpers
│  └─ session_loader.py            # load_session()
├─ tools/
│  ├─ plots/                       # CLI plot scripts (3 examples)
│  └─ generate_gallery.py          # rebuilds docs/gallery.md from YAML sidecars
├─ docs/                           # GitHub Pages (MkDocs)
└─ tests/                          # minimal tests used by CI
```

## Docs Overview (GitHub Pages)
If you’re not sure where something lives in the docs, use this map:
- **Home (Quickstart):** install, run 1-2 plots, rebuild gallery, preview site
    - https://ksherr0.github.io/fastf1_portfolio/
- **Gallery:** all pre-rendered visuals; each card shows the script + params used
    - https://ksherr0.github.io/fastf1_portfolio/gallery/
- **How it works:** pipeline (plot script → PNG+YAML → gallery generator → MkDocs), caching notes
    - https://ksherr0.github.io/fastf1_portfolio/how-it-works/
- **API → Session Loader:** `load_session(year, gp, session, *, cache)`
    - https://ksherr0.github.io/fastf1_portfolio/api/session_loader/
- **API → Plotting Helpers:** `apply_style, get_team_color, get_compound_color, savefig`, etc
    - https://ksherr0.github.io/fastf1_portfolio/api/plotting/
- **Reference (module index):** top-level functions in `fastf1_portfolio`
    - https://ksherr0.github.io/fastf1_portfolio/reference/fastf1_portfolio/

## Requirements
- **Python:** 3.13+
- Works on all operating systems

## Install
```bash
git clone https://github.com/ksherr0/fastf1_portfolio.git
cd fastf1_portfolio
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate 
# macOS/Linux
source .venv/bin/activate

# editable install with dev extras (linters, tests, docs)
pip install -e .[dev]
```

## Pre-checks (what CI runs)
Run locally to match CI behavior:
```bash
ruff check .
black --check .
pytest -q
```

## Generate the Plots (CLI)
Example plot scripts live in `tools/plots/`. They write:
- a **PNG** to `docs/assets/gallery/`
- a **YAML** sidecar with metadata (title, params, code path)
> Tip: the first run may download data; use `--cache .fastf1-cache` for speed on subsequent runs.

### Tyre Strategy
```bash
python tools/plots/tyre_strategy.py \
  --year 2025 \
  --event "Italian Grand Prix" \
  --cache .fastf1-cache
```
#### Plot Params
- `--event` (e.g., "Monaco" or "Italian Grand Prix")
- `--driver-order` = `results`|`alpha`|comma list (`VER,LEC,HAM`)

### DRS Effectiveness
```bash
python tools/plots/drs_effectiveness.py \
  --year 2025 \
  --gp "Italian Grand Prix" \
  --session R \
  --driver VER \
  --cache .fastf1-cache
```
#### Plot Params
- `--driver` (3-letter code), `--session` (default `R`)
- `--n-points`,`--accel-threshold-kmh-s`,`--sustain-sec`

### Drivers' Championship Points
```bash
python tools/plots/driver_championship.py \
  --year 2024 \
  --include-sprints \
  --color-variant secondary \
  --min-total-points 0 \
  --cache .fastf1-cache
```
#### Plot Params
- `--include-sprints` (include sprint points)
- `--color-variant`=`primary`|`secondary`
- `--min-total-points`

### Common Flags
- `--cache PATH`              # FastF1 cache dir
- `--outdir DIR`              # Output for PNG/YAML
- `--title TEXT`              # Override default title
- `--dpi INT`                 # image DPI

## Update the Gallery Page
After generating/refreshing plots:
```bash
python tools/generate_gallery.py
```
This reads all YAML sidecars and **rewrites the cards** in `docs/gallery.md` between the `AUTO-GALLERY` markers.

## Where to Find Outputs
- Images: `docs/assets/gallery/*.png`
- YAML sidecars: `docs/assets/gallery/*.yml`
- Docs preview: `mkdocs serve` → open http://127.0.0.1:8000

## Programmatic Usage (quick examples)
Short notebook/script examples that mirror the CLI outputs.
1) Load a session & apply style
```python
from fastf1_portfolio import apply_style
from fastf1_portfolio.session_loader import load_session

apply_style()
session = load_session(2024, "Monaco", "R", cache=".fastf1-cache")
print(session.event["EventName"], session.name)  # e.g., "Monaco Grand Prix R"
```
2) Build a chart in Python (tyre strategy)
```python
from fastf1_portfolio.session_loader import load_session
from fastf1_portfolio.charts.tyre_strategy import (
    TyreStrategyParams, build_tyre_strategy
)

s = load_session(2025, "Italian Grand Prix", "R", cache=".fastf1-cache")
params = TyreStrategyParams(driver_order=["VER", "LEC", "HAM"], dpi=220)
fig, ax = build_tyre_strategy(
    s,
    params=params,
    out_path="docs/assets/gallery/italian_grand_prix_2025_tyre_strategy.png"
)
```

## Troubleshooting & Tips
- **Cache:** pass `--cache .fastf1-cache` (or another path). Delete the folder to refresh data.
- **Gallery not updating:** re-run `python tools/generate_gallery.py` after creating PNG/YAML.
- **Docs:** `mkdocs serve` to preview, `mkdocs build` for the static site.
- **Style:** call `apply_style()` before plotting in custom code.

> Built on [FastF1](https://docs.fastf1.dev/)