# Project Structure

FastF1 Analytics separates reusable Python code from runnable analysis scripts, generated documentation content, and tests. Knowing which area owns a decision makes the project easier to read and keeps changes in the right place.

## Repository map

```text
fastf1_analytics/
├── src/fastf1_analytics/
│   ├── charts/                  reusable chart builders
│   ├── plotting.py              shared styles, colors, and figure output
│   ├── session_loader.py        FastF1 session loading and cache setup
│   ├── utils.py                 small shared data helpers
│   └── cli.py                   package command-line example
├── tools/
│   ├── plots/                   runnable analysis scripts
│   ├── generate_gallery.py      builds the gallery page from sidecars
│   └── generate_case_studies.py builds case-study pages from sidecars
├── docs/
│   ├── assets/gallery/          rendered PNGs and YAML sidecars
│   ├── case-studies/            narrative visual walkthroughs
│   ├── creating-new-visuals/    architectural and procedural guide
│   ├── api/                     API documentation pages
│   ├── gallery/                 gallery lander and generated chart index
│   └── index.md                 documentation landing page
├── tests/                       automated checks
├── mkdocs.yaml                  site configuration and navigation
├── pyproject.toml               package, dependency, and tool settings
├── .pre-commit-config.yaml      local formatting and test hooks
└── README.md                    repository landing page
```

## Where behavior belongs

### `src/fastf1_analytics/`

This is the reusable library layer. `session_loader.py` owns the common FastF1 loading boundary, `plotting.py` owns shared visual conventions, and `charts/` owns analysis-specific chart builders. Code here should be usable by more than one execution path where practical.

### `tools/plots/`

These scripts are runnable entry points rather than the reusable library itself. They parse command-line arguments, select a session and analysis parameters, call library code, and write a PNG plus YAML sidecar. Run-specific decisions such as the event, driver, threshold, or output directory belong here.

### `docs/`

The documentation source contains both hand-authored pages and generated content. The pages in `creating-new-visuals/` explain the system; case studies explain individual analytical results; API pages expose the reusable modules.

### `tests/`

Tests cover the boundaries between these layers: loading behavior, shared helpers, chart builders, orchestration scripts, metadata rendering, case-study generation, and public imports. They are also examples of the intended contracts for code that may be unfamiliar from its implementation alone.

## Source files and generated files

The usual source-of-truth sequence is:

1. A script in `tools/plots/` runs library code from `src/fastf1_analytics/`.
2. The run writes a PNG and YAML sidecar under `docs/assets/gallery/`.
3. `tools/generate_gallery.py` uses the sidecars to update the marked generated block in `docs/gallery/charts.md`.
4. `tools/generate_case_studies.py` uses the same sidecars and template to write case-study pages.
5. MkDocs reads the Markdown source and writes the built static site under `site/`.

The Python modules, plot scripts, sidecars, Markdown templates, and documentation pages are source inputs. The gallery block, generated case-study pages, and `site/` output are derived from those inputs. When generated content is wrong, fix the relevant source or generator and regenerate it rather than making an isolated change to the output.

## Configuration boundaries

- `pyproject.toml` defines package metadata, dependencies, and Black, Ruff, and mypy settings.
- `.pre-commit-config.yaml` defines local commit and pre-push hooks.
- `mkdocs.yaml` defines the documentation theme, plugins, extensions, and navigation.
- `.github/workflows/` defines the automated CI and documentation workflows.

These files describe how the project is packaged, checked, and published; they do not contain the chart calculations themselves.
