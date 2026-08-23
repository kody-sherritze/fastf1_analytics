# Testing and Quality Checks

The project tests the boundaries where data is transformed, files are generated, and public imports are exposed.

## Focused tests

- `tests/test_session_loader.py` checks the session-loading wrapper and cache setup.
- `tests/test_chart_helpers.py` and `tests/test_helpers.py` cover shared calculations and utility behavior.
- `tests/test_chart_builders.py` exercises chart construction without requiring a full published run.
- `tests/test_orchestration.py` checks script-level analysis behavior, including skipped or empty events.
- `tests/test_gallery.py` verifies sidecar rendering, source URLs, and marker replacement.
- `tests/test_generate_case_studies.py` verifies parameter-to-command conversion, placeholder replacement, and URL-safe slugs.
- `tests/test_imports.py` checks the package's public entry points.

Run the test suite with:

```bash
pytest -q
```

## Static checks

The repository configures Ruff, Black, and strict mypy. The standard local checks are:

```bash
ruff check .
black --check .
pytest -q
mkdocs build
```

The plot scripts can also be exercised directly. These runs may download FastF1 data on the first attempt, so provide a persistent cache path when checking a visual repeatedly.

## What to verify for a new visual

Verify that the analysis handles empty or unavailable data deliberately, the output path is stable, the sidecar parameters match the actual run, and the generated gallery or case-study page builds successfully. The most useful test is usually the narrowest one that checks the new transformation or generator behavior without requiring a network request.
