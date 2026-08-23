# Style and Pre-commit

The project uses a small set of automated tools to keep Python formatting, linting, and basic behavior consistent. These choices are configured in `pyproject.toml` and `.pre-commit-config.yaml`.

## Formatting choices

Black is the formatter for Python files. The project sets:

- a maximum line length of 100 characters; and
- Python 3.13 as the formatting target.

Ruff is the linter. Its configured line length is also 100 characters, so Ruff and Black agree on the primary width convention. Ruff runs with autofix through pre-commit, but the hook exits non-zero when it changes a file. This makes the required follow-up visible: review the edits and run the checks again.

## Type checking

Mypy is configured for Python 3.13 in strict mode. FastF1 and its submodules are exempt from missing-import errors because they do not provide complete type information to this project. Mypy is available as part of the development dependencies, but it is not currently a pre-commit hook or CI command.

That distinction matters when reading the repository's quality policy: Ruff, Black, and pytest are actively automated; mypy is configured for explicit use but is not part of the default automated checks.

## Pre-commit hooks

Install the hooks once after installing the development dependencies:

```bash
pre-commit install
pre-commit install --hook-type pre-push
```

The configured hooks are:

- **Ruff:** runs on commit, applies safe fixes, and reports a failure if fixes were needed.
- **Black:** runs on commit and formats Python files.
- **pytest:** runs on pre-push using `.venv/Scripts/python.exe -m pytest -q` and does not receive filenames from pre-commit; it runs the complete test suite.

On Windows PowerShell, the configured pytest hook expects the project virtual environment at `.venv\Scripts\python.exe`. If the environment has a different location, run `pytest -q` manually or adjust the local hook configuration for that machine.

## Manual checks and CI

The equivalent local checks are:

```bash
ruff check .
black --check .
pytest -q
```

CI runs those same three commands after installing the development dependencies. `mkdocs build` validates the documentation site separately; it is not part of the current CI workflow.

## Style beyond formatting

The plotting code also has a domain-specific style layer. Use `apply_style()` for shared Matplotlib defaults, the team and compound color helpers for consistent semantics, and `savefig()` for predictable figure output. Python formatting tools handle source layout; these plotting helpers handle the visual language of the published charts.
