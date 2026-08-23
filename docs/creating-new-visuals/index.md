# Creating New Visuals

This section explains how FastF1 Analytics turns motorsport data into a published visual. It is for readers who are comfortable with Python and the project dependencies and want to understand how the pieces fit together.

The project separates data access, analysis, chart rendering, and documentation output. A typical run follows this path:

```mermaid
flowchart LR
    A[Plot script] --> B[Session loader]
    B --> C[FastF1 session]
    C --> D[Analysis and filtering]
    D --> E[Chart builder]
    E --> F[PNG and YAML sidecar]
    F --> G[Gallery or case study]
```

## The main boundaries

- `src/fastf1_analytics/session_loader.py` provides a wrapper around FastF1 session retrieval and local caching.
- `src/fastf1_analytics/utils.py` contains small data-shape helpers shared by analyses.
- `src/fastf1_analytics/plotting.py` centralizes styles, colors, formatting, and figure output.
- `src/fastf1_analytics/charts/` contains reusable chart builders.
- `tools/plots/` contains executable analysis scripts that parse arguments, load data, choose parameters, call a chart builder, and write gallery metadata.
- `tools/generate_gallery.py` converts YAML sidecars into the generated gallery grid.
- `tools/generate_case_studies.py` uses the same sidecars to create narrative case-study pages.

The [API reference](../api/reference/fastf1_analytics.md) documents callable interfaces; the [case studies](../case-studies/index.md) show the analytical stories produced by them.

## A practical sequence

1. Decide what question the visual should answer and which FastF1 session contains the required data.
2. Load the session through `load_session()` so cache behavior is consistent.
3. Filter and transform the session data into the comparison the chart needs.
4. Put reusable rendering logic in a chart builder and keep run-specific choices in the plot script.
5. Apply the shared plotting helpers and save the image through `savefig()`.
6. Write a YAML sidecar containing the title, image path, source script, callable, parameters, and tags.
7. Regenerate the gallery or case studies, then build the MkDocs site.
