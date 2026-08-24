# API Overview

<div class="section-lander" markdown>

**The reusable layer behind every visual**

The API is where the project keeps its shared vocabulary: session loading, plotting style, formatting, and chart builders. Use this page to choose the right reference before opening generated signatures and module docs.

</div>

## Public entry points

The package exposes two convenience helpers at the top level:

```python
from fastf1_analytics import apply_style, load_session
```

Chart builders are imported from their individual modules. This keeps the chart surface explicit as the library grows:

```python
from fastf1_analytics.charts.tyre_strategy import (
    TyreStrategyParams,
    build_tyre_strategy,
)
```

## Choose a reference

<div class="grid cards" markdown>

- :material-download-circle-outline: **[Session Loader](session_loader.md)**

    FastF1 session retrieval and cache setup.

- :material-palette-outline: **[Plotting Helpers](plotting.md)**

    Shared styles, colors, formatting, and figure output.

- :material-chart-multiple: **[Chart Builders](chart-builders.md)**

    Available chart modules and their builder contracts.

- :material-code-braces: **[Package Reference](reference/fastf1_analytics.md)**

    Generated top-level package documentation from `mkdocstrings`.

</div>

The reference pages describe callable signatures and parameters. The architectural guide explains when each layer is used.
