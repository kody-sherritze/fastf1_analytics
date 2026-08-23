# Plotting Helpers

The `plotting` module contains the shared presentation layer used by the chart builders. It keeps styles, domain colors, formatting, and figure output consistent without requiring callers to import chart-specific code.

## Typical usage

```python
import matplotlib.pyplot as plt

from fastf1_analytics.plotting import apply_style, get_team_color, savefig

apply_style()
print(get_team_color("Ferrari"))

fig, ax = plt.subplots()
# Draw the analysis-specific content on ax.
savefig(fig, "my_plot.png", dpi=200)
```

The public helpers fall into four groups:

- **Style:** `apply_style()` configures FastF1's Matplotlib theme and readability defaults.
- **Colors:** `get_team_color()`, `get_driver_color()`, `get_compound_color()`, and `lighten_color()` provide consistent domain colors and variants.
- **Formatting:** `fmt_laptime_seconds()` and `seconds_formatter()` format lap-time values.
- **Output:** `savefig()` creates the destination directory, applies a tight layout where possible, and saves the figure.

::: fastf1_analytics.plotting