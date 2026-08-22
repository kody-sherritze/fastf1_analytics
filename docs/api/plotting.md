# API: plotting

Location in repository: src/fastf1_analytics/plotting.py

This module contains presentation helpers and matplotlib utilities used across the plot builders in the charts/ package. It provides a small, stable surface for code that needs to reproduce the project's look-and-feel without importing chart-specific code.

Public helpers
- apply_style(): configure matplotlib rcparams and the project's theme
- get_team_color(team_name_or_code): return the RGB/hex color for a team
- get_compound_color(compound_name): color for tyre compound names (Soft/Medium/Hard)
- savefig(path, **kwargs): save a figure using the project's defaults

Notes on structure
- The chart builders live under src/fastf1_analytics/charts/ and import these helpers (e.g. charts/tyre_strategy.py imports apply_style and get_team_color).
- Prefer apply_style() at the start of scripts or notebooks so all figures are consistent with the gallery output.

Examples
```python
# simple usage in a script or notebook
from fastf1_analytics.plotting import apply_style, get_team_color, savefig

apply_style()
print(get_team_color("Ferrari"))

# use when building a custom figure
fig, ax = plt.subplots()
# ... draw on ax ...
savefig("my_plot.png", dpi=200)
```