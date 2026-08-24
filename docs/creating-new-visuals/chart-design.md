# Designing and Customizing Charts

Chart modules under `src/fastf1_analytics/charts/` combine analysis-specific drawing with shared presentation helpers. This keeps the gallery visually consistent while allowing each visual to use the chart form that fits its question.

## Shared plotting helpers

`fastf1_analytics.plotting` provides the common layer:

- `apply_style()` configures FastF1's Matplotlib styling and readability defaults.
- `get_team_color()` resolves team colors and includes fallbacks for aliases and historic names.
- `get_driver_color()` resolves a driver's team color when a session is available.
- `get_compound_color()` provides consistent tyre compound colors.
- `lighten_color()` creates a secondary variant of an existing color.
- `fmt_laptime_seconds()` and `seconds_formatter()` format lap-time values for display.
- `savefig()` creates the destination directory, applies a tight layout where possible, and saves at the requested DPI.

Use these helpers instead of configuring each chart independently. A new visual should look like it belongs beside the existing gallery output without requiring a new color or export convention.

## Existing chart forms

- `tyre_strategy` draws stint and compound timelines by driver.
- `tyre_performance` compares lap-time distributions by compound and driver.
- `time_in_first` shows cumulative time leading.
- `drs_effectiveness` compares distance-aligned speed traces and identifies braking behavior.

These modules share style and export helpers, but they do not force unrelated analyses into one generic chart abstraction.

## Customization decisions

Expose customization when it changes the analytical reading of the chart: ordering, thresholds, aggregation, annotations, DPI, or title. Keep purely internal Matplotlib details inside the chart builder unless callers genuinely need to control them.
