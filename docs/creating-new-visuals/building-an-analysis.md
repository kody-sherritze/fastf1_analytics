# Building an Analysis

The plot scripts in `tools/plots/` are orchestration layers. They define how a run is configured; the chart modules contain the reusable calculation and rendering logic for a visual.

## Script responsibilities

A plot script generally defines command-line arguments with `argparse`, loads the requested session, constructs a parameter object, chooses an output filename, calls a chart builder, and records the run in a YAML sidecar.

For example, `tools/plots/tyre_strategy.py` converts a comma-separated driver order into a list when needed, loads a race session, creates `TyreStrategyParams`, and calls `build_tyre_strategy()`.

## Analysis responsibilities

Analysis code should make the transformation from FastF1 data to chart inputs inspectable. Existing visuals select race laps or telemetry, remove unusable observations, group by driver or compound, calculate aggregates such as medians or cumulative totals, and align values to a common distance or lap index before comparison.

The exact rules belong near the chart they support. DRS effectiveness needs telemetry alignment and threshold parameters, while tyre strategy needs stint and compound information. These are different analytical contracts even though both produce a Matplotlib figure.

## Parameters are part of the result

Parameters determine what data is included and how it is interpreted. Existing scripts expose choices such as event, session, driver, minimum laps, aggregation method, point count, and thresholds. Keep those choices explicit and write them to the sidecar so a visual can be understood and reproduced.

## Keep orchestration thin

When logic is useful beyond one command-line entry point, place it in `src/fastf1_analytics/charts/` or a shared utility rather than duplicating it in a script. The script should answer "what run should happen?"; the chart module should answer "how is this analysis calculated and drawn?"
