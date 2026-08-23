# 2025 Italian Grand Prix - DRS effect on main straight (VER)

Median speed traces along main straight (DRS ON/OFF)

![2025 Italian Grand Prix - DRS effect on main straight (VER)](../assets/gallery/italian_grand_prix_2025_drs_effect_VER.png){ loading=lazy }

## Why this matters

Median speed traces along main straight (DRS ON/OFF)

This visual addresses a practical decision: what stands out when comparing the selected metric across the field?. It turns raw telemetry and timing data into a clear, decision-ready view that helps explain what happened and why it mattered.

## What the chart shows

- Median speed traces along main straight (DRS ON/OFF) highlights the main pattern across the drs, speed comparison.
- The chart helps explain how the selected metric changes over the event or across the field.
- The result is useful for identifying performance gaps, strategy trade-offs, or timing differences.

## Data and method

- Data source: FastF1 session data for 2025 Italian Grand Prix
- Session: R
- Plot type: race analysis chart
- Filtering/selection: selected session filters and relevant race laps
- Key calculations: derived from lap-level and telemetry data for the selected comparison

## Technical implementation

This was built in Python using the project's reusable tooling:

- FastF1 for session loading and telemetry access
- custom chart helpers from the project library
- Matplotlib for the final rendering and styling
- YAML metadata and gallery generation to keep the output reproducible and easy to review

## Key findings

The main takeaway is that the chart makes the most important comparison immediately visible, which helps explain the analytical story behind the result.

This is the part of the analysis that matters most from a portfolio standpoint: the visual is not just a chart, it shows a decision point, a trend, or a strong signal supported by the underlying race data.

## Reproduce this chart

```bash
python tools/plots/drs_effectiveness.py --year 2025 --event "Italian Grand Prix" --session R --driver VER --n-points 200 --accel-threshold-kmh-s -8.0 --sustain-sec 0.3 --cache .fastf1-cache
```

## Skills demonstrated

- telemetry analysis
- Python plotting
- reproducible reporting
- visual storytelling

## Related examples

- [Creating New Visuals](../creating-new-visuals/index.md)
- [Gallery](../gallery.md)
