# 2025 Italian Grand Prix - Tyre Strategy

Stints and compounds by driver

![2025 Italian Grand Prix - Tyre Strategy](../assets/gallery/italian_grand_prix_2025_tyre_strategy.png){ loading=lazy }

## Why this matters

This chart maps the strategy decisions across the field at the 2025 Italian Grand Prix by showing stint length and compound usage for each driver. It is a strong example of how race strategy can be made readable without losing the underlying performance story.

This visual addresses a practical decision: which strategies were most competitive, and how did compound choices and stint timing separate the field? It turns a complex strategy problem into a clear comparison across drivers and tyre phases.

## What the chart shows

- The chart makes the main strategy patterns visible at a glance, including who chose medium-first or hard-first sequences.
- Top-performing strategy arcs are separated from more conservative or reactive choices.
- The structure makes it easy to compare how compound use evolved throughout the race and which drivers held the strongest late-race tyre life.

## Data and method

- Data source: FastF1 race telemetry and stint data for the 2025 Italian Grand Prix
- Session: race data
- Plot type: tyre strategy timeline by driver
- Filtering/selection: clean race sessions and visible stint transitions by compound
- Key calculations: stint lengths, compound changes, and compound sequencing throughout the race

## Technical implementation

This was built in Python using the project's reusable tooling:

- FastF1 for session loading and stint data access
- custom chart helpers and consistent race styling
- Matplotlib for compact strategy timeline rendering and driver annotations
- YAML metadata and the gallery generator to keep the visual workflow reproducible and easy to review

## Key findings

The strategy chart demonstrates the practical value of multi-layered race analysis: the key story is not only who used which tyre, but how that decision interacted with race pace and degradation timing. That makes this a strong portfolio example because it combines data engineering, strategic analysis, and plotting in one story.

This is the part of the analysis that matters most from a portfolio standpoint: the visual shows a decision-making narrative that is easy to interpret while still grounded in race data.

## Reproduce this chart

```bash
python tools/plots/tyre_strategy.py --year 2025 --event "Italian Grand Prix" --cache .fastf1-cache
```

## Skills demonstrated

- strategy analysis
- race-simulation interpretation
- visual storytelling with timing data
- automated chart generation

## Related examples

- [Gallery](../gallery.md)
- [2025 Italian Grand Prix - Tyre lap times (clean race laps)](./italian-grand-prix-2025-tyre-lap-times-clean-race-laps.md)
- [Creating New Visuals](../creating-new-visuals/index.md)
